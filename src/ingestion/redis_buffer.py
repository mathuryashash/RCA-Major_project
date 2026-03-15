"""Redis Streams buffer for metric ingestion pipeline."""

import json
import logging
from typing import List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Redis availability flag
# ---------------------------------------------------------------------------
try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    redis = None  # type: ignore[assignment]
    REDIS_AVAILABLE = False
    logger.info("redis package not installed — RedisBuffer will operate as no-op")


class RedisBuffer:
    """Push / consume MetricPoint batches via a Redis Stream.

    Graceful degradation: if the ``redis`` package is not installed or the
    Redis server is unreachable, every public method is a no-op that logs a
    warning and returns an empty result.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        stream_name: str = "rca_metrics",
        consumer_group: str = "rca_pipeline",
    ):
        self.stream_name = stream_name
        self.consumer_group = consumer_group
        self._conn: "redis.Redis | None" = None  # type: ignore[name-defined]

        if not REDIS_AVAILABLE:
            logger.warning("redis package unavailable — RedisBuffer disabled")
            return

        try:
            self._conn = redis.Redis(host=host, port=port, decode_responses=True)
            self._conn.ping()
            # Ensure consumer group exists (MKSTREAM creates the stream if absent)
            try:
                self._conn.xgroup_create(
                    self.stream_name, self.consumer_group, id="0", mkstream=True
                )
            except redis.exceptions.ResponseError as exc:
                # "BUSYGROUP Consumer Group name already exists" is expected
                if "BUSYGROUP" not in str(exc):
                    raise
            logger.info(
                "RedisBuffer connected to %s:%s stream=%s group=%s",
                host,
                port,
                stream_name,
                consumer_group,
            )
        except Exception as exc:
            logger.warning("Redis connection failed (%s) — buffer disabled", exc)
            self._conn = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(self, metrics: List) -> int:
        """Push a batch of MetricPoint objects to the Redis Stream.

        Each metric is serialised as a JSON string in the ``data`` field.
        Returns the number of metrics successfully pushed (0 if unavailable).
        """
        if self._conn is None:
            logger.debug("RedisBuffer.push called but Redis unavailable — no-op")
            return 0

        count = 0
        for mp in metrics:
            try:
                payload = json.dumps(mp.to_dict())
                self._conn.xadd(self.stream_name, {"data": payload})
                count += 1
            except Exception as exc:
                logger.warning("Failed to push metric to Redis: %s", exc)
        return count

    def consume(
        self,
        consumer_name: str = "worker-1",
        batch_size: int = 100,
        block_ms: int = 1000,
    ) -> List:
        """Read up to *batch_size* messages from the consumer group.

        Returns a list of MetricPoint objects (empty if Redis is unavailable).
        """
        # Avoid circular import at module level
        from src.ingestion.metric_ingest import MetricPoint

        if self._conn is None:
            logger.debug("RedisBuffer.consume called but Redis unavailable — no-op")
            return []

        points: List[MetricPoint] = []
        try:
            results = self._conn.xreadgroup(
                groupname=self.consumer_group,
                consumername=consumer_name,
                streams={self.stream_name: ">"},
                count=batch_size,
                block=block_ms,
            )
            for _stream, messages in results or []:
                for msg_id, fields in messages:
                    try:
                        data = json.loads(fields["data"])
                        points.append(MetricPoint.from_dict(data))
                        # Acknowledge the message
                        self._conn.xack(self.stream_name, self.consumer_group, msg_id)
                    except Exception as exc:
                        logger.warning("Failed to parse message %s: %s", msg_id, exc)
        except Exception as exc:
            logger.warning("RedisBuffer.consume error: %s", exc)

        return points

    # ------------------------------------------------------------------
    def health_check(self) -> bool:
        """Return True if Redis is connected and responding."""
        if self._conn is None:
            return False
        try:
            return self._conn.ping()
        except Exception:
            return False
