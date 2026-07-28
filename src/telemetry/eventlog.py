"""Windows Event Log ingestion with channel-scoped transactional watermarks."""

import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

from . import config, store
from .logsetup import get_logger
from .redaction import redact

try:
    import win32evtlog
except ImportError:  # pragma: no cover - platform dependency
    win32evtlog = None

_NS = {"e": "http://schemas.microsoft.com/win/2004/08/events/event"}
_LEVELS = {0: "Information", 1: "Critical", 2: "Error", 3: "Warning", 4: "Information", 5: "Verbose"}
_LOGGER = get_logger(__name__)


def watermark_key(channel: str) -> str:
    return f"evtlog_watermark_{channel.lower()}"


def _timestamp(value: str) -> int:
    value = value.replace("Z", "+00:00")
    if "." in value:
        prefix, _, fraction = value.partition(".")
        digits, _, offset = fraction.partition("+")
        value = f"{prefix}.{digits[:6]}+{offset}" if offset else f"{prefix}.{digits[:6]}"
    parsed = datetime.fromisoformat(value)
    return int((parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).timestamp())


def parse_event_xml(xml: str) -> dict[str, object] | None:
    try:
        root = ET.fromstring(xml)
        system = root.find("e:System", _NS)
        if system is None:
            return None
        provider, event_id = system.find("e:Provider", _NS), system.find("e:EventID", _NS)
        record_id, created, level = system.find("e:EventRecordID", _NS), system.find("e:TimeCreated", _NS), system.find("e:Level", _NS)
        if event_id is None or record_id is None or created is None:
            return None
        values = [item.text.strip() for item in root.findall("e:EventData/e:Data", _NS) if item.text and item.text.strip()]
        return {"ts": _timestamp(created.attrib["SystemTime"]), "record_id": int(record_id.text),
                "provider": provider.attrib.get("Name") if provider is not None else None,
                "event_id": int(event_id.text), "level": _LEVELS.get(int(level.text) if level is not None else 0, "Information"),
                "data": " | ".join(values)}
    except (ET.ParseError, KeyError, TypeError, ValueError):
        return None


def _allowed(record: dict[str, object]) -> bool:
    provider = str(record["provider"])
    if provider not in config.EVENT_ALLOWLIST:
        return False
    permitted_ids = config.EVENT_ALLOWLIST[provider]
    return permitted_ids is None or int(record["event_id"]) in permitted_ids


class EventLogReader:
    def __init__(self, channel: str) -> None:
        self.channel = channel

    def _query(self, watermark: int):
        return win32evtlog.EvtQuery(
            self.channel, win32evtlog.EvtQueryChannelPath | win32evtlog.EvtQueryForwardDirection,
            f"*[System[(EventRecordID > {watermark})]]", None,
        )

    def newest_record_id(self) -> int | None:
        if win32evtlog is None:
            return None
        handle = None
        try:
            handle = win32evtlog.EvtQuery(self.channel, win32evtlog.EvtQueryChannelPath | win32evtlog.EvtQueryReverseDirection, "*", None)
            records = win32evtlog.EvtNext(handle, 1)
            parsed = parse_event_xml(win32evtlog.EvtRender(records[0], win32evtlog.EvtRenderEventXml)) if records else None
            return int(parsed["record_id"]) if parsed else None
        except Exception:
            _LOGGER.exception("Could not determine newest Event Log record for %s", self.channel)
            return None
        finally:
            if handle:
                win32evtlog.EvtClose(handle)

    def _mark_invalid(self, conn, newest: int | None) -> None:
        now = int(time.time())
        last_seen = conn.execute("SELECT MAX(ts) FROM events WHERE channel = ?", (self.channel,)).fetchone()[0]
        conn.execute("BEGIN")
        try:
            store.set_meta(conn, watermark_key(self.channel), str(newest or 0))
            store.record_collection_gap(conn, self.channel, last_seen, now, now)
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise

    def read_new(self, conn, capture_messages: bool = False, limit: int = 500) -> int:
        if win32evtlog is None:
            return 0
        watermark = int(store.get_meta(conn, watermark_key(self.channel), "0"))
        newest = self.newest_record_id()
        if newest is not None and watermark > newest:
            self._mark_invalid(conn, newest)
            return 0
        handle = None
        try:
            handle = self._query(watermark)
            records: list[dict[str, object]] = []
            seen_highest = watermark
            while len(records) < limit:
                batch = win32evtlog.EvtNext(handle, 64)
                if not batch:
                    break
                for item in batch:
                    parsed = parse_event_xml(win32evtlog.EvtRender(item, win32evtlog.EvtRenderEventXml))
                    if parsed:
                        seen_highest = max(seen_highest, int(parsed["record_id"]))
                    if parsed and _allowed(parsed):
                        records.append(parsed)
                    if len(records) >= limit:
                        break
                if len(records) >= limit:
                    break
        except Exception:
            _LOGGER.exception("Could not read %s Event Log; keeping its watermark", self.channel)
            return 0
        finally:
            if handle:
                win32evtlog.EvtClose(handle)
        if not records:
            # The query may have contained only uninteresting events. Advance
            # the watermark to the newest known record so they are not parsed
            # again on every five-minute poll.
            if seen_highest > watermark:
                store.set_meta(conn, watermark_key(self.channel), str(seen_highest))
            return 0
        highest = seen_highest
        rows = [(record["ts"], self.channel, record["record_id"], record["provider"], record["event_id"], record["level"],
                 redact(str(record["data"])) if capture_messages else None) for record in records]
        conn.execute("BEGIN")
        try:
            conn.executemany("INSERT OR IGNORE INTO events (ts, channel, record_id, provider, event_id, level, message_redacted) VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
            store.set_meta(conn, watermark_key(self.channel), str(highest))
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        return len(rows)
