"""The headless collection loop; it never invokes RCA detection."""

import ctypes
import time

from . import config, sampler, store
from .eventlog import EventLogReader
from .logsetup import get_logger
from .rates import CounterTracker
from .sampler import ProcessSampler

_LOGGER = get_logger(__name__)
_MUTEX_HANDLE = None
_ERROR_ALREADY_EXISTS = 183


def consent_granted(conn) -> bool:
    return store.get_meta(conn, "consent_granted", "0") == "1"


def grant_consent(conn) -> None:
    store.set_meta(conn, "consent_granted", "1")
    store.set_meta(conn, "consent_granted_at", str(int(time.time())))


def acquire_singleton() -> bool:
    global _MUTEX_HANDLE
    if not hasattr(ctypes, "windll"):
        return True
    _MUTEX_HANDLE = ctypes.windll.kernel32.CreateMutexW(None, True, config.MUTEX_NAME)
    return bool(_MUTEX_HANDLE) and ctypes.windll.kernel32.GetLastError() != _ERROR_ALREADY_EXISTS


#: Twenty consecutive failures is ten minutes at the sampling cadence. A
#: transient fault clears well inside that; anything still failing is not
#: going to be fixed by a twenty-first attempt.
MAX_CONSECUTIVE_FAILURES = 20


def request_stop() -> None:
    config.app_dir().mkdir(parents=True, exist_ok=True)
    config.stop_flag_path().touch()


class Collector:
    def __init__(self, conn, capture_messages: bool = False) -> None:
        self.conn, self.capture_messages = conn, capture_messages
        # swap in use is a gauge, not a counter: it falls as pages are freed.
        self._tracker = CounterTracker(signed={"swap_used_bytes"})
        self._processes = ProcessSampler()
        self._readers = [EventLogReader(channel) for channel in config.EVENT_CHANNELS]
        self._last_system_mono = self._last_process_mono = self._last_event_mono = self._last_purge_mono = None

    @staticmethod
    def should_burst(row: dict) -> bool:
        return any(((row.get(key) or 0.0) > threshold) for key, threshold in (
            ("cpu_pct", config.BURST_CPU_PCT), ("mem_pct", config.BURST_MEM_PCT), ("disk_busy_pct", config.BURST_DISK_BUSY_PCT),
        ))

    def tick_system(self, wall_now: float, mono_now: float) -> dict:
        if self._last_system_mono is not None and mono_now - self._last_system_mono > config.gap_threshold_s():
            self._tracker.reset()
        self._last_system_mono = mono_now
        raw = sampler.sample_system_raw()
        elapsed_ms, deltas = self._tracker.tick(raw["counters"], mono_now)
        row = sampler.build_sample_row(raw, elapsed_ms, deltas)
        store.insert_sample(self.conn, int(wall_now), row)
        return row

    def maybe_tick_processes(self, wall_now: float, mono_now: float, row: dict) -> int:
        cadence = config.PROCESS_BURST_CADENCE_S if self.should_burst(row) else config.PROCESS_CADENCE_S
        if self._last_process_mono is not None and mono_now - self._last_process_mono < cadence:
            return 0
        elapsed = None if self._last_process_mono is None else mono_now - self._last_process_mono
        self._last_process_mono = mono_now
        rows = self._processes.sample(config.TOP_N, elapsed)
        store.insert_proc_samples(self.conn, int(wall_now), rows)
        return len(rows)

    def maybe_poll_events(self, mono_now: float) -> int:
        if self._last_event_mono is not None and mono_now - self._last_event_mono < config.EVENT_POLL_S:
            return 0
        self._last_event_mono = mono_now
        total = 0
        for reader in self._readers:
            try:
                total += reader.read_new(self.conn, self.capture_messages)
            except Exception:
                _LOGGER.exception("Event Log collection failed for %s", reader.channel)
        return total

    def maybe_purge(self, wall_now: float, mono_now: float) -> None:
        if self._last_purge_mono is not None and mono_now - self._last_purge_mono < 86400:
            return
        self._last_purge_mono = mono_now
        store.purge_proc_samples(self.conn, int(wall_now) - config.PROC_RETENTION_DAYS * 86400)
        store.purge_events(self.conn, int(wall_now) - config.EVENT_RETENTION_DAYS * 86400)

    def run_once(self, now: float | None = None, mono_now: float | None = None) -> None:
        if not consent_granted(self.conn):
            raise PermissionError("Telemetry collection requires recorded consent")
        wall_now = time.time() if now is None else now
        mono_now = time.monotonic() if mono_now is None else mono_now
        row = self.tick_system(wall_now, mono_now)
        self.maybe_tick_processes(wall_now, mono_now, row)
        self.maybe_poll_events(mono_now)
        self.maybe_purge(wall_now, mono_now)

    def run_forever(self) -> None:
        """Sample until asked to stop, surviving anything a tick can throw.

        A tick failure is logged and the loop continues -- a transient WMI or
        Event Log error must not end a collection run that is building a
        baseline over hours. Failures are counted, and a run that is failing
        every single tick gives up rather than writing a log entry every
        thirty seconds forever: at that point something is wrong that
        retrying will not fix.
        """
        config.stop_flag_path().unlink(missing_ok=True)
        consecutive_failures = 0
        while not config.stop_flag_path().exists():
            started = time.monotonic()
            try:
                self.run_once()
                consecutive_failures = 0
            except Exception:
                consecutive_failures += 1
                _LOGGER.exception(
                    "Telemetry tick failed (%d in a row)", consecutive_failures
                )
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    _LOGGER.error(
                        "Giving up after %d consecutive failures; run "
                        "'status' and check this log.", consecutive_failures
                    )
                    return
            time.sleep(max(0.0, config.SYSTEM_CADENCE_S - (time.monotonic() - started)))
