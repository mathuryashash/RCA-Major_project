"""Counter differencing against measured monotonic time."""

import time


class CounterTracker:
    """Track monotonically increasing counters without assuming a cadence."""

    def __init__(self) -> None:
        self._last_mono: float | None = None
        self._last: dict[str, float] = {}

    def reset(self) -> None:
        self._last_mono = None
        self._last = {}

    def tick(
        self, counters: dict[str, float], now: float | None = None,
    ) -> tuple[int | None, dict[str, float | None]]:
        now = time.monotonic() if now is None else now
        if self._last_mono is None:
            self._last_mono = now
            self._last = dict(counters)
            return None, {name: None for name in counters}

        elapsed_ms = int(round((now - self._last_mono) * 1000))
        deltas = {
            name: (
                None if previous is None or value < previous or elapsed_ms <= 0
                else value - previous
            )
            for name, value in counters.items()
            for previous in (self._last.get(name),)
        }
        self._last_mono = now
        self._last = dict(counters)
        return elapsed_ms, deltas
