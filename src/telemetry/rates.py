"""Counter differencing against measured monotonic time."""

import time


class CounterTracker:
    """Difference counters against measured monotonic time.

    Monotonic counters (bytes read, CPU seconds) only ever rise, so a decrease
    means the counter was reset and the delta is unknowable -- reported as None.

    Gauges named in ``signed`` legitimately rise and fall (swap in use, for
    example). Treating one as a monotonic counter yields None on every decrease,
    which downstream cleaning then drops as an unusable row: a gauge sampled
    this way silently destroys roughly half the history.
    """

    def __init__(self, signed: set[str] | None = None) -> None:
        self._last_mono: float | None = None
        self._last: dict[str, float] = {}
        self._signed = signed or set()

    def reset(self) -> None:
        self._last_mono = None
        self._last = {}

    def _delta(self, name: str, value: float, previous: float | None, elapsed_ms: int) -> float | None:
        if previous is None or elapsed_ms <= 0:
            return None
        if name in self._signed:
            return value - previous          # may be negative; that is the signal
        return None if value < previous else value - previous

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
            name: self._delta(name, value, self._last.get(name), elapsed_ms)
            for name, value in counters.items()
        }
        self._last_mono = now
        self._last = dict(counters)
        return elapsed_ms, deltas
