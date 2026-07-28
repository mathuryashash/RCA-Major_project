"""psutil-based system and process sampling."""

import ctypes
import os
from ctypes import wintypes

import psutil

SAMPLE_COLUMNS = (
    "elapsed_ms", "cpu_pct", "cpu_pct_max_core", "cpu_freq_mhz", "cpu_freq_ratio",
    "mem_pct", "mem_available_mb", "swap_pct", "swap_used_bytes", "swap_used_delta",
    "disk_read_bps", "disk_write_bps", "disk_busy_pct", "disk_free_pct",
    "net_sent_bps", "net_recv_bps", "process_count", "battery_pct",
    "battery_drain_rate", "power_plugged", "on_battery", "user_idle_sec",
    "foreground_app", "cpu_busy_s_delta", "mem_used_bytes",
    "disk_read_bytes_delta", "disk_write_bytes_delta",
    "gpu_util_pct", "gpu_mem_used_bytes", "gpu_temp_c",
)

#: Used to migrate an existing database when a channel is added.
SAMPLE_COLUMN_TYPES = {
    column: ("TEXT" if column == "foreground_app" else "REAL")
    for column in SAMPLE_COLUMNS
}

_IS_WINDOWS = hasattr(ctypes, "windll")

# NVML is the only temperature source available here: psutil's
# sensors_temperatures() returns None on Windows. It is optional -- a machine
# with no NVIDIA GPU simply records NULL for these channels.
from .logsetup import get_logger

_LOGGER = get_logger(__name__)

try:
    import pynvml

    pynvml.nvmlInit()
    _NVML_READY = pynvml.nvmlDeviceGetCount() > 0
    _LOGGER.info("NVML ready: %s device(s)", pynvml.nvmlDeviceGetCount())
except Exception as exc:  # noqa: BLE001 - absent driver, no GPU, or NVML load failure
    pynvml, _NVML_READY = None, False
    _LOGGER.warning("NVML unavailable, GPU channels will be NULL: %r", exc)

_GPU_ERROR_LOGGED = False

_NO_GPU = {"gpu_util_pct": None, "gpu_mem_used_bytes": None, "gpu_temp_c": None}


def _read_gpu() -> dict[str, float | None]:
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    return {
        "gpu_util_pct": float(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu),
        "gpu_mem_used_bytes": float(pynvml.nvmlDeviceGetMemoryInfo(handle).used),
        "gpu_temp_c": float(pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)),
    }


def sample_gpu() -> dict[str, float | None]:
    """GPU utilisation, memory and temperature, or NULLs where unavailable.

    On an Optimus laptop the discrete GPU powers down when idle. That
    invalidates the whole NVML context, not merely the device handle: the
    first sample succeeds and every later one fails with NVMLError_Unknown.
    Re-acquiring the handle is not enough, so a failure re-initialises NVML
    and retries once. Without this the GPU channels record a single value at
    startup and NULL forever afterwards.
    """
    if not _NVML_READY:
        return dict(_NO_GPU)
    try:
        return _read_gpu()
    except Exception:  # noqa: BLE001 - stale context after the GPU slept
        try:
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001 - already down; the re-init is what matters
            pass
        try:
            pynvml.nvmlInit()
            return _read_gpu()
        except Exception as exc:  # noqa: BLE001 - GPU genuinely asleep this tick
            global _GPU_ERROR_LOGGED
            if not _GPU_ERROR_LOGGED:
                # Logged once: a dGPU that sleeps often would fill the log.
                _LOGGER.warning("GPU sampling failed after NVML re-init: %r", exc)
                _GPU_ERROR_LOGGED = True
            return dict(_NO_GPU)


class _LASTINPUTINFO(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]


def user_idle_sec() -> float | None:
    if not _IS_WINDOWS:
        return None
    info = _LASTINPUTINFO(ctypes.sizeof(_LASTINPUTINFO))
    if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
        return None
    return max(0.0, (ctypes.windll.kernel32.GetTickCount() - info.dwTime) / 1000.0)


def foreground_app() -> str | None:
    """Return only the foreground executable name, never its window title."""
    if not _IS_WINDOWS:
        return None
    hwnd = ctypes.windll.user32.GetForegroundWindow()
    pid = wintypes.DWORD()
    if not hwnd or not ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid)):
        return None
    try:
        return psutil.Process(pid.value).name()
    except (psutil.Error, OSError):
        return None


def sample_system_raw() -> dict[str, dict[str, object]]:
    vm, swap, cpu_times = psutil.virtual_memory(), psutil.swap_memory(), psutil.cpu_times()
    disk, network = psutil.disk_io_counters(), psutil.net_io_counters()
    per_core = psutil.cpu_percent(percpu=True)
    try:
        frequency = psutil.cpu_freq()
    except (OSError, NotImplementedError):
        frequency = None
    try:
        battery = psutil.sensors_battery()
    except (OSError, NotImplementedError):
        battery = None
    try:
        drive = os.environ.get("SystemDrive", "C:") + "\\"
        free_percent = 100.0 - psutil.disk_usage(drive).percent
    except OSError:
        free_percent = None
    gauges = {
        "cpu_pct": psutil.cpu_percent(), "cpu_pct_max_core": max(per_core) if per_core else None,
        "cpu_freq_mhz": frequency.current if frequency else None,
        "cpu_freq_ratio": frequency.current / frequency.max if frequency and frequency.max else None,
        "mem_pct": vm.percent, "mem_available_mb": vm.available / 1024 / 1024,
        "mem_used_bytes": vm.used, "swap_pct": swap.percent, "swap_used_bytes": swap.used,
        "disk_free_pct": free_percent, "process_count": len(psutil.pids()),
        "battery_pct": battery.percent if battery else None,
        "power_plugged": int(battery.power_plugged) if battery else None,
        "on_battery": int(not battery.power_plugged) if battery else None,
        "user_idle_sec": user_idle_sec(), "foreground_app": foreground_app(),
    }
    return {"gauges": gauges, "counters": {
        "cpu_busy_s": cpu_times.user + cpu_times.system,
        "disk_read_bytes": float(disk.read_bytes) if disk else 0.0,
        "disk_write_bytes": float(disk.write_bytes) if disk else 0.0,
        "disk_busy_ms": float(disk.read_time + disk.write_time) if disk else 0.0,
        "net_sent_bytes": float(network.bytes_sent) if network else 0.0,
        "net_recv_bytes": float(network.bytes_recv) if network else 0.0,
        "swap_used_bytes": float(swap.used),
        "battery_pct_counter": float(battery.percent) if battery else 0.0,
    }}


def _rate(delta: float | None, elapsed_ms: int | None) -> float | None:
    return None if delta is None or not elapsed_ms else delta * 1000.0 / elapsed_ms


def build_sample_row(raw: dict, elapsed_ms: int | None, deltas: dict[str, float | None]) -> dict[str, object]:
    gauges = raw["gauges"]
    busy = deltas.get("disk_busy_ms")
    disk_busy = None if busy is None or not elapsed_ms else min(100.0, max(0.0, busy * 100.0 / elapsed_ms))
    battery_delta = deltas.get("battery_pct_counter")
    drain = None
    if battery_delta is not None and elapsed_ms and not gauges["power_plugged"]:
        drain = max(0.0, -battery_delta) * 3_600_000 / elapsed_ms
    return {
        "elapsed_ms": elapsed_ms, "cpu_pct": gauges["cpu_pct"], "cpu_pct_max_core": gauges["cpu_pct_max_core"],
        "cpu_freq_mhz": gauges["cpu_freq_mhz"], "cpu_freq_ratio": gauges["cpu_freq_ratio"],
        "mem_pct": gauges["mem_pct"], "mem_available_mb": gauges["mem_available_mb"],
        "swap_pct": gauges["swap_pct"], "swap_used_bytes": gauges["swap_used_bytes"],
        "swap_used_delta": deltas.get("swap_used_bytes"), "disk_read_bps": _rate(deltas.get("disk_read_bytes"), elapsed_ms),
        "disk_write_bps": _rate(deltas.get("disk_write_bytes"), elapsed_ms), "disk_busy_pct": disk_busy,
        "disk_free_pct": gauges["disk_free_pct"], "net_sent_bps": _rate(deltas.get("net_sent_bytes"), elapsed_ms),
        "net_recv_bps": _rate(deltas.get("net_recv_bytes"), elapsed_ms), "process_count": gauges["process_count"],
        "battery_pct": gauges["battery_pct"], "battery_drain_rate": drain, "power_plugged": gauges["power_plugged"],
        "on_battery": gauges["on_battery"], "user_idle_sec": gauges["user_idle_sec"],
        "foreground_app": gauges["foreground_app"], "cpu_busy_s_delta": deltas.get("cpu_busy_s"),
        "mem_used_bytes": gauges["mem_used_bytes"], "disk_read_bytes_delta": deltas.get("disk_read_bytes"),
        "disk_write_bytes_delta": deltas.get("disk_write_bytes"),
        **sample_gpu(),
    }


class ProcessSampler:
    """Sample the union of the largest CPU and RSS processes safely."""

    def __init__(self) -> None:
        self._previous: dict[tuple[int, float], tuple[float, int, int]] = {}
        self._cores = psutil.cpu_count() or 1

    def sample(self, top_n: int, elapsed_s: float | None) -> list[dict[str, object]]:
        current: dict[tuple[int, float], tuple[float, int, int]] = {}
        rows: list[dict[str, object]] = []
        has_baseline = bool(self._previous)
        for process in psutil.process_iter(["name", "create_time", "cpu_times", "memory_info", "io_counters"]):
            try:
                info, times, memory = process.info, process.info["cpu_times"], process.info["memory_info"]
                if times is None or memory is None or info["create_time"] is None:
                    continue
                key = (process.pid, info["create_time"])
                cpu_time = times.user + times.system
                io = info["io_counters"]
                read, write = (io.read_bytes, io.write_bytes) if io else (0, 0)
                current[key] = (cpu_time, read, write)
                if not has_baseline:
                    continue
                old = self._previous.get(key)
                cpu_delta, read_delta, write_delta = (0.0, 0, 0) if old is None else (
                    max(0.0, cpu_time - old[0]), max(0, read - old[1]), max(0, write - old[2]),
                )
                rows.append({"pid": process.pid, "create_time": info["create_time"], "name": info["name"],
                             "cpu_pct": cpu_delta * 100 / (elapsed_s * self._cores) if elapsed_s else 0.0,
                             "cpu_time_delta_s": cpu_delta, "rss": memory.rss,
                             "io_read_delta": read_delta, "io_write_delta": write_delta})
            except (psutil.Error, OSError, KeyError):
                continue
        self._previous = current
        chosen = {(row["pid"], row["create_time"]): row for row in (
            sorted(rows, key=lambda row: row["cpu_time_delta_s"], reverse=True)[:top_n] +
            sorted(rows, key=lambda row: row["rss"], reverse=True)[:top_n]
        )}
        return list(chosen.values())
