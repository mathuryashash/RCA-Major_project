"""Task Scheduler integration for the collector."""

import subprocess
import sys
from pathlib import Path

from . import config


def build_command(executable: str, target: str | None) -> str:
    return f'"{executable}" run' if target is None else f'"{executable}" "{target}"'


def _source_launcher() -> Path:
    launcher = config.source_launcher_path()
    launcher.parent.mkdir(parents=True, exist_ok=True)
    source_root = Path(__file__).resolve().parents[1]
    launcher.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(source_root)!r})\n"
        "from telemetry.__main__ import main\n"
        "raise SystemExit(main(['run']))\n",
        encoding="utf-8",
    )
    return launcher


def default_command() -> str:
    if getattr(sys, "frozen", False):
        return build_command(sys.executable, None)
    return build_command(sys.executable, str(_source_launcher()))


def _run(args: list[str]) -> bool:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=15).returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def register(command: str | None = None) -> bool:
    return _run(["schtasks", "/Create", "/TN", config.TASK_NAME, "/TR", command or default_command(), "/SC", "ONLOGON", "/F"])


def unregister() -> bool:
    return _run(["schtasks", "/Delete", "/TN", config.TASK_NAME, "/F"])


def is_registered() -> bool:
    return _run(["schtasks", "/Query", "/TN", config.TASK_NAME])
