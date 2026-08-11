"""Autostart registration for the collector.

Uses the per-user Startup folder rather than Task Scheduler. `schtasks /SC
ONLOGON` was tried first and fails with "Access is denied" without elevation:
creating a logon-triggered task is an administrative operation. Prompting for
UAC to run a diagnostic tool that only reads the local machine is a poor trade,
and the Startup folder achieves the same thing for the current user with no
privileges at all.

Everything is invoked through a generated .cmd wrapper. Passing the command
directly caused a real failure: the interpreter path and the launcher path both
need quoting, and a user profile containing a space (`C:\\Users\\yashash mathur`)
made the nested quotes split into separate arguments.
"""

import os
import subprocess
import sys
from pathlib import Path

from . import config
from .logsetup import get_logger

_LOGGER = get_logger(__name__)

_CMD_NAME = "rca-collector.cmd"


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


#: PyInstaller ships the collector as its own executable beside the GUI.
COLLECTOR_EXE = "RCA-Collector.exe"


def collector_executable() -> Path | None:
    """Locate the packaged collector, or None when running from source.

    Never fall back to sys.executable. In a frozen build that is the desktop
    application, so "start the collector" launched a second GUI, which started
    a third on its own launch, and so on -- a fork bomb that opened windows
    until the machine was cleared.
    """
    if not getattr(sys, "frozen", False):
        return None
    running = Path(sys.executable)
    # The collector restarting itself is the one case where sys.executable is
    # the right answer, and it needs no existence check.
    if running.name.lower() == COLLECTOR_EXE.lower():
        return running
    here = running.parent
    for candidate in (
        here / COLLECTOR_EXE,                           # same folder
        here.parent / "RCA-Collector" / COLLECTOR_EXE,  # sibling dist folder
    ):
        if candidate.exists():
            return candidate
    return None


def default_command() -> str | None:
    """Command that starts the collector, or None if it cannot be located."""
    if getattr(sys, "frozen", False):
        collector = collector_executable()
        return build_command(str(collector), None) if collector else None
    return build_command(sys.executable, str(_source_launcher()))


def default_argv() -> list[str] | None:
    """The same launch, as an argument vector rather than a shell string.

    The Startup wrapper has to be a string because a .cmd file is read by
    cmd.exe, but launching from inside the app does not: passing a list skips
    the shell entirely, so nothing in the profile path is ever interpreted.
    cmd.exe expands %NAME% even inside double quotes, which quoting cannot
    defend against.
    """
    if getattr(sys, "frozen", False):
        collector = collector_executable()
        return [str(collector), "run"] if collector else None
    return [sys.executable, str(_source_launcher())]


#: Where Windows lists installed programs for the current user. Per-user, so
#: no elevation is needed and nothing is written for other accounts.
_UNINSTALL_KEY = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\LocalRCA"


def register_uninstall_entry() -> bool:
    """List the app in Add/Remove Programs.

    A ZIP that registers itself to run at every logon, with no entry in the
    place Windows users look to remove things, is indistinguishable from
    something unwanted. The entry points at the collector's own uninstall.
    """
    collector = collector_executable()
    if collector is None:
        return False                    # running from source: nothing installed
    try:
        import winreg

        from . import config
        from version import __version__

        with winreg.CreateKey(winreg.HKEY_CURRENT_USER, _UNINSTALL_KEY) as key:
            for name, value in (
                ("DisplayName", "LocalRCA — Local Root Cause Analysis"),
                ("DisplayVersion", __version__),
                ("Publisher", "LocalRCA"),
                ("InstallLocation", str(collector.parent.parent)),
                ("UninstallString", f'"{collector}" uninstall'),
                ("URLInfoAbout", "https://github.com/mathuryashash/RCA-Major_project"),
                ("Comments", f"Collected telemetry is stored at {config.app_dir()}"),
            ):
                winreg.SetValueEx(key, name, 0, winreg.REG_SZ, value)
            for name in ("NoModify", "NoRepair"):
                winreg.SetValueEx(key, name, 0, winreg.REG_DWORD, 1)
        return True
    except (ImportError, OSError):
        return False


def _start_menu_shortcut() -> Path:
    appdata = os.environ.get("APPDATA")
    base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
    return base / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "LocalRCA.lnk"


def desktop_executable() -> Path | None:
    """The GUI beside the collector, or None when running from source."""
    collector = collector_executable()
    if collector is None:
        return None
    candidate = collector.parent.parent / "RCA-Desktop" / "RCA-Desktop.exe"
    return candidate if candidate.exists() else None


def create_start_menu_shortcut() -> bool:
    """Put the app where Windows search will find it.

    An Add/Remove Programs entry lists the app under Settings, but Start menu
    search indexes shortcuts -- so without one the application is installed,
    running at logon, and unfindable by name.
    """
    target = desktop_executable()
    if target is None:
        return False                    # source checkout: nothing to point at
    try:
        import win32com.client

        shell = win32com.client.Dispatch("WScript.Shell")
        link = shell.CreateShortCut(str(_start_menu_shortcut()))
        link.TargetPath = str(target)
        link.WorkingDirectory = str(target.parent)
        link.IconLocation = str(target)
        link.Description = "Local Root Cause Analysis"
        link.save()
        return _start_menu_shortcut().exists()
    except Exception:                   # noqa: BLE001 - a missing shortcut is cosmetic
        return False


def remove_start_menu_shortcut() -> bool:
    try:
        _start_menu_shortcut().unlink(missing_ok=True)
        return True
    except OSError:
        return False


def remove_uninstall_entry() -> bool:
    """Take the Add/Remove Programs entry away again."""
    try:
        import winreg

        winreg.DeleteKey(winreg.HKEY_CURRENT_USER, _UNINSTALL_KEY)
        return True
    except FileNotFoundError:
        return True                     # already absent is the desired state
    except (ImportError, OSError):
        return False


def startup_dir() -> Path:
    appdata = os.environ.get("APPDATA")
    base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
    return base / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"


def _shortcut_path() -> Path:
    return startup_dir() / _CMD_NAME


def register(command: str | None = None) -> bool:
    """Write a startup wrapper that launches the collector at logon."""
    command = command or default_command()
    if not command:
        _LOGGER.warning("Collector executable not found; nothing registered.")
        return False
    path = _shortcut_path()
    try:
        # `start ""` detaches so the console window closes immediately, and
        # cmd.exe reads the batch file in the console (OEM) code page.
        # Encode before opening: write_text() truncates first and encodes
        # second, so a profile path outside the code page left an empty .cmd
        # that Windows would still run at logon.
        # ponytail: such a path cannot be registered at all; `chcp 65001` in
        # the wrapper would lift that ceiling.
        body = ("@echo off\r\n" f'start "" /min {command}\r\n').encode("oem")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    except (OSError, UnicodeError, LookupError):
        # LookupError: the "oem" codec exists only on Windows, and this
        # registers a Windows startup entry, so failing is the right answer.
        return False
    return path.exists()


def unregister() -> bool:
    try:
        _shortcut_path().unlink(missing_ok=True)
    except OSError:
        return False
    return not _shortcut_path().exists()


def is_registered() -> bool:
    return _shortcut_path().exists()


def start_now(command: str | None = None) -> bool:
    """Launch the collector immediately, detached from this console."""
    # A caller-supplied string still goes through the shell, since that is what
    # it asked for; the default path builds an argument vector instead so the
    # profile path is never handed to cmd.exe to interpret.
    argv = command or default_argv()
    if not argv:
        # Better to collect nothing than to launch the wrong executable.
        _LOGGER.warning("Collector executable not found; not starting one.")
        return False
    try:
        subprocess.Popen(
            argv,
            shell=isinstance(argv, str),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except OSError:
        return False
    return True
