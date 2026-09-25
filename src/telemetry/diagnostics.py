"""Build a local diagnostics bundle for bug reports.

The application's promise is that nothing leaves the machine on its own, so
support cannot work by the app phoning home. Instead the user asks for a zip,
looks at it if they like, and attaches it to an issue themselves.

What goes in is an *allowlist*, never "the app folder minus the private
bits": a denylist silently starts leaking the day someone adds a new file to
%LOCALAPPDATA%\\RCA. Telemetry rows, the database, the model and generated
reports are never read into the bundle. Every piece of text that is copied is
passed through `scrub()` so the Windows username, profile path and machine
name do not travel with it.
"""

from __future__ import annotations

import getpass
import json
import os
import platform
import re
import sqlite3
import sys
import tempfile
import time
import zipfile
from pathlib import Path

from . import config

#: Text files copied (scrubbed) from the app folder. Rotated backups are
#: picked up by suffix; see `_log_files`.
_LOG_NAMES = ("collector.log", "desktop.log")

#: Small, non-personal support files. timing.json holds measured training
#: rates; the supervisor script shows how collection is kept alive, which is
#: where most "it stopped collecting" reports turn out to live. The logon
#: launcher itself lives in the Startup folder, not here; see `_log_files`.
_EXTRA_NAMES = ("timing.json", "supervise.ps1")

#: Settings read from the meta table. Listed explicitly so a value added
#: there later is not exported without someone deciding it should be.
_META_KEYS = (
    "schema_version",
    "consent_granted",
    "consent_granted_at",
    "capture_messages",
    "update_check_enabled",
)

#: Store-summary fields that describe collection health. Deliberately omits
#: "latest" (a full telemetry row) -- the bundle carries counts, never values.
_HEALTH_KEYS = (
    "exists", "size_bytes", "samples", "events", "proc_samples", "gaps",
    "sampling_gaps", "gap_hours", "expected_samples", "coverage_pct",
    "quarantined", "quarantined_bytes", "first_ts", "last_ts",
)

#: Hard ceiling per copied file. Logs rotate at 1 MB, but a file that has
#: grown past that is itself a symptom, and its tail is what matters.
_MAX_FILE_BYTES = 2_000_000

README = """\
LocalRCA diagnostics bundle
===========================

Created locally by "Export diagnostics" in LocalRCA. Nothing was sent
anywhere; you choose whether to share this file.

Included
  system.json  app version, Windows version, Python version, build type
  health.json  collection health: row COUNTS, coverage %, sampling gaps,
               first/last sample time, database size, collector registered
               / paused, whether a model file exists, a few settings flags
  logs/        collector and desktop logs (including rotated copies), plus
               timing.json and the logon launcher scripts if present

NOT included
  the telemetry database (no samples, events, process lists or app names
  from it), the trained model, generated RCA reports.

Redaction
  Your Windows username, profile path and computer name are replaced with
  <user> / <computer> in every file. Log lines may still mention program
  names that appeared in error messages -- look through logs/ before
  sharing if that matters to you.
"""


# --------------------------------------------------------------------------
# Redaction


def _identities() -> tuple[list[str], list[str]]:
    """(user names, computer names) that must not appear in the bundle.

    The profile folder name is not always the login name (renamed accounts,
    Microsoft-account profiles truncated to five letters), so both are taken.
    """
    users = {os.environ.get("USERNAME"), Path(os.path.expanduser("~")).name}
    try:
        users.add(getpass.getuser())
    except Exception:  # noqa: BLE001 - getuser raises assorted errors
        pass
    computers = {os.environ.get("COMPUTERNAME"), platform.node()}
    return _usable(users), _usable(computers)


def _usable(names) -> list[str]:
    # Very short names would shred ordinary words ("a", "it"), so they are
    # left to the path-based rules, which still catch them inside paths.
    # Longest first so "jane.doe" is replaced before "jane" can split it.
    return sorted({n for n in names if n and len(n) >= 3}, key=len, reverse=True)


def _token(name: str) -> re.Pattern:
    # Letters/digits either side mean the name is part of a longer word, which
    # is left alone; path separators, spaces and quotes are real boundaries.
    return re.compile(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])", re.IGNORECASE)


#: Any `X:\Users\<name>` left over -- another account's profile, or ours
#: spelt with forward or doubled (JSON-escaped) separators. A folder name may
#: contain spaces ("Jane Doe"), so the segment runs to the next separator when
#: there is one; otherwise it stops at whitespace, so "C:\Users\bob failed"
#: does not swallow the rest of the log line.
_USERS_DIR = re.compile(
    r"([A-Za-z]:(?:\\\\|\\|/)Users(?:\\\\|\\|/))(?!<user>)"
    r"(?:[^\\/\r\n\"':;,<>|]+(?=[\\/])|[^\\/\s\"':;,<>|]+)",
    re.IGNORECASE)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")


def scrub(text: str) -> str:
    """Remove the user's identity from free text copied into the bundle."""
    if not text:
        return text or ""
    home = os.path.expanduser("~")
    users, computers = _identities()
    output = text
    # The whole profile path first: when the folder name contains a space
    # ("C:\Users\Jane Doe") the generic path rule below would stop at the
    # space and leave the surname behind.
    # Three spellings: native, forward-slash (Qt and pathlib.as_posix) and
    # doubled backslashes (anything that went through json.dumps).
    folder_name = Path(home).name
    if folder_name and len(home) > 3:
        for variant in {home, home.replace("\\", "/"), home.replace("\\", "\\\\")}:
            # Keep the matched text's own drive/"Users" casing; only the
            # folder name is personal.
            output = re.sub(
                re.escape(variant),
                lambda match: match.group(0)[: -len(folder_name)] + "<user>",
                output, flags=re.IGNORECASE)
    for name in users:
        output = _token(name).sub("<user>", output)
    for name in computers:
        output = _token(name).sub("<computer>", output)
    output = _USERS_DIR.sub(lambda match: match.group(1) + "<user>", output)
    return _EMAIL.sub("<email>", output)


# --------------------------------------------------------------------------
# Collection


def _read_tail(path: Path) -> str:
    size = path.stat().st_size
    with open(path, "rb") as handle:
        if size > _MAX_FILE_BYTES:
            handle.seek(size - _MAX_FILE_BYTES)
        data = handle.read()
    return data.decode("utf-8", errors="replace")


def _log_files(folder: Path) -> list[Path]:
    """Existing logs and their rotated backups, oldest-first per log."""
    found = []
    for name in _LOG_NAMES:
        for suffix in range(config.LOG_BACKUPS + 2, 0, -1):   # a little slack
            candidate = folder / f"{name}.{suffix}"
            if candidate.is_file():
                found.append(candidate)
        if (folder / name).is_file():
            found.append(folder / name)
    found.extend(folder / name for name in _EXTRA_NAMES if (folder / name).is_file())
    try:
        from . import schedule
        launcher = schedule.startup_dir() / schedule._CMD_NAME
        if launcher.is_file():
            found.append(launcher)
    except Exception:  # noqa: BLE001 - a missing launcher is itself the finding
        pass
    return found


def _read_meta(db: Path) -> dict:
    """Allowlisted settings, opened read-only.

    Read-only matters twice: the collector may be writing, and a plain
    connect() on a missing path would create an empty database as a side
    effect of the user asking for a bug report.
    """
    if not db.exists():
        return {}
    connection = sqlite3.connect(db.as_uri() + "?mode=ro", uri=True, timeout=5)
    try:
        placeholders = ", ".join("?" for _ in _META_KEYS)
        rows = connection.execute(
            f"SELECT key, value FROM meta WHERE key IN ({placeholders})", _META_KEYS
        ).fetchall()
    finally:
        connection.close()
    return dict(rows)


def _jsonable(value):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    return value


def system_info() -> dict:
    from version import __version__

    try:
        edition = platform.win32_edition() if hasattr(platform, "win32_edition") else None
    except Exception:  # noqa: BLE001 - registry lookups fail in odd sandboxes
        edition = None
    return {
        "app_version": __version__,
        "frozen_build": bool(getattr(sys, "frozen", False)),
        "platform": platform.platform(),
        "windows_release": platform.release(),
        "windows_version": platform.version(),
        "windows_edition": edition,
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def health_info() -> dict:
    """Collection health. Every probe is independent: a diagnostics export
    is most wanted exactly when something is broken, so one failure is
    recorded in the output rather than aborting the whole bundle."""
    from . import analysis, schedule, secure_storage

    folder = config.app_dir()
    health: dict = {"errors": {}}

    try:
        summary = analysis.store_summary(config.db_path())
        store = {key: _jsonable(summary.get(key)) for key in _HEALTH_KEYS}
        # Channel NAMES only: which sensors work on this hardware is a
        # common support question and says nothing about the user.
        store["available_channels"] = _jsonable(summary.get("available", set()))
        health["store"] = store
    except Exception as error:  # noqa: BLE001
        health["errors"]["store_summary"] = repr(error)

    try:
        health["settings"] = _read_meta(config.db_path())
    except Exception as error:  # noqa: BLE001
        health["errors"]["settings"] = repr(error)

    for key, probe in (("collector_registered", schedule.is_registered),
                       ("collection_paused", schedule.collection_paused),
                       ("dpapi_available", secure_storage.dpapi_available)):
        try:
            health[key] = bool(probe())
        except Exception as error:  # noqa: BLE001
            health["errors"][key] = repr(error)

    def model_info() -> dict:
        model = folder / "telemetry_model.pt"
        if not model.is_file():
            return {"exists": False, "size_bytes": 0, "modified": None}
        stat = model.stat()
        return {"exists": True, "size_bytes": stat.st_size,
                "modified": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(stat.st_mtime))}

    def report_count() -> int:
        reports = folder / "reports"
        return sum(1 for p in reports.iterdir() if p.is_file()) if reports.is_dir() else 0

    def app_dir_files() -> dict:
        # Names and sizes of top-level files only: spots stray WAL files,
        # quarantined databases and a missing launcher at a glance.
        if not folder.is_dir():
            return {}
        return {p.name: p.stat().st_size for p in sorted(folder.iterdir()) if p.is_file()}

    for key, probe in (("model", model_info), ("report_count", report_count),
                       ("app_dir_files", app_dir_files)):
        try:
            health[key] = probe()
        except OSError as error:
            health["errors"][key] = repr(error)
    health["app_dir"] = str(folder)
    return health


def default_filename() -> str:
    return f"LocalRCA-diagnostics-{time.strftime('%Y%m%d-%H%M%S')}.zip"


def export(destination: Path | str) -> Path:
    """Write the diagnostics zip to `destination` and return its path.

    Written to a temporary file beside the target and renamed into place, so
    a failure part-way never leaves a truncated zip that looks complete.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    folder = config.app_dir()

    handle, temporary = tempfile.mkstemp(
        prefix=".diagnostics-", suffix=".zip.tmp", dir=destination.parent)
    os.close(handle)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            bundle.writestr("README.txt", README)
            # ensure_ascii=False: the default escapes a non-ASCII profile
            # folder ("José" -> "Jos\\u00e9") before scrub() sees it, and the
            # escaped spelling matches none of the redaction rules.
            bundle.writestr("system.json",
                            scrub(json.dumps(system_info(), indent=2, ensure_ascii=False)))
            bundle.writestr("health.json",
                            scrub(json.dumps(health_info(), indent=2, default=str,
                                             ensure_ascii=False)))
            for path in _log_files(folder):
                try:
                    bundle.writestr(f"logs/{path.name}", scrub(_read_tail(path)))
                except OSError as error:
                    bundle.writestr(f"logs/{path.name}.unreadable.txt", scrub(repr(error)))
        os.replace(temporary, destination)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return destination
