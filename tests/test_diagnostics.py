"""The diagnostics bundle: what it contains, what it must never contain."""

import json
import re
import sqlite3
import time
import zipfile

import pytest

from telemetry import config, diagnostics, schedule, store

#: Identity the tests pretend to be. A space in the profile folder is the
#: case a naive "C:\Users\<non-space>+" rule gets wrong.
USER = "janedoe"
HOME = r"C:\Users\Jane Doe"
COMPUTER = "JANES-LAPTOP"
SECRET_APP = "SecretDiary.exe"


@pytest.fixture
def app_dir(tmp_path, monkeypatch):
    """A fake %LOCALAPPDATA%\\RCA with every kind of file the real one holds."""
    folder = tmp_path / "RCA"
    (folder / "reports").mkdir(parents=True)
    # Never touch the real user data, whichever helper a code path uses.
    monkeypatch.setattr(config, "app_dir", lambda: folder)
    monkeypatch.setattr(config, "db_path", lambda: folder / "telemetry.db")
    monkeypatch.setattr(schedule, "is_registered", lambda: True)
    # The logon launcher lives in the Startup folder; keep the real one out.
    startup = tmp_path / "Startup"
    startup.mkdir()
    monkeypatch.setattr(schedule, "startup_dir", lambda: startup)
    (startup / schedule._CMD_NAME).write_text(
        "@echo off" + chr(10) + f'powershell -File "{HOME}' + chr(92) + 'supervise.ps1"',
        encoding="utf-8")
    monkeypatch.setenv("USERPROFILE", HOME)
    monkeypatch.setenv("USERNAME", USER)
    monkeypatch.setenv("COMPUTERNAME", COMPUTER)
    for name in ("LOGNAME", "USER", "LNAME", "HOME"):
        monkeypatch.delenv(name, raising=False)

    conn = store.connect(folder / "telemetry.db")
    store.init_schema(conn)
    now = int(time.time())
    for offset in range(0, 300, 30):
        store.insert_sample(conn, now - 300 + offset, {
            "cpu_pct": 12.5, "mem_pct": 40.0, "foreground_app": SECRET_APP})
    conn.execute(
        "INSERT INTO events (ts, channel, record_id, provider, event_id, level, message_redacted)"
        " VALUES (?, 'Application', 1, 'Application Error', 1000, 'Error', ?)",
        (now, f"{SECRET_APP} crashed"))
    store.set_meta(conn, "consent_granted", "1")
    store.set_meta(conn, "update_check_enabled", "0")
    store.set_meta(conn, "watermark_System", "123456")        # not allowlisted
    conn.commit()
    conn.close()

    (folder / "collector.log").write_text(
        f"2026-09-25 INFO telemetry: started from {HOME}\\AppData\\Local\\RCA\n"
        f"2026-09-25 ERROR telemetry: {USER} on {COMPUTER} failed; "
        "see C:/Users/Jane Doe/AppData/x.txt and C:\\Users\\otheruser\\y\n",
        encoding="utf-8")
    (folder / "collector.log.1").write_text("older collector line\n", encoding="utf-8")
    (folder / "desktop.log").write_text(
        '{"path": "C:\\\\Users\\\\Jane Doe\\\\Desktop"} mail jane@example.com\n',
        encoding="utf-8")
    (folder / "desktop.log.2").write_text("oldest desktop line\n", encoding="utf-8")
    (folder / "timing.json").write_text('{"epoch": 0.4}', encoding="utf-8")
    (folder / "supervise.ps1").write_text(f"& '{HOME}\\collector.exe'", encoding="utf-8")
    (folder / "telemetry_model.pt").write_bytes(b"MODEL-BYTES-" + SECRET_APP.encode())
    (folder / "reports" / "rca_report.md").write_text(f"{SECRET_APP} caused it")
    (folder / "telemetry.db.corrupt-1").write_bytes(b"x" * 10)
    return folder


def _read(bundle):
    with zipfile.ZipFile(bundle) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def test_bundle_has_expected_members(app_dir, tmp_path):
    bundle = diagnostics.export(tmp_path / "out" / "diag.zip")
    assert bundle.exists()
    names = set(_read(bundle))
    assert names == {
        "README.txt", "system.json", "health.json",
        "logs/collector.log", "logs/collector.log.1",
        "logs/desktop.log", "logs/desktop.log.2",
        "logs/timing.json", "logs/supervise.ps1", "logs/rca-collector.cmd",
    }
    # No temporary file left beside the result.
    assert [p.name for p in bundle.parent.iterdir()] == ["diag.zip"]


def test_bundle_never_contains_telemetry_database_model_or_reports(app_dir, tmp_path):
    contents = _read(diagnostics.export(tmp_path / "diag.zip"))
    for name, data in contents.items():
        assert not name.endswith((".db", ".pt", ".md")), name
        assert b"SQLite format" not in data, name
        assert b"MODEL-BYTES" not in data, name
        # The foreground app and event text live only in telemetry rows, so
        # finding them anywhere means a row leaked.
        assert SECRET_APP.encode() not in data, name
    health = json.loads(contents["health.json"])
    assert "latest" not in health["store"]                  # no raw sample row
    assert "watermark_System" not in health["settings"]     # meta is allowlisted


def test_health_summary_reports_collection_state(app_dir, tmp_path):
    (app_dir / "stop.flag").write_text("")
    health = json.loads(_read(diagnostics.export(tmp_path / "diag.zip"))["health.json"])
    assert health["store"]["samples"] == 10
    assert health["store"]["events"] == 1
    assert health["store"]["coverage_pct"] == pytest.approx(100.0)
    assert health["store"]["sampling_gaps"] == 0
    assert health["store"]["quarantined"] == 1
    assert "cpu_pct" in health["store"]["available_channels"]
    assert health["collector_registered"] is True
    assert health["collection_paused"] is True
    assert health["settings"] == {"consent_granted": "1", "update_check_enabled": "0",
                                  "schema_version": str(config.SCHEMA_VERSION)}
    assert health["model"]["exists"] is True
    assert health["report_count"] == 1
    assert "telemetry.db" in health["app_dir_files"]
    assert health["errors"] == {}


def test_system_info_carries_version_and_os(app_dir, tmp_path):
    from version import __version__

    system = json.loads(_read(diagnostics.export(tmp_path / "diag.zip"))["system.json"])
    assert system["app_version"] == __version__
    assert system["platform"]
    assert system["python"]


def test_identity_is_redacted_everywhere(app_dir, tmp_path):
    contents = _read(diagnostics.export(tmp_path / "diag.zip"))
    leaks = [USER, "jane doe", COMPUTER.lower(), "otheruser", "jane@example.com"]
    # health.json records the (test) app folder, which sits under whoever is
    # really running the suite. That profile is "another account" as far as
    # the faked identity knows, so the generic X:\Users\<name> rule must
    # catch it. (Elsewhere in the temp path, e.g. "pytest-of-<name>", it is
    # only caught for the real user's own identity, which is faked here.)
    parts = [part.lower() for part in tmp_path.parts]
    real = parts[parts.index("users") + 1] if "users" in parts else None
    for name, data in contents.items():
        text = data.decode("utf-8").lower()
        for leak in leaks:
            assert leak not in text, (name, leak)
        if real:
            assert not re.search(r"users[\\/]+" + re.escape(real), text), name
    log = contents["logs/collector.log"].decode()
    assert r"C:\Users\<user>\AppData\Local\RCA" in log
    assert "<user> on <computer> failed" in log
    assert "C:/Users/<user>/AppData" in log
    assert "<user>" in contents["logs/desktop.log"].decode()


@pytest.mark.parametrize("text, expected", [
    (r"C:\Users\Jane Doe\file", r"C:\Users\<user>\file"),
    ("c:/users/jane doe/file", "c:/users/<user>/file"),
    (r"C:\\Users\\Jane Doe\\file", r"C:\\Users\\<user>\\file"),
    (r"D:\Users\someone.else\x", r"D:\Users\<user>\x"),
    (r"C:\Users\Other Person\x", r"C:\Users\<user>\x"),
    (r"C:\Users\bob failed to start", r"C:\Users\<user> failed to start"),
    ("janedoe logged in", "<user> logged in"),
    # Part of a longer word is not the username; rewriting it would make
    # logs harder to read without protecting anything.
    ("xjanedoex", "xjanedoex"),
    ("", ""),
])
def test_scrub(app_dir, text, expected):
    assert diagnostics.scrub(text) == expected


def test_non_ascii_profile_folder_is_redacted_in_json(app_dir, tmp_path, monkeypatch):
    # json.dumps escapes non-ASCII by default ("José" -> "Jos\u00e9") before
    # scrub() runs, and the escaped spelling matches no redaction rule. Names
    # like this are common, so the app folder recorded in health.json would
    # carry the real name out in the bundle.
    home = "C:\\Users\\José Díaz"
    monkeypatch.setenv("USERPROFILE", home)
    monkeypatch.setenv("USERNAME", "josediaz")
    monkeypatch.setattr(config, "app_dir", lambda: type(app_dir)(home + "\\AppData\\Local\\RCA"))
    health = _read(diagnostics.export(tmp_path / "diag.zip"))["health.json"].decode("utf-8")
    assert "Jos" not in health and "D\\u00edaz" not in health, health
    assert "<user>" in health


def test_filesystem_probe_failure_does_not_abort_export(app_dir, tmp_path, monkeypatch):
    # A report vanishing or a permission error mid-listing is exactly when a
    # bug report is wanted; it must be recorded, not end the export.
    real_iterdir = type(app_dir).iterdir

    def flaky_iterdir(self):
        if self.name == "reports":
            raise PermissionError("denied")
        return real_iterdir(self)

    monkeypatch.setattr(type(app_dir), "iterdir", flaky_iterdir)
    contents = _read(diagnostics.export(tmp_path / "diag.zip"))
    health = json.loads(contents["health.json"])
    assert "denied" in health["errors"]["report_count"]
    assert "logs/collector.log" in contents


def test_export_survives_missing_database(app_dir, tmp_path):
    for leftover in app_dir.glob("telemetry.db*"):
        leftover.unlink()
    health = json.loads(_read(diagnostics.export(tmp_path / "diag.zip"))["health.json"])
    assert health["store"]["exists"] is False
    assert health["settings"] == {}
    # Asking for a bug report must not create an empty database.
    assert not (app_dir / "telemetry.db").exists()


def test_export_records_probe_failures_instead_of_aborting(app_dir, tmp_path, monkeypatch):
    from telemetry import analysis

    def broken(*_args, **_kwargs):
        raise sqlite3.DatabaseError("database disk image is malformed")

    monkeypatch.setattr(analysis, "store_summary", broken)
    contents = _read(diagnostics.export(tmp_path / "diag.zip"))
    health = json.loads(contents["health.json"])
    assert "malformed" in health["errors"]["store_summary"]
    assert "logs/collector.log" in contents          # the rest still arrived


def test_failed_export_leaves_no_partial_file(app_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, "system_info", lambda: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        diagnostics.export(tmp_path / "diag.zip")
    assert list(tmp_path.glob("*.zip*")) == []
    assert list(tmp_path.glob(".diagnostics-*")) == []


def test_oversized_log_is_truncated_to_its_tail(app_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, "_MAX_FILE_BYTES", 100)
    (app_dir / "collector.log").write_text("a" * 500 + "THE-END", encoding="utf-8")
    log = _read(diagnostics.export(tmp_path / "diag.zip"))["logs/collector.log"]
    assert len(log) == 100
    assert log.endswith(b"THE-END")


def test_default_filename():
    name = diagnostics.default_filename()
    assert name.startswith("LocalRCA-diagnostics-") and name.endswith(".zip")


# --------------------------------------------------------------------------
# Desktop wiring


@pytest.fixture
def window(qtbot, monkeypatch, app_dir):
    from pipeline import engine
    from desktop.main_window import MainWindow

    monkeypatch.setattr(engine, "model_status", lambda path: engine.ModelStatus(
        exists=False, reason="No model has been trained yet."))
    main = MainWindow()
    qtbot.addWidget(main)
    return main


def test_header_has_export_diagnostics_button(window):
    assert window.diagnostics_button.text().startswith("Export diagnostics")
    assert "no collected telemetry" in window.diagnostics_button.toolTip()


def test_export_button_writes_bundle_and_reports_path(window, tmp_path, monkeypatch):
    import desktop.main_window as main_window

    target = tmp_path / "chosen"                       # no extension: one is added
    offered = {}

    def fake_dialog(_parent, _title, start, _filter):
        offered["start"] = start
        return str(target), "Zip archive (*.zip)"

    shown = {}
    monkeypatch.setattr(main_window.QFileDialog, "getSaveFileName", fake_dialog)
    monkeypatch.setattr(main_window.QMessageBox, "information",
                        lambda _p, _t, text: shown.setdefault("text", text))

    path = window.export_diagnostics()
    assert path == tmp_path / "chosen.zip" and path.exists()
    assert "LocalRCA-diagnostics-" in offered["start"]
    assert str(path) in shown["text"]
    assert "does not include any collected telemetry" in shown["text"]


def test_export_button_cancel_does_nothing(window, tmp_path, monkeypatch):
    import desktop.main_window as main_window

    monkeypatch.setattr(main_window.QFileDialog, "getSaveFileName",
                        lambda *_args: ("", ""))
    monkeypatch.setattr(main_window.QMessageBox, "information",
                        lambda *_args: pytest.fail("no dialog on cancel"))
    assert window.export_diagnostics() is None


def test_export_button_reports_failure(window, tmp_path, monkeypatch):
    import desktop.main_window as main_window

    monkeypatch.setattr(main_window.QFileDialog, "getSaveFileName",
                        lambda *_args: (str(tmp_path / "d.zip"), ""))
    monkeypatch.setattr(diagnostics, "export",
                        lambda _dest: (_ for _ in ()).throw(OSError("disk full")))
    warned = {}
    monkeypatch.setattr(main_window.QMessageBox, "warning",
                        lambda _p, _t, text: warned.setdefault("text", text))
    assert window.export_diagnostics() is None
    assert "disk full" in warned["text"]
