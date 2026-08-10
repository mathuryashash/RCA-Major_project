"""Command-line behaviour that makes a promise to the user."""

import io
import logging
import sys

import pytest

from telemetry import __main__ as cli


def _isolate(monkeypatch, app_dir):
    """Point every path at a temp directory before anything destructive runs."""
    assert "Temp" in str(app_dir) or "pytest" in str(app_dir)
    monkeypatch.setattr(cli.config, "app_dir", lambda: app_dir)
    monkeypatch.setattr(cli.config, "db_path", lambda: app_dir / "telemetry.db")
    monkeypatch.setattr(cli.schedule, "unregister", lambda: True)


def test_delete_all_data_erases_the_whole_application_directory(tmp_path, monkeypatch):
    """"Erase all local data" must not leave reports, models or logs behind.

    Deleting only telemetry.db left collector.log in place, and that log
    records exception tracebacks carrying the user's profile path.
    """
    app_dir = tmp_path / "RCA"
    (app_dir / "reports").mkdir(parents=True)
    (app_dir / "telemetry.db").write_text("db")
    (app_dir / "telemetry.db-wal").write_text("wal")
    (app_dir / "collector.log").write_text("traceback: C:\\Users\\someone\\...")
    (app_dir / "collector.log.1").write_text("older traceback")
    (app_dir / "telemetry_model.pt").write_bytes(b"model")
    (app_dir / "reports" / "rca_report.md").write_text("chrome.exe caused it")
    _isolate(monkeypatch, app_dir)

    assert cli._delete_data() == 0
    assert not app_dir.exists()


@pytest.mark.skipif(sys.platform != "win32", reason="relies on Windows delete-locking")
def test_delete_all_data_keeps_the_stop_flag_while_the_collector_holds_the_database(tmp_path, monkeypatch):
    """The stop signal lives in the directory being erased.

    Clearing the directory first deleted stop.flag within microseconds of
    writing it. The collector polls once per cadence, so it never saw the
    signal, never exited, and kept the database locked -- while the model and
    the reports were destroyed on every retry. delete-all-data was the only
    way to stop a collector, so this made one unstoppable.
    """
    app_dir = tmp_path / "RCA"
    app_dir.mkdir(parents=True)
    database = app_dir / "telemetry.db"
    database.write_text("db")
    (app_dir / "telemetry_model.pt").write_bytes(b"model")
    _isolate(monkeypatch, app_dir)
    monkeypatch.setattr(cli.config, "STOP_WAIT_S", 0.5)

    # Windows refuses to unlink a file another handle has open, which is
    # exactly the state a running collector leaves the database in.
    with open(database, "rb"):
        assert cli._delete_data() == 1
        assert (app_dir / "stop.flag").exists()         # collector can still see it
        assert (app_dir / "telemetry_model.pt").exists()  # nothing traded for nothing
        assert database.exists()


def test_delete_all_data_leaves_other_loggers_usable(tmp_path, monkeypatch):
    """Closing the collector's log handler must not deafen the whole process.

    logging.shutdown() would close every handler in the process, not just this
    one -- and it leaves them attached, so checking attachment proves nothing.
    """
    _isolate(monkeypatch, tmp_path / "RCA")
    stream = io.StringIO()
    unrelated = logging.getLogger("some.other.library")
    handler = logging.StreamHandler(stream)
    unrelated.addHandler(handler)

    try:
        cli._delete_data()
        unrelated.error("still here")
        assert "still here" in stream.getvalue()
    finally:
        unrelated.removeHandler(handler)


def test_delete_all_data_clears_rendered_figures(tmp_path, monkeypatch):
    """Figures are written outside the data directory and were left behind.

    The desktop app renders the graph and timeline into a mkdtemp directory
    cleaned only at interpreter exit, which never runs if the app is killed.
    Those files carry the metric values behind an incident, so "erase all
    local data" has to reach them too.
    """
    import tempfile

    fake_temp = tmp_path / "temp"
    fake_temp.mkdir()
    leftover = fake_temp / "rca_desktop_abc123"
    leftover.mkdir()
    (leftover / "figure_1.html").write_text("cpu_pct spiked at 02:47")
    unrelated = fake_temp / "something_else"
    unrelated.mkdir()

    _isolate(monkeypatch, tmp_path / "RCA")
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(fake_temp))

    assert cli._delete_data() == 0
    assert not leftover.exists()
    assert unrelated.exists(), "only this application's figures may be removed"
