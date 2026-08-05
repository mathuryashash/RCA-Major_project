"""Command-line behaviour that makes a promise to the user."""

import logging

from telemetry import __main__ as cli


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

    monkeypatch.setattr(cli.config, "app_dir", lambda: app_dir)
    monkeypatch.setattr(cli.schedule, "unregister", lambda: True)
    monkeypatch.setattr(cli, "request_stop", lambda: None)

    assert cli._delete_data() == 0
    assert not app_dir.exists()


def test_delete_all_data_leaves_other_loggers_alone(tmp_path, monkeypatch):
    """Closing the collector's log handler must not deafen the whole process."""
    monkeypatch.setattr(cli.config, "app_dir", lambda: tmp_path / "RCA")
    monkeypatch.setattr(cli.schedule, "unregister", lambda: True)
    monkeypatch.setattr(cli, "request_stop", lambda: None)

    unrelated = logging.getLogger("some.other.library")
    handler = logging.NullHandler()
    unrelated.addHandler(handler)

    cli._delete_data()

    assert handler in unrelated.handlers
    unrelated.removeHandler(handler)
