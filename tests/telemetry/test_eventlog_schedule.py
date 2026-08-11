import sys

import pytest

from telemetry import schedule, store
from telemetry.eventlog import _allowed, parse_event_xml, watermark_key


def test_event_xml_parser_and_channel_watermarks():
    xml = """<Event xmlns='http://schemas.microsoft.com/win/2004/08/events/event'><System><Provider Name='Microsoft-Windows-Kernel-Power'/><EventID>41</EventID><Level>1</Level><TimeCreated SystemTime='2026-07-27T14:33:05.1234567Z'/><EventRecordID>9</EventRecordID></System><EventData><Data>C:\\Users\\a</Data></EventData></Event>"""
    parsed = parse_event_xml(xml)
    assert parsed and parsed["event_id"] == 41 and parsed["record_id"] == 9
    assert _allowed(parsed)
    assert not _allowed({"provider": "unrelated", "event_id": 1})
    assert watermark_key("System") != watermark_key("Application")


def test_schedule_source_command_uses_a_launcher(tmp_path, monkeypatch):
    monkeypatch.setattr(schedule.config, "app_dir", lambda: tmp_path)
    command = schedule.default_command()
    assert "telemetry_launcher.py" in command
    assert (tmp_path / "telemetry_launcher.py").exists()


def test_schedule_frozen_collector_restarts_its_own_executable(monkeypatch):
    monkeypatch.setattr(schedule.sys, "frozen", True, raising=False)
    monkeypatch.setattr(schedule.sys, "executable", r"C:\LocalRCA\RCA-Collector\RCA-Collector.exe")

    assert schedule.default_command() == r'"C:\LocalRCA\RCA-Collector\RCA-Collector.exe" run'


def test_register_writes_and_unregister_removes_startup_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(schedule.config, "app_dir", lambda: tmp_path)
    monkeypatch.setattr(schedule, "startup_dir", lambda: tmp_path / "Startup")

    assert schedule.is_registered() is False
    assert schedule.register() is True
    assert schedule.is_registered() is True

    body = (tmp_path / "Startup" / "rca-collector.cmd").read_text()
    assert "telemetry_launcher.py" in body

    assert schedule.unregister() is True
    assert schedule.is_registered() is False


def test_startup_wrapper_quotes_paths_containing_spaces(tmp_path, monkeypatch):
    """A user profile with a space broke the previous schtasks registration."""
    spaced = tmp_path / "yashash mathur" / "RCA"
    monkeypatch.setattr(schedule.config, "app_dir", lambda: spaced)
    monkeypatch.setattr(schedule, "startup_dir", lambda: tmp_path / "Startup")

    schedule.register()
    body = (tmp_path / "Startup" / "rca-collector.cmd").read_text()
    assert '"' in body
    assert f'"{spaced / "telemetry_launcher.py"}"' in body


@pytest.mark.skipif(sys.platform != "win32", reason="Windows Event Log only")
def test_event_log_reader_talks_to_the_real_api(tmp_path):
    """Exercise the live pywin32 calls, not just the XML parser.

    A call to a pywin32 function that does not exist in the installed build
    raises AttributeError, which the reader's broad handler swallowed -- so
    ingestion returned 0 forever while every parser test still passed.
    """
    pytest.importorskip("win32evtlog")
    from telemetry import store
    from telemetry.eventlog import EventLogReader, watermark_key

    reader = EventLogReader("System")
    newest = reader.newest_record_id()
    assert isinstance(newest, int) and newest > 0, "live EvtQuery/EvtNext/EvtRender must work"

    conn = store.connect(tmp_path / "t.db")
    store.init_schema(conn)
    reader.read_new(conn, limit=50)

    # Either rows landed or the watermark advanced past uninteresting records;
    # both prove the query path ran rather than failing silently.
    rows = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    mark = store.get_meta(conn, watermark_key("System"), "0")
    assert rows > 0 or int(mark) > 0


def test_frozen_build_never_launches_itself(monkeypatch, tmp_path):
    """A frozen desktop app must not be used as the collector command.

    default_command() fell back to sys.executable when frozen. In the packaged
    GUI that is RCA-Desktop.exe, so "start the collector" opened a second
    window, which opened a third on its own startup -- a fork bomb.
    """
    fake_gui = tmp_path / "RCA-Desktop" / "RCA-Desktop.exe"
    fake_gui.parent.mkdir(parents=True)
    fake_gui.write_text("x")

    monkeypatch.setattr(schedule.sys, "frozen", True, raising=False)
    monkeypatch.setattr(schedule.sys, "executable", str(fake_gui))

    # No collector beside it: refuse rather than launch the GUI again.
    assert schedule.collector_executable() is None
    assert schedule.default_command() is None
    assert schedule.start_now() is False
    assert schedule.register() is False


def test_frozen_build_finds_the_sibling_collector(monkeypatch, tmp_path):
    fake_gui = tmp_path / "RCA-Desktop" / "RCA-Desktop.exe"
    fake_gui.parent.mkdir(parents=True)
    fake_gui.write_text("x")
    collector = tmp_path / "RCA-Collector" / "RCA-Collector.exe"
    collector.parent.mkdir(parents=True)
    collector.write_text("x")

    monkeypatch.setattr(schedule.sys, "frozen", True, raising=False)
    monkeypatch.setattr(schedule.sys, "executable", str(fake_gui))

    assert schedule.collector_executable() == collector
    assert "RCA-Collector.exe" in schedule.default_command()
    assert "RCA-Desktop.exe" not in schedule.default_command()


def test_startup_wrapper_survives_a_non_ascii_profile_path(tmp_path, monkeypatch):
    """A non-ASCII username raised UnicodeEncodeError straight through install.

    `except OSError` does not catch it, so the command died with a traceback
    instead of reporting that the startup entry could not be written.
    """
    monkeypatch.setattr(schedule.config, "app_dir", lambda: tmp_path / "Ярослав" / "RCA")
    monkeypatch.setattr(schedule, "startup_dir", lambda: tmp_path / "Startup")

    registered = schedule.register()                # must not raise
    assert registered is schedule.is_registered()   # and must report honestly


def test_start_now_launches_without_a_shell(tmp_path, monkeypatch):
    """cmd.exe expands %NAME% even inside double quotes.

    The Startup wrapper must be a string because a .cmd is read by cmd.exe,
    but launching from inside the app has no such need: an argument vector
    means nothing in the profile path is ever interpreted.
    """
    monkeypatch.setattr(schedule.config, "app_dir", lambda: tmp_path / "RCA")
    captured = {}

    def _fake_popen(argv, **kwargs):
        captured["argv"] = argv
        captured["shell"] = kwargs.get("shell")
        return object()

    monkeypatch.setattr(schedule.subprocess, "Popen", _fake_popen)

    assert schedule.start_now() is True
    assert isinstance(captured["argv"], list), captured["argv"]
    assert captured["shell"] is False


def test_start_menu_shortcut_is_skipped_without_a_packaged_app(monkeypatch):
    """A source checkout must not advertise itself as installed software."""
    monkeypatch.setattr(schedule, "collector_executable", lambda: None)

    assert schedule.desktop_executable() is None
    assert schedule.create_start_menu_shortcut() is False
    # Removing one that was never made is the desired end state, not an error.
    assert schedule.remove_start_menu_shortcut() is True


def test_start_menu_shortcut_points_at_the_desktop_app(tmp_path, monkeypatch):
    """Windows search indexes shortcuts, so without one the app is unfindable."""
    collector = tmp_path / "RCA-Collector" / "RCA-Collector.exe"
    collector.parent.mkdir(parents=True)
    collector.write_text("x")
    gui = tmp_path / "RCA-Desktop" / "RCA-Desktop.exe"
    gui.parent.mkdir(parents=True)
    gui.write_text("x")

    monkeypatch.setattr(schedule, "collector_executable", lambda: collector)

    # It must target the GUI, never the collector: searching for the app and
    # launching a console collector would be the wrong result.
    assert schedule.desktop_executable() == gui
