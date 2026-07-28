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
