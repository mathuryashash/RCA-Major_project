"""Collection must not stop silently, and must not look healthy when it has.

Observed: a rebuild briefly deleted RCA-Collector.exe, the supervisor read the
missing file as "uninstalled" and exited for good, and collection stopped for
about an hour while the Captured Data tab kept saying "Collecting every 30
seconds" -- because nothing ever looked at the age of the newest sample.
"""

import os
import shutil
import subprocess
import sys
import time
import uuid

import pandas as pd
import pytest

from telemetry import config, schedule


# --- the supervisor ----------------------------------------------------------

def _supervisor_text(tmp_path, monkeypatch, collector=None):
    monkeypatch.setattr(config, "app_dir", lambda: tmp_path / "RCA")
    collector = collector or tmp_path / "RCA-Collector" / "RCA-Collector.exe"
    return schedule._write_supervisor(collector).read_text(encoding="utf-8")


def test_supervisor_does_not_treat_a_missing_exe_as_an_uninstall(tmp_path, monkeypatch):
    script = _supervisor_text(tmp_path, monkeypatch)
    # The exact line that ended supervision during a rebuild.
    assert "if (-not (Test-Path $collector)) { break }" not in script
    assert str(schedule.MISSING_COLLECTOR_GRACE_S) in script, "the wait must be bounded"
    assert "stop.flag" in script, "a pause must still end it"
    assert "$PSCommandPath" in script, "unregistering (which deletes the script) must still end it"


def test_supervisor_restart_budget_is_per_crash_loop_not_per_session(tmp_path, monkeypatch):
    script = _supervisor_text(tmp_path, monkeypatch)
    assert str(schedule.HEALTHY_RUN_S) in script
    assert "$attempt = 0 }" in script, "a long healthy run must reset the budget"


def test_supervisor_only_watches_a_collector_in_its_own_session(tmp_path, monkeypatch):
    # The collector mutex is per session, so on a shared PC another user's
    # collector must not count as ours: watching it would keep this user's
    # collection off for as long as the other user stays logged in.
    script = _supervisor_text(tmp_path, monkeypatch)
    assert "$session = [Diagnostics.Process]::GetCurrentProcess().SessionId" in script
    assert "$_.SessionId -eq $session" in script


def _powershell():
    if sys.platform != "win32":
        return None
    return shutil.which("powershell")


def _launch_supervisor(tmp_path, monkeypatch, collector):
    """Run the real generated script, with short timings and its own mutex."""
    monkeypatch.setattr(schedule, "SUPERVISOR_POLL_S", 1)
    # Never contend with a supervisor that is really running on this machine.
    monkeypatch.setattr(schedule, "SUPERVISOR_MUTEX", f"Local\\RCATest-{uuid.uuid4().hex}")
    monkeypatch.setattr(config, "app_dir", lambda: tmp_path / "RCA")
    script = schedule._write_supervisor(collector)
    env = dict(os.environ, LOCALAPPDATA=str(tmp_path))
    process = subprocess.Popen(
        [_powershell(), "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    return script, process


def _fake_collector(path):
    """A stand-in collector: records that it ran, then asks to be stopped."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "@echo off\r\n"
        'echo ran>>"%~dp0ran.txt"\r\n'
        'type nul > "%LOCALAPPDATA%\\RCA\\stop.flag"\r\n',
        encoding="ascii",
    )


@pytest.mark.skipif(_powershell() is None, reason="needs Windows PowerShell")
def test_supervisor_waits_out_a_rebuild_and_then_starts_the_collector(tmp_path, monkeypatch):
    # Not named RCA-Collector: the supervisor watches a running process of
    # that name instead of launching, and one may be running on this machine.
    collector = tmp_path / "dist" / "RCA-Collector" / "FakeCollector.cmd"
    (tmp_path / "RCA").mkdir()
    _, process = _launch_supervisor(tmp_path, monkeypatch, collector)
    try:
        # The executable is missing, as it is mid-rebuild. The old supervisor
        # was gone within a second of this; the new one must still be here.
        time.sleep(4)
        assert process.poll() is None, (
            "supervisor exited while the collector was temporarily missing: "
            + process.stderr.read().decode(errors="replace")
        )

        _fake_collector(collector)                 # the rebuild finishes
        process.wait(timeout=30)
        assert (collector.parent / "ran.txt").exists(), "the collector was never restarted"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


@pytest.mark.skipif(_powershell() is None, reason="needs Windows PowerShell")
def test_supervisor_leaves_when_unregistered_while_waiting(tmp_path, monkeypatch):
    """Waiting must not outlive an uninstall: unregister deletes the script."""
    collector = tmp_path / "gone" / "RCA-Collector.exe"
    (tmp_path / "RCA").mkdir()
    script, process = _launch_supervisor(tmp_path, monkeypatch, collector)
    try:
        time.sleep(3)
        assert process.poll() is None
        script.unlink()
        assert process.wait(timeout=15) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


# --- restarting --------------------------------------------------------------

_OLD_TEMPLATE = "if (-not (Test-Path $collector)) { break }  # old supervisor\n"


def _installed(tmp_path, monkeypatch):
    collector = tmp_path / "RCA-Collector" / "RCA-Collector.exe"
    collector.parent.mkdir(parents=True)
    collector.write_text("x")
    monkeypatch.setattr(config, "app_dir", lambda: tmp_path / "RCA")
    monkeypatch.setattr(schedule, "collector_executable", lambda: collector)
    schedule.supervisor_path().parent.mkdir(parents=True)
    schedule.supervisor_path().write_text(_OLD_TEMPLATE, encoding="utf-8")
    launched = []
    monkeypatch.setattr(schedule.subprocess, "Popen", lambda argv, **kw: launched.append(argv))
    return launched


def test_restart_relaunches_supervision_and_the_collector(tmp_path, monkeypatch):
    launched = _installed(tmp_path, monkeypatch)
    started = []
    monkeypatch.setattr(schedule, "start_now", lambda: started.append(1) or True)
    config.stop_flag_path().touch()            # a stale leftover would stop it at once

    assert schedule.restart_collection() is True
    assert not config.stop_flag_path().exists()
    joined = [" ".join(map(str, argv)) for argv in launched]
    assert any("supervise.ps1" in argv for argv in joined), joined
    assert started, "the collector must start now, not after a supervisor's backoff"


def test_relaunching_supervision_installs_the_fixed_template(tmp_path, monkeypatch):
    """The script is written once at registration; the fix must reach it."""
    _installed(tmp_path, monkeypatch)
    schedule.resume_collection()
    script = schedule.supervisor_path().read_text(encoding="utf-8")
    assert "old supervisor" not in script
    assert str(schedule.MISSING_COLLECTOR_GRACE_S) in script


def test_refresh_does_not_resurrect_an_unregistered_supervisor(tmp_path, monkeypatch):
    _installed(tmp_path, monkeypatch)
    schedule.supervisor_path().unlink()
    assert schedule.refresh_supervisor() is False
    assert not schedule.supervisor_path().exists()


# --- the interface -----------------------------------------------------------

def _summary(age_s: float) -> dict:
    last = pd.Timestamp(time.time() - age_s, unit="s", tz="UTC")
    return {
        "path": "telemetry.db", "exists": True, "size_bytes": 1024,
        "samples": 2880, "events": 0, "proc_samples": 0, "gaps": 0,
        "first_ts": last - pd.Timedelta(days=1), "last_ts": last,
        "latest": {}, "available": set(), "sampling_gaps": 0, "gap_hours": 0.0,
        "expected_samples": 2880, "coverage_pct": 99.0,
        "quarantined": 0, "quarantined_bytes": 0,
    }


@pytest.fixture
def make_view(qtbot, tmp_path, monkeypatch):
    from desktop.views import data_view

    monkeypatch.setattr(config, "app_dir", lambda: tmp_path)
    monkeypatch.setattr(config, "db_path", lambda: tmp_path / "telemetry.db")
    restarts = []
    monkeypatch.setattr(schedule, "restart_collection", lambda: restarts.append(1) or True)

    def build(age_s):
        monkeypatch.setattr(data_view, "store_summary", lambda: _summary(age_s))
        view = data_view.DataView()
        qtbot.addWidget(view)
        view.restarts = restarts
        return view

    return build


def test_a_stopped_collector_is_reported_not_described_as_collecting(make_view):
    view = make_view(age_s=60 * 60)                # the hour it went unnoticed

    assert "Collecting every" not in view.collection_state.text()
    assert "stopped" in view.collection_state.text()
    assert "60 minutes" in view.collection_state.text()
    # History says 99% coverage; that must not paint a dead collector green.
    assert view.summary_label.objectName() == "dataSummaryError"
    assert "stopped" in view.summary_label.text()
    assert view.restart_button.isVisibleTo(view)


def test_a_fresh_sample_reads_as_healthy(make_view):
    view = make_view(age_s=20)
    assert view.collection_state.text() == f"Collecting every {config.SYSTEM_CADENCE_S} seconds."
    assert view.summary_label.objectName() == "dataSummaryGood"
    assert not view.restart_button.isVisibleTo(view)


def test_a_pause_is_not_reported_as_a_failure(make_view):
    config.app_dir().mkdir(parents=True, exist_ok=True)
    config.stop_flag_path().touch()
    view = make_view(age_s=6 * 3600)
    assert "paused" in view.collection_state.text()
    assert view.summary_label.objectName() != "dataSummaryError"
    assert not view.restart_button.isVisibleTo(view)


def test_restart_is_one_click_and_says_it_is_waiting(make_view):
    view = make_view(age_s=60 * 60)
    view.restart_button.click()

    assert view.restarts == [1]
    assert "waiting for the first new sample" in view.collection_state.text()
    assert not view.restart_button.isEnabled(), "no double-launching while it starts"
    assert view.summary_label.objectName() == "dataSummaryWarn"


def test_a_failed_restart_is_said_out_loud(make_view, monkeypatch):
    view = make_view(age_s=60 * 60)
    monkeypatch.setattr(schedule, "restart_collection", lambda: False)
    view.restart_button.click()
    assert "Could not restart" in view.collection_state.text()
    assert view.collection_state.objectName() == "errorText"


def test_a_restart_that_does_not_take_is_admitted(make_view, monkeypatch):
    from desktop.views import data_view

    view = make_view(age_s=60 * 60)
    view.restart_button.click()
    # Past the grace period and still no new sample.
    view._restart_requested_at -= data_view.RESTART_GRACE_S + 1
    view.refresh()
    assert "still stopped after a restart" in view.collection_state.text()
    assert view.restart_button.isEnabled()
    assert view.summary_label.objectName() == "dataSummaryError"


def test_an_unreadable_database_is_not_called_a_stopped_collector(make_view, monkeypatch):
    from desktop.views import data_view

    view = make_view(age_s=60 * 60)

    def busy():
        raise RuntimeError("database is locked")

    monkeypatch.setattr(data_view, "store_summary", busy)
    view.refresh()
    assert not view.restart_button.isVisibleTo(view)
    assert view.summary_label.objectName() == "dataSummaryNeutral"


def test_the_error_style_exists_in_the_theme():
    from desktop import theme

    assert "QLabel#dataSummaryError" in theme.DARK_QSS


def test_age_is_described_in_units_a_person_would_use():
    from desktop.views.data_view import describe_age

    assert describe_age(60) == "1 minute"
    assert describe_age(45 * 60) == "45 minutes"
    assert describe_age(3 * 3600) == "3 hours"
    assert describe_age(3 * 86400) == "3 days"
