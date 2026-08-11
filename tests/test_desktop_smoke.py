import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from PySide6.QtCore import Qt

from desktop.main_window import MainWindow


def test_main_window_boots(qtbot, monkeypatch):
    # Pin the model state: whether one exists on the developer's machine must
    # not decide whether this test passes.
    from pipeline import engine

    monkeypatch.setattr(engine, "model_status", lambda path: engine.ModelStatus(
        exists=False, reason="No model has been trained yet."))

    window = MainWindow()
    qtbot.addWidget(window)
    # The version rides in the title so a bug report identifies its build.
    from version import __version__

    assert window.windowTitle().startswith("AI-Powered Root Cause Analysis")
    assert __version__ in window.windowTitle()
    assert window.tabs.count() == 3          # Captured Data, Stage 1, Stage 2
    assert window.tabs.tabText(0) == "Captured Data"
    assert window.state.model_trained is False


def test_stage2_stays_locked_without_a_model(qtbot, monkeypatch):
    """Enabling the stage is not enough: RCA needs a usable model artifact."""
    from pipeline import engine

    monkeypatch.setattr(engine, "model_status", lambda path: engine.ModelStatus(
        exists=False, reason="No model has been trained yet."))

    window = MainWindow()
    qtbot.addWidget(window)
    assert window.stage2.run_button.isEnabled() is False

    window.state.model_trained = True
    window.stage2.set_enabled(True)
    assert window.stage2.run_button.isEnabled() is False
    # isVisible() would be False purely because the window was never shown.
    assert window.stage2.model_warning.isVisibleTo(window.stage2) is True
    assert "Train a model in Stage 1" in window.stage2.model_warning.text()


def test_stage2_unlocks_once_a_model_exists(qtbot, monkeypatch):
    from pipeline import engine

    monkeypatch.setattr(engine, "model_status", lambda path: engine.ModelStatus(
        exists=True, age_days=1.0))

    window = MainWindow()
    qtbot.addWidget(window)
    window.stage2.set_enabled(True)
    assert window.stage2.run_button.isEnabled() is True


def test_stage1_train_button_gated_on_uninterrupted_baseline(qtbot, monkeypatch):
    from pipeline import engine
    from telemetry.analysis import BaselineStatus

    monkeypatch.setattr(engine, "baseline_readiness", lambda path, window_size=12: BaselineStatus(
        clean_samples=100, clean_days=1.0,
        uninterrupted_samples=100, current_run_samples=100, required_samples=2512,
        ready=False, days_remaining=2.0))

    window = MainWindow()
    qtbot.addWidget(window)
    window.stage1.refresh_status()
    assert window.stage1.train_button.isEnabled() is False
    assert "2.00 days remaining" in window.stage1.remaining_label.text()


def test_stage1_train_button_triggers_worker_once_ready(qtbot, monkeypatch):
    import desktop.workers as workers_module
    from pipeline import engine
    from telemetry.analysis import BaselineStatus

    monkeypatch.setattr(engine, "baseline_readiness", lambda path, window_size=12: BaselineStatus(
        clean_samples=8640, clean_days=3.0,
        uninterrupted_samples=8640, current_run_samples=8640, required_samples=2512,
        ready=True, days_remaining=0.0))
    monkeypatch.setattr(workers_module.TrainWorker, "start", lambda self: None)

    window = MainWindow()
    qtbot.addWidget(window)
    window.stage1.refresh_status()
    assert window.stage1.train_button.isEnabled() is True

    qtbot.mouseClick(window.stage1.train_button, Qt.LeftButton)
    assert window.stage1.worker is not None


def test_stage2_unlocks_from_a_model_trained_in_an_earlier_session(qtbot, monkeypatch):
    """Reopening the app must not discard a model that already exists.

    Stage 2 used to unlock only via the model_trained signal, so a model
    trained yesterday left the tab locked until the user retrained today.
    """
    from pipeline import engine

    monkeypatch.setattr(engine, "model_status", lambda path: engine.ModelStatus(
        exists=True, age_days=0.5))

    window = MainWindow()
    qtbot.addWidget(window)

    assert window.state.model_trained is True
    assert window.stage2.run_button.isEnabled() is True


def test_windowed_launch_without_std_handles_gets_valid_ones(tmp_path):
    """A console=False frozen build starts with fds 1 and 2 invalid.

    The packaged app died with 0xC0000409 in Qt6Core about forty seconds in,
    silently, and runs with this in place. Redirecting its output also made
    the crash disappear, so it never reproduced from source or in a console
    build. The mechanism is not established -- it is not a slot exception
    reaching qFatal, since PySide6 measurably survives those.
    """
    import subprocess

    src = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    result_path = tmp_path / "result.txt"
    probe = tmp_path / "probe.py"
    probe.write_text(
        "import os, sys\n"
        f"sys.path.insert(0, {src!r})\n"
        "for fd in (1, 2):\n"
        "    os.close(fd)\n"
        "sys.stdout = None\n"
        "sys.stderr = None\n"
        "import desktop.main\n"                 # runs _ensure_std_handles on import
        "ok = True\n"
        "for fd in (1, 2):\n"
        "    try:\n"
        "        os.fstat(fd)\n"
        "    except OSError:\n"
        "        ok = False\n"
        "ok = ok and sys.stdout is not None and sys.stderr is not None\n"
        "print('written to a real handle')\n"   # would raise before the fix
        f"open({str(result_path)!r}, 'w').write('ok' if ok else 'bad')\n",
        encoding="utf-8",
    )

    subprocess.run([sys.executable, str(probe)], timeout=300, check=True)
    assert result_path.read_text() == "ok"


def test_unhandled_exceptions_reach_the_collector_log(tmp_path, monkeypatch):
    """A failure in a timer must land on disk, not in the void.

    PySide6 keeps running after an exception in a slot; it prints a traceback
    and carries on. A windowed build has nowhere to print, so a view that
    fails every thirty seconds does so invisibly and merely looks frozen.
    """
    import logging

    from desktop import main as desktop_main

    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(self.format(record))

    handler = _Capture()
    logging.getLogger("desktop").addHandler(handler)
    original = sys.excepthook
    try:
        # The hook chains to whatever it replaced. Pin that to a no-op first:
        # pytest-qt installs a hook that fails the test on any exception it
        # sees, which would flag the one deliberately raised here.
        sys.excepthook = lambda *args: None
        desktop_main._install_crash_logging()
        try:
            raise RuntimeError("timer blew up")
        except RuntimeError:
            sys.excepthook(*sys.exc_info())
    finally:
        sys.excepthook = original
        logging.getLogger("desktop").removeHandler(handler)

    assert any("timer blew up" in text for text in records), records


def test_crash_log_is_not_the_collectors_log_file():
    """Two processes cannot share one RotatingFileHandler on Windows.

    Rollover renames the file, which fails while the collector holds it open,
    and logging swallows that error -- measured, six records written and three
    survive, losing lines from both writers.
    """
    from telemetry import config

    assert config.desktop_log_path() != config.log_path()


def test_workers_do_not_shadow_qthread_start():
    """Assigning to self.start replaces the method that launches the thread.

    InferenceWorker stored the window start as self.start, so worker.start()
    called a Timestamp instead of QThread.start. The click handler raised
    TypeError, no worker ever ran, and Stage 2 sat at 0% looking like a hung
    analysis. Training was unaffected only because its worker takes no such
    argument -- which is exactly how the outage stayed hidden.
    """
    import pandas as pd

    from desktop.workers import DetectIncidentsWorker, InferenceWorker, TrainWorker

    workers = [
        InferenceWorker(24, 5,
                        start=pd.Timestamp("2026-08-01", tz="UTC"),
                        end=pd.Timestamp("2026-08-02", tz="UTC")),
        TrainWorker(5, 12),
        DetectIncidentsWorker(),
    ]
    for worker in workers:
        assert callable(worker.start), f"{type(worker).__name__}.start is not callable"
        assert callable(worker.quit), f"{type(worker).__name__}.quit is not callable"


def test_figures_can_be_opened_full_screen_with_a_legend(qtbot, monkeypatch):
    """Both figures are unreadable in a short tab and need explaining.

    The graph and timeline each carry a caption saying what is drawn, and a
    button that opens the same figure on its own with a close control.
    """
    import plotly.graph_objects as go
    from pipeline import engine

    monkeypatch.setattr(engine, "model_status", lambda path: engine.ModelStatus(
        exists=True, age_days=1.0))

    window = MainWindow()
    qtbot.addWidget(window)

    from desktop.views import graph_panel

    opened = []
    # exec() would block the test on a modal dialog.
    monkeypatch.setattr(graph_panel._FullScreenFigure, "exec",
                        lambda self: opened.append(self))

    for panel in (window.stage2.graph_view, window.stage2.timeline_view):
        assert panel._legend, "every figure needs a legend explaining it"
        assert panel.expand_button.isEnabled()

        # Nothing plotted yet: expanding must be a no-op, not a crash.
        before = len(opened)
        panel.open_full_screen()
        assert len(opened) == before, "expanding an empty panel must do nothing"

        panel.show_figure(go.Figure())
        panel.open_full_screen()
        assert len(opened) == before + 1

    assert len(opened) == 2, "both figures must be openable full screen"
    for dialog in opened:
        assert dialog.windowTitle()                      # names what is shown
        assert dialog.isFullScreen()
        buttons = [b.text() for b in dialog.findChildren(type(window.stage2.run_button))]
        assert any("Close" in text for text in buttons), buttons
        # Close here rather than handing these to qtbot: it holds weak
        # references and closes at teardown, by which point Python has already
        # dropped the last reference and the C++ object is gone.
        dialog.close()


def test_root_cause_table_columns_fill_the_panel(qtbot, monkeypatch):
    """The columns huddled at the left, leaving the panel mostly empty."""
    from PySide6.QtWidgets import QHeaderView
    from pipeline import engine

    monkeypatch.setattr(engine, "model_status", lambda path: engine.ModelStatus(
        exists=True, age_days=1.0))

    window = MainWindow()
    qtbot.addWidget(window)
    header = window.stage2.root_cause_table.horizontalHeader()
    assert header.sectionResizeMode(0) == QHeaderView.Stretch


def test_consent_is_asked_in_the_interface_and_gates_collection(qtbot, tmp_path, monkeypatch):
    """Consent could only be given on a command line.

    Someone who opened the desktop app and never read the README was never
    asked, so nothing was ever collected. Declining must also be honoured.
    """
    from PySide6.QtWidgets import QDialog

    from desktop import consent as consent_module
    from telemetry import collector, config, store

    monkeypatch.setattr(config, "app_dir", lambda: tmp_path)
    monkeypatch.setattr(config, "db_path", lambda: tmp_path / "telemetry.db")

    monkeypatch.setattr(consent_module.QDialog, "exec", lambda self: QDialog.Rejected)
    assert consent_module.ensure_consent() is False, "declining must not start collection"

    connection = store.connect(tmp_path / "telemetry.db")
    try:
        assert collector.consent_granted(connection) is False
    finally:
        connection.close()

    monkeypatch.setattr(consent_module.QDialog, "exec", lambda self: QDialog.Accepted)
    assert consent_module.ensure_consent() is True

    connection = store.connect(tmp_path / "telemetry.db")
    try:
        assert collector.consent_granted(connection) is True
    finally:
        connection.close()

    # Already granted: no dialog, and still true.
    monkeypatch.setattr(consent_module.QDialog, "exec",
                        lambda self: pytest.fail("must not ask twice"))
    assert consent_module.ensure_consent() is True


def test_disclosure_names_what_is_collected():
    """A consent screen that does not say what it collects is not consent."""
    from desktop.consent import DISCLOSURE

    lowered = DISCLOSURE.lower()
    for expected in ("event log", "process", "network", "30 days", "never recorded"):
        assert expected in lowered, expected


def test_window_carries_the_application_icon(qtbot, monkeypatch):
    """A missing icon is cosmetic and must never stop the app opening."""
    from desktop.branding import app_icon, icon_path
    from pipeline import engine

    monkeypatch.setattr(engine, "model_status", lambda path: engine.ModelStatus(
        exists=True, age_days=1.0))

    assert icon_path().exists(), f"icon missing at {icon_path()}"
    assert not app_icon().isNull()

    window = MainWindow()
    qtbot.addWidget(window)
    assert not window.windowIcon().isNull()


def test_drift_warns_but_does_not_lock_the_stage(qtbot, monkeypatch):
    """One look at an old incident used to disable analysis entirely.

    Drift is measured against whichever window was analysed, so examining any
    older incident reports the model stale -- a property of that window, not
    of the model. Latching the run button off left no way to try a different
    range or lag without retraining first.
    """
    from pipeline import engine

    monkeypatch.setattr(engine, "model_status", lambda path: engine.ModelStatus(
        exists=True, age_days=1.0))

    window = MainWindow()
    qtbot.addWidget(window)
    window.stage2.set_enabled(True)
    assert window.stage2.run_button.isEnabled() is True

    window.stage2._model_stale = True
    window.stage2._apply_model_gate(True)

    assert window.stage2.run_button.isEnabled() is True, "a second run must stay possible"
    assert window.stage2.model_warning.isVisibleTo(window.stage2) is True
    assert "drift" in window.stage2.model_warning.text().lower()


def test_figure_panels_are_not_blank_before_a_run(qtbot, monkeypatch):
    """An untouched web view paints white, which reads as a broken chart."""
    from pipeline import engine

    monkeypatch.setattr(engine, "model_status", lambda path: engine.ModelStatus(
        exists=True, age_days=1.0))

    window = MainWindow()
    qtbot.addWidget(window)
    for panel in (window.stage2.graph_view, window.stage2.timeline_view):
        assert panel._figure is None            # nothing plotted yet
        assert hasattr(panel, "show_placeholder")


def test_full_screen_figure_fills_the_window(qtbot, monkeypatch):
    """The figures are built at a fixed height that left most of a screen blank.

    Expanding one has to drop that height and let it track the window, and
    must not disturb the copy still displayed in the tab behind the dialog.
    """
    import os

    import plotly.graph_objects as go
    from pipeline import engine

    monkeypatch.setattr(engine, "model_status", lambda path: engine.ModelStatus(
        exists=True, age_days=1.0))

    window = MainWindow()
    qtbot.addWidget(window)
    panel = window.stage2.timeline_view

    figure = go.Figure(go.Scatter(x=[1, 2, 3], y=[1, 2, 3]))
    figure.update_layout(height=420)

    panel._render(figure, panel.view, fill=False)
    tabbed = open(os.path.join(panel._tmp_dir, "figure_1.html"), encoding="utf-8").read()
    panel._render(figure, panel.view, fill=True)
    expanded = open(os.path.join(panel._tmp_dir, "figure_2.html"), encoding="utf-8").read()

    assert '"height":420' in tabbed.replace(" ", "")
    assert '"height":420' not in expanded.replace(" ", "")
    assert '"autosize":true' in expanded.replace(" ", "")
    assert figure.layout.height == 420, "the tab's own figure must be untouched"
    # The page body is white by default, which shows as a band around the plot.
    assert "background:#151a2e" in expanded


def test_agreeing_completes_the_install_without_a_command_line(qtbot, tmp_path, monkeypatch):
    """Consent used to leave the user with a PowerShell step to discover.

    Agreeing to continuous collection is agreeing to the thing that makes it
    continuous, so the dialog now finishes the setup -- and says so.
    """
    from desktop import main as desktop_main
    from telemetry import config, schedule

    monkeypatch.setattr(config, "app_dir", lambda: tmp_path)
    monkeypatch.setattr(config, "db_path", lambda: tmp_path / "telemetry.db")

    done = []
    monkeypatch.setattr(desktop_main, "ensure_consent", lambda: True, raising=False)
    monkeypatch.setattr(schedule, "start_now", lambda: done.append("start") or True)
    monkeypatch.setattr(schedule, "is_registered", lambda: False)
    monkeypatch.setattr(schedule, "register", lambda: done.append("logon") or True)
    monkeypatch.setattr(schedule, "register_uninstall_entry", lambda: done.append("arp") or True)
    monkeypatch.setattr(schedule, "create_start_menu_shortcut", lambda: done.append("menu") or True)
    monkeypatch.setattr("desktop.consent.ensure_consent", lambda parent=None: True)

    desktop_main._ensure_collector_running()

    assert done == ["start", "logon", "arp", "menu"], done


def test_declining_registers_nothing(qtbot, tmp_path, monkeypatch):
    """Declining must leave the machine exactly as it was found."""
    from desktop import main as desktop_main
    from telemetry import config, schedule

    monkeypatch.setattr(config, "app_dir", lambda: tmp_path)
    monkeypatch.setattr(config, "db_path", lambda: tmp_path / "telemetry.db")

    touched = []
    for name in ("start_now", "register", "register_uninstall_entry", "create_start_menu_shortcut"):
        monkeypatch.setattr(schedule, name, lambda *a, n=name: touched.append(n) or True)
    monkeypatch.setattr("desktop.consent.ensure_consent", lambda parent=None: False)

    desktop_main._ensure_collector_running()

    assert touched == [], touched


def test_disclosure_states_what_agreeing_registers():
    """Doing more than the dialog says would undermine the consent."""
    from desktop.consent import DISCLOSURE

    lowered = DISCLOSURE.lower()
    for expected in ("logon", "start menu", "remove", "administrator"):
        assert expected in lowered, expected


def test_a_missing_start_menu_shortcut_is_restored_on_launch(qtbot, tmp_path, monkeypatch):
    """Windows deletes shortcuts whose target has gone.

    Replacing the executable -- a rebuild, or extracting a new release over an
    old one -- removes the target long enough to qualify, and the entry
    disappears. An installed application that cannot be found by name is the
    same as an uninstalled one.
    """
    from desktop import main as desktop_main
    from telemetry import config, schedule

    monkeypatch.setattr(config, "app_dir", lambda: tmp_path)
    monkeypatch.setattr(config, "db_path", lambda: tmp_path / "telemetry.db")
    monkeypatch.setattr("desktop.consent.ensure_consent", lambda parent=None: True)
    monkeypatch.setattr(schedule, "start_now", lambda: True)

    remade = []
    # Already installed: the logon entry survived, only the shortcut is gone.
    monkeypatch.setattr(schedule, "is_registered", lambda: True)
    monkeypatch.setattr(schedule, "start_menu_shortcut_exists", lambda: False)
    monkeypatch.setattr(schedule, "create_start_menu_shortcut", lambda: remade.append(1) or True)
    monkeypatch.setattr(schedule, "register", lambda: pytest.fail("must not re-register autostart"))

    desktop_main._ensure_collector_running()
    assert remade == [1], "the shortcut should have been put back"

    # Present already: leave it alone.
    remade.clear()
    monkeypatch.setattr(schedule, "start_menu_shortcut_exists", lambda: True)
    desktop_main._ensure_collector_running()
    assert remade == []
