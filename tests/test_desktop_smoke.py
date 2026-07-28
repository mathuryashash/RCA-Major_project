import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from PySide6.QtCore import Qt

from desktop.main_window import MainWindow


def test_main_window_boots(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    assert window.windowTitle() == "AI-Powered Root Cause Analysis"
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


def test_stage2_relocks_after_a_run_reports_a_stale_model(qtbot, monkeypatch):
    """Drift is only measurable during a run, so the gate latches afterwards."""
    from pipeline import engine

    monkeypatch.setattr(engine, "model_status", lambda path: engine.ModelStatus(
        exists=True, age_days=1.0))

    window = MainWindow()
    qtbot.addWidget(window)
    window.stage2.set_enabled(True)
    assert window.stage2.run_button.isEnabled() is True

    window.stage2._model_stale = True
    window.stage2._apply_model_gate(True)
    assert window.stage2.run_button.isEnabled() is False
    assert "Retrain" in window.stage2.model_warning.text()


def test_stage1_train_button_gated_on_uninterrupted_baseline(qtbot, monkeypatch):
    from pipeline import engine
    from telemetry.analysis import BaselineStatus

    monkeypatch.setattr(engine, "baseline_readiness", lambda path: BaselineStatus(
        clean_samples=100, clean_days=1.0,
        uninterrupted_samples=100, required_samples=2512,
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

    monkeypatch.setattr(engine, "baseline_readiness", lambda path: BaselineStatus(
        clean_samples=8640, clean_days=3.0,
        uninterrupted_samples=8640, required_samples=2512,
        ready=True, days_remaining=0.0))
    monkeypatch.setattr(workers_module.TrainWorker, "start", lambda self: None)

    window = MainWindow()
    qtbot.addWidget(window)
    window.stage1.refresh_status()
    assert window.stage1.train_button.isEnabled() is True

    qtbot.mouseClick(window.stage1.train_button, Qt.LeftButton)
    assert window.stage1.worker is not None
