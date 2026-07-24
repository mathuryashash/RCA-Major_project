import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from PySide6.QtCore import Qt

from desktop.main_window import MainWindow


def test_main_window_boots(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    assert window.windowTitle() == "AI-Powered Root Cause Analysis"
    assert window.tabs.count() == 2
    assert window.state.model_trained is False


def test_stage2_locked_until_trained(qtbot):
    window = MainWindow()
    qtbot.addWidget(window)
    assert window.stage2.run_button.isEnabled() is False

    window.state.model_trained = True
    window.stage2.set_enabled(True)
    assert window.stage2.run_button.isEnabled() is True


def test_stage1_train_button_triggers_worker(qtbot, monkeypatch):
    import desktop.workers as workers_module

    monkeypatch.setattr(workers_module.TrainWorker, "start", lambda self: None)

    window = MainWindow()
    qtbot.addWidget(window)
    qtbot.mouseClick(window.stage1.train_button, Qt.LeftButton)

    assert window.stage1.worker is not None
