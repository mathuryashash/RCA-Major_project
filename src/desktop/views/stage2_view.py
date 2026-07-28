"""Run RCA over a selected window of observed local telemetry."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout, QSpinBox,
    QPushButton, QProgressBar, QLabel, QTableWidget, QTableWidgetItem,
    QTabWidget, QFileDialog, QPlainTextEdit,
)

from desktop.workers import InferenceWorker
from desktop.views.graph_panel import PlotlyWebView
from pipeline.visualizations import build_timeline_figure, draw_causal_graph


class Stage2View(QWidget):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self.worker = None
        self._last_payload = None
        layout = QVBoxLayout(self)
        self.locked_label = QLabel("Train a telemetry model in Stage 1 first.")
        layout.addWidget(self.locked_label)
        config = QGroupBox("Observed Incident Window")
        form = QFormLayout()
        self.hours_spin = QSpinBox()
        self.hours_spin.setRange(1, 168)
        self.hours_spin.setValue(24)
        form.addRow("Lookback (hours)", self.hours_spin)
        self.lag_spin = QSpinBox()
        self.lag_spin.setRange(2, 10)
        self.lag_spin.setValue(5)
        form.addRow("Granger Max Lag", self.lag_spin)
        config.setLayout(form)
        layout.addWidget(config)
        self.run_button = QPushButton("Run RCA on Collected Telemetry")
        self.run_button.setObjectName("primaryAction")
        self.run_button.clicked.connect(self._on_run_clicked)
        layout.addWidget(self.run_button)
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("")
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.status_label)
        self.results_tabs = QTabWidget()
        self.root_cause_table = QTableWidget()
        self.root_cause_table.setColumnCount(6)
        self.root_cause_table.setHorizontalHeaderLabels(["Rank", "Metric", "Score", "Confidence", "Outflow", "Downstream"])
        self.results_tabs.addTab(self.root_cause_table, "Root Causes")
        self.graph_view = PlotlyWebView()
        self.results_tabs.addTab(self.graph_view, "Causal Graph")
        self.timeline_view = PlotlyWebView()
        self.results_tabs.addTab(self.timeline_view, "Anomaly Timeline")
        self.report_text = QPlainTextEdit()
        self.report_text.setReadOnly(True)
        self.results_tabs.addTab(self.report_text, "Report")
        layout.addWidget(self.results_tabs, stretch=1)
        exports = QHBoxLayout()
        self.export_md_button = QPushButton("Export Markdown Report")
        self.export_json_button = QPushButton("Export JSON Report")
        self.export_md_button.clicked.connect(self._export_md)
        self.export_json_button.clicked.connect(self._export_json)
        exports.addWidget(self.export_md_button)
        exports.addWidget(self.export_json_button)
        layout.addLayout(exports)
        self.set_enabled(False)

    def set_enabled(self, enabled):
        self.locked_label.setVisible(not enabled)
        self.run_button.setEnabled(enabled)

    def _on_run_clicked(self):
        self.run_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.worker = InferenceWorker(self.hours_spin.value(), self.lag_spin.value())
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_progress(self, pct, message):
        self.progress_bar.setValue(pct)
        self.status_label.setText(message)

    def _on_finished(self, payload):
        self._last_payload = payload
        self.state.last_causal_results = payload["causal_results"]
        self.state.last_root_causes = payload["root_causes"]
        self.state.last_incident_scaled = payload["incident_scaled"]
        self.state.last_anomaly_scores = payload["anomaly_scores"]
        self.state.last_anomaly_times = payload["anomaly_times"]
        self.state.last_report = payload["report"]
        root_causes = payload["root_causes"]
        self.root_cause_table.setRowCount(len(root_causes))
        for row, rc in enumerate(root_causes):
            values = [str(rc["rank"]), rc["metric"], f"{rc['composite_score']:.4f}", rc["confidence"],
                      f"{rc.get('scores_breakdown', {}).get('causal_outflow', 0):.3f}",
                      ", ".join(rc.get("downstream_effects", [])) or "—"]
            for col, value in enumerate(values):
                self.root_cause_table.setItem(row, col, QTableWidgetItem(value))
        self.root_cause_table.resizeColumnsToContents()
        graph = payload["causal_results"]["causal_graph"]
        self.graph_view.show_figure(draw_causal_graph(graph, root_causes[0]["metric"] if root_causes else ""))
        self.timeline_view.show_figure(build_timeline_figure(payload["incident_scaled"], payload["anomaly_scores"], payload["anomaly_times"]))
        self.report_text.setPlainText(payload["report"]["md_report"])
        self.status_label.setText("RCA complete")
        self.run_button.setEnabled(True)

    def _on_failed(self, message):
        self.status_label.setText(f"Failed: {message}")
        self.run_button.setEnabled(True)

    def _export_md(self):
        if not self._last_payload:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Markdown Report", "report.md", "Markdown (*.md)")
        if path:
            with open(path, "w", encoding="utf-8") as report:
                report.write(self._last_payload["report"]["md_report"])

    def _export_json(self):
        if not self._last_payload:
            return
        import json
        path, _ = QFileDialog.getSaveFileName(self, "Export JSON Report", "report.json", "JSON (*.json)")
        if path:
            with open(path, "w", encoding="utf-8") as report:
                json.dump(self._last_payload["report"]["json_report"], report, indent=2, default=str)
