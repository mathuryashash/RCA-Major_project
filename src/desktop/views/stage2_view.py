"""Stage 2 tab: inject a failure scenario and run the full RCA pipeline."""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QComboBox, QSlider, QSpinBox, QPushButton, QProgressBar, QLabel,
    QTableWidget, QTableWidgetItem, QTabWidget, QFileDialog, QPlainTextEdit,
)
from PySide6.QtCore import Qt

from desktop.workers import InferenceWorker
from desktop.views.graph_panel import PlotlyWebView
from pipeline.visualizations import draw_causal_graph, build_timeline_figure

# Only the scenarios SyntheticMetricsGenerator.inject_failure_scenario actually
# implements — anything else returns an unmodified frame and no anomalies.
SCENARIO_DESCRIPTIONS = {
    "database_slow_query": "Simulates slow DB queries causing cascading latency and throughput drops",
    "memory_leak": "Gradual memory exhaustion leading to OOM errors and degraded performance",
    "cpu_spike": "CPU saturation from a runaway process, causing request queuing and timeouts",
}


class Stage2View(QWidget):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self.worker = None
        self._last_payload = None

        layout = QVBoxLayout(self)

        self.locked_label = QLabel("Train a model in Stage 1 first.")
        layout.addWidget(self.locked_label)

        config_box = QGroupBox("Failure Injection")
        form = QFormLayout()

        self.scenario_combo = QComboBox()
        self.scenario_combo.addItems(list(SCENARIO_DESCRIPTIONS.keys()))
        self.scenario_combo.currentTextChanged.connect(self._on_scenario_changed)
        form.addRow("Failure Scenario", self.scenario_combo)

        self.scenario_desc_label = QLabel(SCENARIO_DESCRIPTIONS[self.scenario_combo.currentText()])
        self.scenario_desc_label.setWordWrap(True)
        form.addRow("", self.scenario_desc_label)

        sev_row = QHBoxLayout()
        self.severity_slider = QSlider(Qt.Horizontal)
        self.severity_slider.setRange(1, 10)
        self.severity_slider.setValue(8)
        self.severity_value_label = QLabel("0.8")
        self.severity_slider.valueChanged.connect(
            lambda v: self.severity_value_label.setText(f"{v / 10:.1f}")
        )
        sev_row.addWidget(self.severity_slider, stretch=3)
        sev_row.addWidget(self.severity_value_label, stretch=1)
        form.addRow("Severity (0.1 – 1.0)", sev_row)

        self.lag_spin = QSpinBox()
        self.lag_spin.setRange(2, 10)
        self.lag_spin.setValue(5)
        form.addRow("Granger Max Lag", self.lag_spin)

        config_box.setLayout(form)
        layout.addWidget(config_box)

        self.run_button = QPushButton("Simulate Incident && Run Full RCA")
        self.run_button.setObjectName("primaryAction")
        self.run_button.clicked.connect(self._on_run_clicked)
        layout.addWidget(self.run_button)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        self.results_tabs = QTabWidget()

        self.root_cause_table = QTableWidget()
        self.root_cause_table.setColumnCount(6)
        self.root_cause_table.setHorizontalHeaderLabels(
            ["Rank", "Metric", "Composite Score", "Confidence", "Causal Outflow", "Downstream"]
        )
        self.results_tabs.addTab(self.root_cause_table, "Root Causes")

        self.graph_view = PlotlyWebView()
        self.results_tabs.addTab(self.graph_view, "Causal Graph")

        self.timeline_view = PlotlyWebView()
        self.results_tabs.addTab(self.timeline_view, "Anomaly Timeline")

        self.report_text = QPlainTextEdit()
        self.report_text.setReadOnly(True)
        self.results_tabs.addTab(self.report_text, "Markdown Report")

        layout.addWidget(self.results_tabs, stretch=1)

        export_row = QHBoxLayout()
        self.export_md_button = QPushButton("Export Markdown Report")
        self.export_json_button = QPushButton("Export JSON Report")
        self.export_md_button.clicked.connect(self._export_md)
        self.export_json_button.clicked.connect(self._export_json)
        export_row.addWidget(self.export_md_button)
        export_row.addWidget(self.export_json_button)
        layout.addLayout(export_row)

        self.set_enabled(False)

    def set_enabled(self, enabled: bool):
        self.locked_label.setVisible(not enabled)
        self.run_button.setEnabled(enabled)

    def _on_scenario_changed(self, scenario: str):
        self.scenario_desc_label.setText(SCENARIO_DESCRIPTIONS.get(scenario, ""))

    def _on_run_clicked(self):
        self.run_button.setEnabled(False)
        self.progress_bar.setValue(0)

        self.worker = InferenceWorker(
            normal_df=self.state.normal_df,
            feat_cols=self.state.feat_cols,
            detector=self.state.detector,
            failure_type=self.scenario_combo.currentText(),
            severity=self.severity_slider.value() / 10,
            max_granger_lag=self.lag_spin.value(),
            seed=self.state.seed,
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_progress(self, pct: int, message: str):
        self.progress_bar.setValue(pct)
        self.status_label.setText(message)

    def _on_finished(self, payload: dict):
        self._last_payload = payload
        self.state.last_causal_results = payload["causal_results"]
        self.state.last_root_causes = payload["root_causes"]
        self.state.last_incident_scaled = payload["incident_scaled"]
        self.state.last_anomaly_scores = payload["anomaly_scores"]
        self.state.last_anomaly_times = payload["anomaly_times"]
        self.state.last_metadata = payload["metadata"]
        self.state.last_report = payload["report"]

        root_causes = payload["root_causes"]
        self.root_cause_table.setRowCount(len(root_causes))
        for row, rc in enumerate(root_causes):
            downstream = rc.get("downstream_effects", [])
            downstream_str = ", ".join(downstream[:3]) + (f" (+{len(downstream) - 3} more)" if len(downstream) > 3 else "")
            values = [
                str(rc["rank"]), rc["metric"], f"{rc['composite_score']:.4f}",
                rc["confidence"], f"{rc.get('scores_breakdown', {}).get('causal_outflow', 0):.3f}",
                downstream_str or "—",
            ]
            for col, val in enumerate(values):
                self.root_cause_table.setItem(row, col, QTableWidgetItem(val))
        self.root_cause_table.resizeColumnsToContents()

        causal_graph = payload["causal_results"]["causal_graph"]
        top_metric = root_causes[0]["metric"] if root_causes else ""
        self.graph_view.show_figure(draw_causal_graph(causal_graph, top_metric))

        self.timeline_view.show_figure(build_timeline_figure(
            payload["incident_scaled"], payload["anomaly_scores"], payload["anomaly_times"]
        ))

        self.report_text.setPlainText(payload["report"]["md_report"])

        self.run_button.setEnabled(True)
        self.status_label.setText("Pipeline complete")

    def _on_failed(self, message: str):
        self.status_label.setText(f"Failed: {message}")
        self.run_button.setEnabled(True)

    def _export_md(self):
        if not self._last_payload:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export Markdown Report", "report.md", "Markdown (*.md)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._last_payload["report"]["md_report"])

    def _export_json(self):
        if not self._last_payload:
            return
        import json
        path, _ = QFileDialog.getSaveFileName(self, "Export JSON Report", "report.json", "JSON (*.json)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self._last_payload["report"]["json_report"], f, indent=2, default=str)
