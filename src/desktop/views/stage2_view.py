"""Run RCA over a selected window of observed local telemetry."""

import pandas as pd
from PySide6.QtCore import QDateTime
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout, QSpinBox,
    QPushButton, QProgressBar, QLabel, QTableWidget, QTableWidgetItem,
    QTabWidget, QFileDialog, QPlainTextEdit, QComboBox, QDateTimeEdit,
    QHeaderView, QSizePolicy,
)

from desktop.workers import DetectIncidentsWorker, InferenceWorker, model_path
from desktop.views.graph_panel import PlotlyWebView
from pipeline import engine
from pipeline.visualizations import build_timeline_figure, draw_causal_graph

CUSTOM_RANGE = "Custom range …"


class Stage2View(QWidget):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self.worker = None
        self.detect_worker = None
        self._last_payload = None
        self._model_stale = False
        layout = QVBoxLayout(self)
        config = QGroupBox("Observed Incident Window")
        form = QFormLayout()

        incident_row = QHBoxLayout()
        self.incident_combo = QComboBox()
        self.incident_combo.addItem(CUSTOM_RANGE, None)
        self.incident_combo.currentIndexChanged.connect(self._on_incident_changed)
        self.refresh_button = QPushButton("Find Incidents")
        self.refresh_button.clicked.connect(self._on_refresh_incidents)
        incident_row.addWidget(self.incident_combo, stretch=3)
        incident_row.addWidget(self.refresh_button, stretch=1)
        form.addRow("Detected incident", incident_row)

        self.start_edit = QDateTimeEdit(QDateTime.currentDateTime().addSecs(-24 * 3600))
        self.end_edit = QDateTimeEdit(QDateTime.currentDateTime())
        for edit in (self.start_edit, self.end_edit):
            edit.setCalendarPopup(True)
            edit.setDisplayFormat("yyyy-MM-dd HH:mm")
        form.addRow("Range start", self.start_edit)
        form.addRow("Range end", self.end_edit)

        self.lag_spin = QSpinBox()
        self.lag_spin.setRange(2, 10)
        self.lag_spin.setValue(5)
        form.addRow("Granger Max Lag", self.lag_spin)
        config.setLayout(form)
        layout.addWidget(config)

        self.model_warning = QLabel("")
        self.model_warning.setObjectName("warningBanner")
        self.model_warning.setWordWrap(True)
        self.model_warning.setVisible(False)
        layout.addWidget(self.model_warning)

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
        # Fill the panel rather than leaving the columns huddled on the left,
        # and let the last column take the slack.
        header = self.root_cause_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.Stretch)
        header.setStretchLastSection(True)
        self.root_cause_table.verticalHeader().setVisible(False)
        self.root_cause_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.results_tabs.addTab(self.root_cause_table, "Root Causes")

        self.graph_view = PlotlyWebView(
            title="Causal Graph",
            legend=(
                "Each circle is an anomalous metric; colour runs from red for the "
                "highest-ranked candidate through orange for the rest. An arrow means "
                "Granger causality survived FDR correction and the effect-size floor, "
                "and is labelled with the lag at which it was strongest. "
                "<b>No arrows means no causal link was established</b> — which can be "
                "because none exists, or because the window was too short to test one."
            ),
        )
        self.results_tabs.addTab(self.graph_view, "Causal Graph")
        self.timeline_view = PlotlyWebView(
            title="Anomaly Timeline",
            legend=(
                "One line per metric, scaled 0–1 so unrelated units can share an axis, "
                "for the five most anomalous metrics. Each red dashed vertical line "
                "marks when that metric first crossed its anomaly threshold; their "
                "left-to-right order is the ordering the root-cause ranking treats as "
                "temporal priority."
            ),
        )
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
        # Only ever called with True after Stage 1 finishes training, so a
        # freshly trained model clears any latched staleness.
        if enabled:
            self._model_stale = False
        self.refresh_button.setEnabled(enabled)
        self._apply_model_gate(enabled)

    def _apply_model_gate(self, stage_enabled: bool):
        """Refuse to run RCA without a usable model.

        A missing model is known cheaply. Drift is only known after a run, so a
        run that reports staleness latches the gate closed until retraining.
        """
        status = engine.model_status(model_path())
        if not status.exists:
            self.model_warning.setText(
                f"RCA unavailable — {status.reason} Train a model in Stage 1 first."
            )
            self.model_warning.setVisible(True)
            self.run_button.setEnabled(False)
            return
        if self._model_stale:
            self.model_warning.setText(
                "RCA disabled — the model no longer matches current usage "
                f"(reconstruction error drifted past {engine.STALENESS_RATIO}x its "
                "training-time reference). Retrain in Stage 1."
            )
            self.model_warning.setVisible(True)
            self.run_button.setEnabled(False)
            return
        self.model_warning.setVisible(False)
        self.run_button.setEnabled(stage_enabled)

    def _on_incident_changed(self):
        """Custom range fields only apply when no detected incident is chosen."""
        incident = self.incident_combo.currentData()
        is_custom = incident is None
        self.start_edit.setEnabled(is_custom)
        self.end_edit.setEnabled(is_custom)
        if incident is not None:
            self.start_edit.setDateTime(QDateTime.fromSecsSinceEpoch(int(incident.start.timestamp())))
            self.end_edit.setDateTime(QDateTime.fromSecsSinceEpoch(int(incident.end.timestamp())))

    def _on_refresh_incidents(self):
        self.refresh_button.setEnabled(False)
        self.status_label.setText("Scanning collected history for incidents …")
        self.detect_worker = DetectIncidentsWorker()
        self.detect_worker.finished_ok.connect(self._on_incidents_found)
        self.detect_worker.failed.connect(self._on_incidents_failed)
        self.detect_worker.start()

    def _on_incidents_found(self, incidents):
        self.incident_combo.blockSignals(True)
        self.incident_combo.clear()
        for incident in incidents:
            self.incident_combo.addItem(
                f"{incident.start:%Y-%m-%d %H:%M}  {incident.duration_minutes:.0f}m  "
                f"[{incident.trigger}]  {incident.label}",
                incident,
            )
        self.incident_combo.addItem(CUSTOM_RANGE, None)
        self.incident_combo.blockSignals(False)
        self.incident_combo.setCurrentIndex(0)
        self._on_incident_changed()
        self.status_label.setText(
            f"{len(incidents)} incident(s) found." if incidents
            else "No incidents found in the retained history."
        )
        self.refresh_button.setEnabled(True)

    def _on_incidents_failed(self, message):
        self.status_label.setText(f"Incident scan failed: {message}")
        self.refresh_button.setEnabled(True)

    def _on_run_clicked(self):
        self.run_button.setEnabled(False)
        self.progress_bar.setValue(0)
        incident = self.incident_combo.currentData()
        if incident is not None:
            start, end, trigger = incident.start, incident.end, incident.trigger
        else:
            start = pd.Timestamp(self.start_edit.dateTime().toSecsSinceEpoch(), unit="s", tz="UTC")
            end = pd.Timestamp(self.end_edit.dateTime().toSecsSinceEpoch(), unit="s", tz="UTC")
            trigger = "custom range"
        self.worker = InferenceWorker(24, self.lag_spin.value(), start=start, end=end, trigger=trigger)
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
        # No resizeColumnsToContents(): the header is in Stretch mode so the
        # columns fill the panel instead of huddling at the left edge.
        graph = payload["causal_results"]["causal_graph"]
        self.graph_view.show_figure(draw_causal_graph(graph, root_causes[0]["metric"] if root_causes else ""))
        self.timeline_view.show_figure(build_timeline_figure(payload["incident_scaled"], payload["anomaly_scores"], payload["anomaly_times"]))
        self.report_text.setPlainText(payload["report"]["md_report"])
        evidence = payload.get("evidence", {})
        self.status_label.setText(
            f"RCA complete — {evidence.get('surviving_causal_edges', 0)} causal edge(s) "
            f"survived correction, {evidence.get('attributed_processes', 0)} process(es) attributed."
        )
        # Latch the gate closed if this run showed the model has drifted.
        self._model_stale = bool(payload.get("model_stale"))
        self._apply_model_gate(True)

    def _on_failed(self, message):
        # RCA reports progress per stage, so a failure mid-analysis left the
        # bar parked at 40% or 55% -- reading as "still working" beside the
        # failure text.
        self.progress_bar.setValue(0)
        self.status_label.setText(f"Failed: {message}")
        self._apply_model_gate(True)

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
