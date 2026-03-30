"""
PipelineOrchestrator — chains all RCA modules into a single analysis flow.

Flow (enhanced with metric + deep-learning modules):
  1. Read logs (M1A) + Read metrics from CSV / TimescaleDB (M1B)
  2. Parse log templates (M2A) + Preprocess metrics (M2B)
  3. Detect log anomalies via TF-IDF + LogBERT ensemble (M3)
  4. Detect metric anomalies via LSTM AE + Temporal Transformer (M3)
  5. Compute unified anomaly scores (M3)
  6. Build causal graph via Granger + FDR + PC Algorithm (M4)
  7. KHBN posterior refinement (M4)
  8. 5-factor RCA scoring (M4/M5)
  9. Generate NLG narrative (M5) + Remediation plan (M7)

All new modules degrade gracefully — the pipeline still works with only
the original TF-IDF + Granger baseline when optional deps are missing.
"""

import glob as glob_mod
import os
import re
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

from src.preprocessing.log_parser import LogTemplateExtractor
from src.models.anomaly_detector import SimpleLogAnomalyDetector, MetricAnomalyDetector
from src.models.causal_inference import CausalInferenceEngine
from src.reporting.nlg_generator import NLGGenerator
from src.remediation.remediation_engine import RemediationEngine
from src.common.config import load_config
from src.models.unified_scorer import UnifiedAnomalyScorer
from src.api.exceptions import PipelineError, ModelLoadError

# --- Graceful imports for optional heavy modules ---

try:
    from src.models.logbert import LogBERTDetector, TRANSFORMERS_AVAILABLE
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from src.models.deep_learning import MetricDeepAnomalyDetector

    LSTM_AE_AVAILABLE = True
except ImportError:
    LSTM_AE_AVAILABLE = False

try:
    from src.models.temporal_transformer import TemporalTransformerDetector

    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False

try:
    from src.preprocessing.metric_preprocessor import MetricPreprocessor

    PREPROCESSOR_AVAILABLE = True
except ImportError:
    PREPROCESSOR_AVAILABLE = False

try:
    from src.models.khbn import KHBNModel

    KHBN_AVAILABLE = True
except ImportError:
    KHBN_AVAILABLE = False

try:
    from src.ingestion.metric_ingest import MetricIngestionService

    METRIC_INGEST_AVAILABLE = True
except ImportError:
    METRIC_INGEST_AVAILABLE = False

try:
    from src.ingestion.log_ingest import LogFileWatcher, read_full_file

    LOG_WATCHER_AVAILABLE = True
except ImportError:
    LOG_WATCHER_AVAILABLE = False

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    End-to-end RCA pipeline orchestrator.

    Given a time window and optional service filters, this class:
    1. Reads raw log files from configured sources
    2. Parses them via Drain3 template extraction
    3. Detects anomalies using TF-IDF cosine distance
    4. Builds a causal graph from anomaly signal time series (Granger)
    5. Ranks root causes via Personalized PageRank
    6. Generates an NLG narrative report
    7. Produces a remediation plan
    """

    def __init__(self):
        self.config = load_config()

        # --- Core modules (always available) ---
        self.log_parser = LogTemplateExtractor()
        self.log_anomaly_detector = SimpleLogAnomalyDetector()
        self.metric_anomaly_detector = MetricAnomalyDetector()
        self.causal_engine = CausalInferenceEngine(
            max_lag=self.config.get("causal_inference", {}).get("lags", 6),
            alpha=self.config.get("causal_inference", {}).get("fdr_alpha", 0.05),
        )
        self.nlg_generator = NLGGenerator()
        self.remediation_engine = RemediationEngine()
        self.unified_scorer = UnifiedAnomalyScorer(
            alpha=self.config.get("anomaly_detection", {}).get("alpha", 0.80),
            anomaly_threshold=self.config.get("anomaly_detection", {}).get(
                "anomaly_threshold", 0.5
            ),
        )

        # --- Log File Watcher (real-time tailing) ---
        self._log_watcher = None
        if LOG_WATCHER_AVAILABLE:
            try:
                watcher_config = self.config.get("log_watching", {})
                if watcher_config.get("enabled", False):
                    poll_interval = watcher_config.get("poll_interval_seconds", 1.0)
                    max_buffer = watcher_config.get("max_buffer_size", 10000)
                    self._log_watcher = LogFileWatcher(
                        poll_interval_seconds=poll_interval, max_buffer_size=max_buffer
                    )
                    logger.info("LogFileWatcher initialized")
            except Exception as exc:
                logger.warning("LogFileWatcher init failed: %s", exc)

        # --- Optional modules (graceful degradation) ---

        # LogBERT — deep log anomaly detector
        self.logbert = None
        if TRANSFORMERS_AVAILABLE:
            try:
                self.logbert = LogBERTDetector()
                logger.info("LogBERT detector initialised")
            except Exception as exc:
                logger.warning(
                    "LogBERT init failed, falling back to TF-IDF only: %s", exc
                )

        # Metric preprocessor
        self.metric_preprocessor = None
        if PREPROCESSOR_AVAILABLE:
            try:
                self.metric_preprocessor = MetricPreprocessor(self.config)
                logger.info("MetricPreprocessor initialised")
            except Exception as exc:
                logger.warning("MetricPreprocessor init failed: %s", exc)

        # Metric ingestion service
        self.metric_ingest = None
        if METRIC_INGEST_AVAILABLE:
            try:
                self.metric_ingest = MetricIngestionService(self.config)
                logger.info("MetricIngestionService initialised")
            except Exception as exc:
                logger.warning("MetricIngestionService init failed: %s", exc)

        # LSTM AE and Temporal Transformer — lazy-init (need num_features)
        self.lstm_ae = None
        self.temporal_transformer = None

        # KHBN — Bayesian posterior refinement
        self.khbn = None
        if KHBN_AVAILABLE:
            try:
                self.khbn = KHBNModel()
                logger.info("KHBNModel initialised")
            except Exception as exc:
                logger.warning("KHBN init failed: %s", exc)

        # Regex patterns for parsing log lines (match ingestion module formats)
        self._plaintext_re = re.compile(
            r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) (\w+) (.+)$"
        )
        self._syslog_re = re.compile(
            r"^(\w{3}\s+\d+\s+\d{2}:\d{2}:\d{2}) (\S+) (\S+): (.+)$"
        )

        self._checkpoint_loaded = False
        self._load_model_checkpoints()

    def _is_checkpoint_recent(self, checkpoint_path: str) -> bool:
        """Check if checkpoint is within max age days."""
        cache_enabled = self.config.get("models", {}).get("cache_checkpoints", True)
        if not cache_enabled:
            return False

        max_age_days = self.config.get("models", {}).get("checkpoint_max_age_days", 7)
        checkpoint_dir = Path(checkpoint_path)

        if not checkpoint_dir.exists():
            return False

        meta_file = checkpoint_dir / "logbert_meta.npy"
        if meta_file.exists():
            try:
                meta = np.load(str(meta_file), allow_pickle=True).item()
                saved_at = meta.get("saved_at")
                if saved_at:
                    saved_time = datetime.fromisoformat(saved_at)
                    age = datetime.now() - saved_time
                    return age < timedelta(days=max_age_days)
            except Exception:
                pass
        return False

    def _load_model_checkpoints(self) -> None:
        """Load models from checkpoints if available and recent."""
        if self._checkpoint_loaded:
            return

        cache_enabled = self.config.get("models", {}).get("cache_checkpoints", True)
        if not cache_enabled:
            logger.info("Checkpoint loading disabled in config")
            return

        checkpoint_base = self.config.get("models", {}).get(
            "checkpoint_dir", "./data/models"
        )

        logbert_path = os.path.join(checkpoint_base, "logbert")
        if os.path.exists(logbert_path) and self.logbert is not None:
            if self._is_checkpoint_recent(logbert_path):
                try:
                    self.logbert.load(logbert_path)
                    logger.info("Loaded LogBERT from checkpoint: %s", logbert_path)
                except Exception as exc:
                    logger.warning("Failed to load LogBERT checkpoint: %s", exc)
            else:
                logger.info("LogBERT checkpoint too old, will retrain")

        if LSTM_AE_AVAILABLE and self.lstm_ae is not None:
            lstm_path = os.path.join(checkpoint_base, "lstm_ae")
            if os.path.exists(lstm_path):
                try:
                    self.lstm_ae.load(lstm_path)
                    logger.info("Loaded LSTM AE from checkpoint: %s", lstm_path)
                except Exception as exc:
                    logger.warning("Failed to load LSTM AE checkpoint: %s", exc)

        self._checkpoint_loaded = True

    # ------------------------------------------------------------------
    # 1. LOG READING
    # ------------------------------------------------------------------

    def _read_log_file(self, path: str, fmt: str) -> List[Dict[str, Any]]:
        """Read and parse a single log file into structured records."""
        records = []
        abs_path = os.path.abspath(path)

        if not os.path.exists(abs_path):
            logger.warning("Log file not found: %s", abs_path)
            return records

        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                record = self._parse_line(line, fmt, abs_path)
                if record:
                    records.append(record)

        return records

    def _parse_line(
        self, line: str, fmt: str, source_path: str
    ) -> Optional[Dict[str, Any]]:
        """Parse a single log line — mirrors LogHandler._parse_line."""
        record: Dict[str, Any] = {
            "timestamp": None,
            "level": "INFO",
            "source_file": source_path,
            "raw_message": line,
        }

        try:
            if fmt == "json":
                import json

                data = json.loads(line)
                record["timestamp"] = data.get("ts") or data.get("timestamp")
                record["level"] = (data.get("level", "INFO")).upper()
                record["raw_message"] = data.get("msg") or data.get("message", line)
            elif fmt == "plaintext":
                match = self._plaintext_re.match(line)
                if match:
                    record["timestamp"] = match.group(1)
                    record["level"] = match.group(2).upper()
                    record["raw_message"] = match.group(3)
            elif fmt == "syslog":
                match = self._syslog_re.match(line)
                if match:
                    record["timestamp"] = match.group(1)
                    record["raw_message"] = match.group(4)
        except Exception as exc:
            logger.debug("Parse error on line: %s — %s", line[:80], exc)

        return record

    def _read_all_logs(self) -> Dict[str, List[Dict[str, Any]]]:
        """Read all configured log sources and return {label: [records]}."""
        sources = self.config.get("log_sources", [])
        all_logs: Dict[str, List[Dict[str, Any]]] = {}

        for source in sources:
            label = source["label"]
            records = self._read_log_file(source["path"], source["format"])
            all_logs[label] = records
            logger.info("Read %d records from %s", len(records), label)

        return all_logs

    def start_log_watcher(self) -> bool:
        """
        Start the real-time log file watcher.

        Returns:
            True if watcher started successfully, False otherwise.
        """
        if self._log_watcher is None:
            logger.warning("LogFileWatcher not available")
            return False

        try:
            sources = self.config.get("log_sources", [])
            self._log_watcher.start(sources)
            logger.info("LogFileWatcher started")
            return True
        except Exception as exc:
            logger.warning("Failed to start LogFileWatcher: %s", exc)
            return False

    def stop_log_watcher(self) -> None:
        """Stop the real-time log file watcher."""
        if self._log_watcher is not None:
            self._log_watcher.stop()
            logger.info("LogFileWatcher stopped")

    def get_new_logs(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get new log lines since last read from all watched files.

        Returns:
            Dict mapping label -> list of parsed log records.
        """
        if self._log_watcher is None:
            return {}

        return self._log_watcher.get_new_lines()

    def read_full_file(self, path: str, fmt: str) -> List[Dict[str, Any]]:
        """
        Read entire log file at once (backward compatible).

        Args:
            path: Path to log file
            fmt: Log format (json, plaintext, syslog)

        Returns:
            List of parsed log records
        """
        return self._read_log_file(path, fmt)

    # ------------------------------------------------------------------
    # 1B. METRIC READING
    # ------------------------------------------------------------------

    def _read_metrics(self) -> Optional[pd.DataFrame]:
        """Read metric data from CSV files or MetricIngestionService.

        Returns a DataFrame with DatetimeIndex and metric columns, or None
        if no metric data is available.
        """
        # Try combined CSV first
        combined_path = os.path.join("data", "metrics", "combined_metrics.csv")
        if os.path.exists(combined_path):
            try:
                df = pd.read_csv(combined_path, parse_dates=["timestamp"])
                # Handle long-format CSV (timestamp, metric_name, value, ...)
                if "metric_name" in df.columns and "value" in df.columns:
                    pivoted = df.pivot_table(
                        index="timestamp",
                        columns="metric_name",
                        values="value",
                        aggfunc="mean",
                    )
                    pivoted.index = pd.DatetimeIndex(pivoted.index)
                    if not pivoted.empty:
                        logger.info(
                            "Loaded combined metrics (pivoted): %d rows, %d columns",
                            len(pivoted),
                            len(pivoted.columns),
                        )
                        return pivoted
                else:
                    # Wide-format: timestamp + numeric columns
                    df = df.set_index("timestamp")
                    df = df.select_dtypes(include=[np.number])
                    if not df.empty:
                        logger.info(
                            "Loaded combined metrics: %d rows, %d columns",
                            len(df),
                            len(df.columns),
                        )
                        return df
            except Exception as exc:
                logger.warning("Failed to read combined metrics CSV: %s", exc)

        # Try individual CSV files in data/metrics/
        metrics_dir = os.path.join("data", "metrics")
        if os.path.isdir(metrics_dir):
            csv_files = glob_mod.glob(os.path.join(metrics_dir, "*.csv"))
            frames = []
            for csv_path in csv_files:
                try:
                    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
                    if "metric_name" in df.columns and "value" in df.columns:
                        pivoted = df.pivot_table(
                            index="timestamp",
                            columns="metric_name",
                            values="value",
                            aggfunc="mean",
                        )
                        pivoted.index = pd.DatetimeIndex(pivoted.index)
                        if not pivoted.empty:
                            frames.append(pivoted)
                    else:
                        df = df.set_index("timestamp")
                        df = df.select_dtypes(include=[np.number])
                        if not df.empty:
                            frames.append(df)
                except Exception as exc:
                    logger.debug("Skipping metric file %s: %s", csv_path, exc)

            if frames:
                merged = pd.concat(frames, axis=1)
                # Deduplicate column names
                merged = merged.loc[:, ~merged.columns.duplicated()]
                logger.info(
                    "Loaded metrics from %d CSV files: %d rows, %d columns",
                    len(frames),
                    len(merged),
                    len(merged.columns),
                )
                return merged

        # Try MetricIngestionService (Prometheus / CloudWatch)
        if self.metric_ingest is not None:
            try:
                points = self.metric_ingest.run_scrape_cycle()
                if points:
                    rows = [p.to_dict() for p in points]
                    df = pd.DataFrame(rows)
                    # Pivot: timestamp × metric_name → value
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    pivoted = df.pivot_table(
                        index="timestamp",
                        columns="metric_name",
                        values="value",
                        aggfunc="mean",
                    )
                    if not pivoted.empty:
                        logger.info(
                            "Ingested metrics from live scrape: %d rows, %d columns",
                            len(pivoted),
                            len(pivoted.columns),
                        )
                        return pivoted
            except Exception as exc:
                logger.warning("Metric ingestion scrape failed: %s", exc)

        logger.info("No metric data available — running log-only analysis")
        return None

    # ------------------------------------------------------------------
    # 2. TEMPLATE EXTRACTION
    # ------------------------------------------------------------------

    def _extract_templates(
        self, all_logs: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Run Drain3 template extraction on all log records."""
        for label, records in all_logs.items():
            for rec in records:
                tmpl = self.log_parser.extract_template(rec["raw_message"])
                rec["template"] = tmpl["template"]
                rec["cluster_id"] = tmpl["cluster_id"]
                rec["is_new_template"] = tmpl["is_new"]
        return all_logs

    # ------------------------------------------------------------------
    # 2B. METRIC PREPROCESSING
    # ------------------------------------------------------------------

    def _preprocess_metrics(self, metric_df: pd.DataFrame) -> pd.DataFrame:
        """Fill gaps and normalize metric data.

        Falls back to returning the raw DataFrame if MetricPreprocessor is
        unavailable or fails.
        """
        if self.metric_preprocessor is None:
            logger.debug("MetricPreprocessor not available, returning raw metrics")
            return metric_df

        try:
            filled = self.metric_preprocessor.fill_gaps(metric_df)
            normalized = self.metric_preprocessor.normalize(filled)
            logger.info("Metric preprocessing complete: %d rows", len(normalized))
            return normalized
        except Exception as exc:
            logger.warning("Metric preprocessing failed, using raw data: %s", exc)
            return metric_df

    # ------------------------------------------------------------------
    # 3. ANOMALY DETECTION
    # ------------------------------------------------------------------

    def _detect_anomalies(
        self, all_logs: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, float]:
        """
        Train on the first half of messages (baseline), score the second half.
        Returns {source_label: anomaly_score} as a summary per source.
        """
        anomaly_scores: Dict[str, float] = {}

        for label, records in all_logs.items():
            messages = [r["raw_message"] for r in records]
            if len(messages) < 4:
                anomaly_scores[label] = 0.0
                continue

            # Split: first 60% = healthy baseline, last 40% = window to score
            split = int(len(messages) * 0.6)
            training_msgs = messages[:split]
            test_msgs = messages[split:]

            # Train detector on healthy baseline
            detector = SimpleLogAnomalyDetector(
                threshold_percentile=self.config.get("anomaly_detection", {}).get(
                    "threshold_percentile", 99
                )
            )
            detector.train(training_msgs)

            if not detector.is_trained:
                anomaly_scores[label] = 0.0
                continue

            scores = detector.score(test_msgs)
            # Aggregate: max anomaly score in the window
            max_score = float(np.max(scores)) if len(scores) > 0 else 0.0
            anomaly_scores[label] = round(min(max_score, 1.0), 4)

            # Annotate individual records with scores
            for i, rec in enumerate(records[split:]):
                rec["anomaly_score"] = float(scores[i]) if i < len(scores) else 0.0

        return anomaly_scores

    # ------------------------------------------------------------------
    # 3B. ENHANCED LOG ANOMALY DETECTION (TF-IDF + LogBERT ensemble)
    # ------------------------------------------------------------------

    def _detect_log_anomalies_enhanced(
        self, all_logs: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, float]:
        """Detect log anomalies using TF-IDF + optional LogBERT ensemble.

        Falls back to TF-IDF only if LogBERT is unavailable or fails.
        Returns {source_label: anomaly_score}.
        """
        # Always get TF-IDF scores via existing method
        tfidf_scores = self._detect_anomalies(all_logs)

        # If LogBERT is not available, use unified scorer with TF-IDF only
        if self.logbert is None:
            # Still run through unified scorer for consistency
            for label, records in all_logs.items():
                tfidf_val = tfidf_scores.get(label, 0.0)
                log_levels = [r.get("level", "INFO") for r in records]
                result = self.unified_scorer.score_logs(
                    tfidf_scores=np.array([tfidf_val]),
                    logbert_scores=None,
                    log_levels=log_levels[:1] if log_levels else None,
                )
                if result["unified_scores"].size > 0:
                    tfidf_scores[label] = round(float(result["unified_scores"][0]), 4)
            return tfidf_scores

        # LogBERT available — run ensemble
        # Cap messages to avoid excessive CPU time with transformer models
        LOGBERT_MAX_MSGS = 2000
        enhanced_scores: Dict[str, float] = {}
        for label, records in all_logs.items():
            messages = [r["raw_message"] for r in records]
            sampled_records = list(records)  # shallow copy
            if len(messages) < 4:
                enhanced_scores[label] = tfidf_scores.get(label, 0.0)
                continue

            # Sample if too many messages for LogBERT (CPU-bound transformer)
            if len(messages) > LOGBERT_MAX_MSGS:
                logger.info(
                    "Sampling %d/%d messages for LogBERT (%s)",
                    LOGBERT_MAX_MSGS,
                    len(messages),
                    label,
                )
                step = len(messages) / LOGBERT_MAX_MSGS
                indices = [int(i * step) for i in range(LOGBERT_MAX_MSGS)]
                messages = [messages[idx] for idx in indices]
                sampled_records = [records[idx] for idx in indices]

            split = int(len(messages) * 0.6)
            training_msgs = messages[:split]
            test_msgs = messages[split:]

            # Get per-message LogBERT scores
            logbert_msg_scores = None
            try:
                if not self.logbert.is_trained:
                    self.logbert.train(training_msgs, epochs=1, batch_size=16)
                logbert_msg_scores = self.logbert.score(test_msgs)
            except Exception as exc:
                logger.warning("LogBERT scoring failed for %s: %s", label, exc)

            # Build TF-IDF per-message scores for this source
            tfidf_msg_scores = None
            try:
                detector = SimpleLogAnomalyDetector(
                    threshold_percentile=self.config.get("anomaly_detection", {}).get(
                        "threshold_percentile", 99
                    )
                )
                detector.train(training_msgs)
                if detector.is_trained:
                    tfidf_msg_scores = detector.score(test_msgs)
            except Exception:
                pass

            # Log levels for rarity priors (from sampled records, not full)
            test_records = sampled_records[split:]
            log_levels = [r.get("level", "INFO") for r in test_records]

            # Use unified scorer to combine
            result = self.unified_scorer.score_logs(
                tfidf_scores=tfidf_msg_scores,
                logbert_scores=logbert_msg_scores,
                log_levels=log_levels,
            )

            unified = result["unified_scores"]
            if unified.size > 0:
                max_score = float(np.max(unified))
                enhanced_scores[label] = round(min(max_score, 1.0), 4)

                # Annotate individual records with unified scores
                for i, rec in enumerate(test_records):
                    if i < len(unified):
                        rec["anomaly_score"] = float(unified[i])
            else:
                enhanced_scores[label] = tfidf_scores.get(label, 0.0)

        return enhanced_scores

    # ------------------------------------------------------------------
    # 3C. METRIC ANOMALY DETECTION (LSTM AE + Temporal Transformer)
    # ------------------------------------------------------------------

    def _detect_metric_anomalies(self, metric_df: pd.DataFrame) -> Dict[str, float]:
        """Detect anomalies in metric data using deep learning models.

        Returns {metric_name: anomaly_score} via LSTM AE + Temporal
        Transformer ensemble, unified through UnifiedAnomalyScorer.
        Falls back to simple statistical scoring if DL models are unavailable
        or data is too large for inline training.
        """
        if metric_df is None or metric_df.empty:
            return {}

        # Drop NaN-only columns
        metric_df = metric_df.dropna(axis=1, how="all")
        metric_names = list(metric_df.columns)
        num_features = len(metric_names)

        if num_features == 0:
            return {}

        lstm_ae_scores = None
        transformer_scores = None

        # Only attempt DL training on reasonably sized datasets
        # (avoid timeout on large datasets during pipeline runs)
        max_rows_for_dl_training = 2000

        # LSTM Autoencoder scoring
        if LSTM_AE_AVAILABLE and len(metric_df) <= max_rows_for_dl_training:
            try:
                seq_len = min(60, max(len(metric_df) // 3, 2))
                if self.lstm_ae is None:
                    self.lstm_ae = MetricDeepAnomalyDetector(
                        num_features=num_features,
                        sequence_length=seq_len,
                    )
                # Need enough data for sequences
                if len(metric_df) >= seq_len * 2:
                    split = int(len(metric_df) * 0.6)
                    healthy_df = metric_df.iloc[:split].fillna(0.0)
                    if self.lstm_ae.thresholds is None:
                        self.lstm_ae.train(healthy_df, epochs=3, batch_size=32)
                    lstm_ae_scores = self.lstm_ae.score(metric_df.fillna(0.0))
                    logger.info("LSTM AE scores: %s", lstm_ae_scores)
            except Exception as exc:
                logger.warning("LSTM AE scoring failed: %s", exc)

        # Temporal Transformer scoring
        if TRANSFORMER_AVAILABLE and len(metric_df) <= max_rows_for_dl_training:
            try:
                forecast_horizon = 10
                seq_len = min(60, max(len(metric_df) // 4, 2))
                if self.temporal_transformer is None:
                    self.temporal_transformer = TemporalTransformerDetector(
                        n_features=num_features,
                        sequence_length=seq_len,
                        forecast_horizon=forecast_horizon,
                    )
                required = seq_len + forecast_horizon
                if len(metric_df) >= required * 2:
                    split = int(len(metric_df) * 0.6)
                    healthy_df = metric_df.iloc[:split].fillna(0.0)
                    if self.temporal_transformer._threshold is None:
                        self.temporal_transformer.train(
                            healthy_df, epochs=5, batch_size=32
                        )
                    transformer_scores = self.temporal_transformer.score(
                        metric_df.fillna(0.0)
                    )
                    logger.info("Transformer scores: %s", transformer_scores)
            except Exception as exc:
                logger.warning("Temporal Transformer scoring failed: %s", exc)

        # If no DL scores, fall back to simple statistical anomaly detection
        if lstm_ae_scores is None and transformer_scores is None:
            logger.info(
                "DL models not used (data too large or unavailable), "
                "falling back to statistical metric scoring"
            )
            metric_anomaly_scores: Dict[str, float] = {}
            for name in metric_names:
                col = metric_df[name].dropna()
                if len(col) < 2:
                    metric_anomaly_scores[name] = 0.0
                    continue
                # Simple z-score based anomaly: tail 40% vs head 60%
                split = int(len(col) * 0.6)
                baseline_mean = col.iloc[:split].mean()
                baseline_std = col.iloc[:split].std()
                if baseline_std == 0 or np.isnan(baseline_std):
                    metric_anomaly_scores[name] = 0.0
                    continue
                tail_mean = col.iloc[split:].mean()
                z = abs(tail_mean - baseline_mean) / baseline_std
                # Sigmoid mapping to [0, 1]
                score = 1.0 / (1.0 + np.exp(-2.0 * (z - 2.0)))
                metric_anomaly_scores[name] = round(float(score), 4)
            return metric_anomaly_scores

        # Combine via unified scorer
        result = self.unified_scorer.score_metrics(
            lstm_ae_scores=lstm_ae_scores,
            transformer_scores=transformer_scores,
        )

        unified = result["unified_scores"]
        metric_anomaly_scores = {}
        for i, name in enumerate(metric_names):
            if i < len(unified):
                metric_anomaly_scores[name] = round(float(unified[i]), 4)
            else:
                metric_anomaly_scores[name] = 0.0

        return metric_anomaly_scores

    # ------------------------------------------------------------------
    # 4. BUILD TIME SERIES FOR CAUSAL INFERENCE
    # ------------------------------------------------------------------

    def _build_signal_dataframe(
        self,
        all_logs: Dict[str, List[Dict[str, Any]]],
        anomaly_scores: Dict[str, float],
        metric_scores: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Convert per-source log anomaly patterns into a time-series DataFrame
        suitable for Granger causality testing.

        Strategy: bucket log records into 1-minute windows, compute the fraction
        of anomalous messages (above threshold) in each window per source.

        If metric_scores is provided, adds metric anomaly signals alongside
        log signals in the DataFrame.
        """
        # Collect timestamps for each source — assign minute-level buckets
        source_series: Dict[str, List[float]] = {}

        for label, records in all_logs.items():
            # Use record index as proxy for time ordering
            n = len(records)
            if n == 0:
                continue

            # Create per-record anomaly signal (use level severity as proxy)
            severity_map = {"INFO": 0.0, "WARN": 0.3, "ERROR": 0.7, "CRITICAL": 1.0}
            signal = []
            for rec in records:
                level_score = severity_map.get(rec.get("level", "INFO"), 0.0)
                rec_anomaly = rec.get("anomaly_score", 0.0)
                # Combined signal: blend level severity with anomaly score
                combined = max(level_score, rec_anomaly)
                signal.append(combined)

            source_series[label] = signal

        if not source_series:
            return pd.DataFrame()

        # Pad all series to the same length (longest series)
        max_len = max(len(s) for s in source_series.values())

        # Need at least max_lag * 2 + 1 data points for Granger test
        min_len = self.causal_engine.max_lag * 2 + 2
        max_len = max(max_len, min_len)

        padded: Dict[str, List[float]] = {}
        for label, signal in source_series.items():
            if len(signal) < max_len:
                # Pad with zeros at the beginning (no anomaly before logs start)
                pad = [0.0] * (max_len - len(signal))
                padded[label] = pad + signal
            else:
                padded[label] = signal

        # Add metric anomaly signals if available
        if metric_scores:
            for metric_name, score in metric_scores.items():
                if metric_name not in padded and score > 0.0:
                    # Create a synthetic time series: ramp up to the score
                    ramp_len = max(max_len // 4, 1)
                    signal = [0.0] * (max_len - ramp_len) + [
                        score * (i / ramp_len) for i in range(1, ramp_len + 1)
                    ]
                    padded[metric_name] = signal

        return pd.DataFrame(padded)

    # ------------------------------------------------------------------
    # 5. ROOT CAUSE MAPPING
    # ------------------------------------------------------------------

    _CAUSE_TO_REMEDIATION_KEY = {
        # Map common causal node labels / generator labels to remediation rule keys
        # -- Case Study 1: DB Migration
        "database": "db_migration_applied",
        "db": "db_migration_applied",
        "db_migration": "db_migration_applied",
        "query_latency": "db_migration_applied",
        "connection_pool": "db_migration_applied",
        # -- Case Study 2: Memory Leak
        "memory_leak": "memory_leak_detected",
        "memory_leak_code_deploy": "memory_leak_detected",
        "oom": "memory_leak_detected",
        "memory_growth": "memory_leak_detected",
        # -- Case Study 3: Network Partition
        "network_partition": "network_partition_detected",
        "network_partition_bgp": "network_partition_detected",
        "split_brain": "network_partition_detected",
        "bgp": "network_partition_detected",
        # -- Case Study 4: Thread Pool Exhaustion
        "thread_pool": "thread_pool_exhaustion",
        "thread_pool_background_job": "thread_pool_exhaustion",
        "thread_exhaustion": "thread_pool_exhaustion",
        # -- Case Study 5: DNS Propagation
        "dns": "dns_ttl_misconfiguration",
        "dns_ttl": "dns_ttl_misconfiguration",
        "dns_propagation": "dns_ttl_misconfiguration",
        # -- Case Study 6: CPU Saturation
        "cpu": "cpu_runaway_process",
        "cpu_saturation": "cpu_runaway_process",
        "cpu_runaway": "cpu_runaway_process",
        "runaway_process": "cpu_runaway_process",
        # -- Case Study 7: Connection Pool Leak
        "connection_leak": "connection_leak_bug",
        "connection_pool_leak": "connection_leak_bug",
        "pool_exhaustion": "connection_leak_bug",
        # -- Case Study 8: Cache Stampede
        "cache_stampede": "cache_ttl_expiry",
        "cache_ttl": "cache_ttl_expiry",
        "thundering_herd": "cache_ttl_expiry",
        # -- Case Study 9: Disk Exhaustion / Log Rotation
        "disk_exhaustion": "log_rotation_disabled",
        "disk_full": "log_rotation_disabled",
        "log_rotation": "log_rotation_disabled",
        # -- Case Study 10: MQ Backlog / Consumer Crash
        "mq_backlog": "consumer_service_crash",
        "consumer_crash": "consumer_service_crash",
        "queue_backlog": "consumer_service_crash",
        # -- Generic fallbacks
        "application": "memory_leak_detected",
        "app": "memory_leak_detected",
        "system": "memory_leak_detected",
    }

    def _map_to_remediation_key(self, cause_label: str) -> str:
        """Map a causal graph node label to a remediation rules key."""
        # Direct match first
        if cause_label in self.remediation_engine.rules:
            return cause_label

        # Fuzzy match via known mappings
        lower = cause_label.lower()
        for prefix, key in self._CAUSE_TO_REMEDIATION_KEY.items():
            if prefix in lower:
                return key

        return cause_label  # fallback — remediation engine handles unknown keys

    # ------------------------------------------------------------------
    # 6. FULL PIPELINE RUN
    # ------------------------------------------------------------------

    def run(
        self,
        incident_id: str,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        services: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute the full RCA pipeline and return a structured incident report.

        Parameters
        ----------
        incident_id : str
            Unique incident identifier (e.g. INC-1710500000)
        start_time : str, optional
            ISO-format start of the analysis window
        end_time : str, optional
            ISO-format end of the analysis window
        services : list[str], optional
            Filter to specific service labels

        Returns
        -------
        dict
            Complete incident report with anomalies, causal graph, rankings,
            narrative, and remediation plan.
        """
        detected_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        logger.info("Pipeline run started for %s", incident_id)

        # --- Stage 1A: Read logs ---
        try:
            all_logs = self._read_all_logs()
        except Exception as exc:
            raise PipelineError(str(exc), stage="log_ingestion") from exc

        # Filter by requested services if provided
        if services:
            all_logs = {k: v for k, v in all_logs.items() if k in services}

        if not all_logs or all(len(v) == 0 for v in all_logs.values()):
            return self._empty_report(incident_id, detected_at, "No log data found.")

        # --- Stage 1B: Read metrics (optional — degrades gracefully) ---
        metric_df = None
        try:
            metric_df = self._read_metrics()
        except Exception as exc:
            logger.warning("Metric reading failed, continuing log-only: %s", exc)

        # --- Stage 2A: Template extraction ---
        try:
            all_logs = self._extract_templates(all_logs)
        except Exception as exc:
            raise PipelineError(str(exc), stage="template_extraction") from exc

        # --- Stage 2B: Preprocess metrics (optional) ---
        preprocessed_metrics = None
        if metric_df is not None:
            try:
                preprocessed_metrics = self._preprocess_metrics(metric_df)
            except Exception as exc:
                logger.warning("Metric preprocessing failed: %s", exc)
                preprocessed_metrics = metric_df

        # --- Stage 3A: Log anomaly detection (TF-IDF + LogBERT ensemble) ---
        try:
            anomaly_scores = self._detect_log_anomalies_enhanced(all_logs)
        except Exception as exc:
            raise PipelineError(str(exc), stage="log_anomaly_detection") from exc
        logger.info("Log anomaly scores: %s", anomaly_scores)

        # --- Stage 3B: Metric anomaly detection (LSTM AE + Transformer) ---
        metric_anomaly_scores: Dict[str, float] = {}
        if preprocessed_metrics is not None:
            try:
                metric_anomaly_scores = self._detect_metric_anomalies(
                    preprocessed_metrics
                )
                logger.info("Metric anomaly scores: %s", metric_anomaly_scores)
            except Exception as exc:
                logger.warning("Metric anomaly detection failed: %s", exc)

        # --- Merge all anomaly scores ---
        all_anomaly_scores = dict(anomaly_scores)
        all_anomaly_scores.update(metric_anomaly_scores)

        # Check if any anomalies were actually detected
        if all(v < 0.01 for v in all_anomaly_scores.values()):
            return self._empty_report(
                incident_id,
                detected_at,
                "No anomalies detected in the analysis window.",
            )

        # --- Stage 4: Build signal DataFrame for causal inference ---
        signals_df = self._build_signal_dataframe(
            all_logs, anomaly_scores, metric_scores=metric_anomaly_scores
        )

        if signals_df.empty or signals_df.shape[1] < 2:
            # Need at least 2 signals for causal analysis
            # Fall back to reporting anomalies without causal graph
            return self._anomaly_only_report(
                incident_id, detected_at, all_anomaly_scores, all_logs
            )

        # --- Stage 5: Causal graph construction (Granger + FDR + PC) ---
        try:
            graph = self.causal_engine.build_causal_graph(
                signals_df, use_pc=True, use_fdr=True
            )
        except Exception as exc:
            raise PipelineError(str(exc), stage="causal_graph_construction") from exc
        logger.info(
            "Causal graph: %d nodes, %d edges", len(graph.nodes), len(graph.edges)
        )

        # --- Stage 6: KHBN posterior refinement (optional) ---
        if self.khbn is not None:
            try:
                causal_edges = [
                    (u, v, data.get("weight", 1.0))
                    for u, v, data in graph.edges(data=True)
                ]
                self.khbn.fit(all_anomaly_scores, causal_edges)
                posterior_ranks = self.khbn.posterior_rank()
                if posterior_ranks:
                    logger.info(
                        "KHBN posterior ranks: top=%s (%.4f)",
                        posterior_ranks[0][0],
                        posterior_ranks[0][1],
                    )
            except Exception as exc:
                logger.warning("KHBN refinement failed: %s", exc)

        # --- Stage 7: 5-factor RCA scoring ---
        try:
            temporal_order = sorted(
                all_anomaly_scores.keys(),
                key=lambda k: all_anomaly_scores.get(k, 0.0),
                reverse=True,
            )

            rarity_priors: Dict[str, float] = {}
            for label, records in all_logs.items():
                levels = [r.get("level", "INFO") for r in records]
                if levels:
                    level_scores = [
                        self.unified_scorer.LOG_LEVEL_WEIGHTS.get(lv.upper(), 0.2)
                        for lv in levels
                    ]
                    rarity_priors[label] = float(np.mean(level_scores))
                else:
                    rarity_priors[label] = 0.2
            for metric_name in metric_anomaly_scores:
                if metric_name not in rarity_priors:
                    rarity_priors[metric_name] = 0.3

            event_metadata: Dict[str, Dict[str, Any]] = {}
            for label, records in all_logs.items():
                levels = [r.get("level", "INFO") for r in records]
                worst_level = "INFO"
                for lv in ["CRITICAL", "FATAL", "ERROR", "WARN", "WARNING"]:
                    if lv in [l.upper() for l in levels]:
                        worst_level = lv
                        break
                event_metadata[label] = {
                    "log_level": worst_level,
                    "metric_score": metric_anomaly_scores.get(label, 0.0),
                }

            ranks = self.causal_engine.rank_root_causes(
                graph,
                anomaly_scores=all_anomaly_scores,
                temporal_order=temporal_order,
                rarity_priors=rarity_priors,
                event_metadata=event_metadata,
            )
        except Exception as exc:
            raise PipelineError(str(exc), stage="rca_scoring") from exc

        # --- Stage 8: Determine top root cause ---
        top_cause = (
            ranks[0]["signal"]
            if ranks
            else max(all_anomaly_scores, key=all_anomaly_scores.get)
        )
        top_confidence = (
            float(ranks[0]["rca_score"])
            if ranks
            else float(max(all_anomaly_scores.values()))
        )
        remediation_key = self._map_to_remediation_key(top_cause)

        # --- Stage 9: Remediation plan (with confidence gate) ---
        remediation_plan = self.remediation_engine.get_remediation_plan(
            remediation_key,
            confidence=top_confidence,
            context={"service": top_cause, "incident_id": incident_id},
        )

        # --- Stage 10: Build NLG report data ---
        causal_chain = []
        for u, v, data in graph.edges(data=True):
            causal_chain.append(
                {
                    "from": u,
                    "to": v,
                    "confidence": round(float(data.get("weight", 0.0)), 4),
                    "lag": round(float(data.get("p_value", 0.0)), 4),
                }
            )

        evidence = []
        for label, score in all_anomaly_scores.items():
            # Find the most anomalous log message as evidence
            records = all_logs.get(label, [])
            worst_msg = "Anomalous signal detected"
            worst_ts = None
            worst_score = -1.0
            for rec in records:
                rec_score = float(rec.get("anomaly_score", 0))
                if rec_score >= worst_score:
                    worst_score = rec_score
                    worst_msg = rec.get("raw_message", worst_msg)[:200]
                    worst_ts = rec.get("timestamp")
            # For metric-only signals, generate descriptive evidence
            if not records and label in metric_anomaly_scores:
                worst_msg = f"Metric '{label}' anomaly score: {score:.4f}"
            evidence.append(
                {
                    "source": label,
                    "detail": worst_msg,
                    "score": round(score, 4),
                    "timestamp": worst_ts,
                }
            )

        report_data = {
            "incident_id": incident_id,
            "detected_at": detected_at,
            "summary": self._generate_summary(
                top_cause, causal_chain, all_anomaly_scores
            ),
            "causal_chain": causal_chain,
            "evidence": evidence,
            "root_cause": top_cause,
            "confidence": top_confidence,
            "anomaly_scores": {
                k: round(float(v), 4) for k, v in all_anomaly_scores.items()
            },
            "recommendations": self._flatten_recommendations(remediation_plan),
            "prevention": self._flatten_prevention(remediation_plan),
        }

        # --- Stage 11: Generate NLG narrative ---
        try:
            report_data["narrative"] = self.nlg_generator.generate_narrative(
                report_data
            )
        except Exception as exc:
            logger.warning("NLG narrative generation failed: %s", exc)
            report_data["narrative"] = (
                f"# RCA Report\n\n**Incident:** {incident_id}\n\nAn error occurred during narrative generation."
            )

        # --- Build annotated log lines for /logs endpoint ---
        annotated_logs = []
        for label, records in all_logs.items():
            for rec in records:
                annotated_logs.append(
                    {
                        "source": label,
                        "timestamp": rec.get("timestamp"),
                        "level": rec.get("level", "INFO"),
                        "message": rec.get("raw_message", ""),
                        "template": rec.get("template"),
                        "cluster_id": rec.get("cluster_id"),
                        "anomaly_score": round(float(rec.get("anomaly_score", 0.0)), 4),
                    }
                )
        # Sort by anomaly score descending
        annotated_logs.sort(key=lambda x: x["anomaly_score"], reverse=True)

        # --- Assemble final report ---
        final_report = {
            "incident_id": incident_id,
            "status": "complete",
            "detected_at": detected_at,
            "analysis_window": {
                "start_time": start_time,
                "end_time": end_time,
            },
            "log_sources_analyzed": list(all_logs.keys()),
            "total_records_analyzed": sum(len(v) for v in all_logs.values()),
            "anomalous_metrics": [
                {
                    "name": label,
                    "score": round(float(score), 4),
                    "first_seen": start_time or detected_at,
                }
                for label, score in sorted(
                    all_anomaly_scores.items(), key=lambda x: x[1], reverse=True
                )
            ],
            "causal_graph": {
                "nodes": [
                    {
                        "id": n,
                        "anomaly_score": round(
                            float(all_anomaly_scores.get(n, 0.0)), 4
                        ),
                    }
                    for n in graph.nodes
                ],
                "edges": causal_chain,
            },
            "ranked_causes": [
                {
                    "rank": i + 1,
                    "cause": r["signal"],
                    "confidence": round(float(r["rca_score"]), 4),
                    "pagerank_score": round(float(r.get("pagerank_score", 0)), 4),
                    "temporal_priority": round(float(r.get("temporal_priority", 0)), 4),
                    "anomaly_score": round(float(r.get("anomaly_score", 0)), 4),
                    "rarity_prior": round(float(r.get("rarity_prior", 0)), 4),
                    "event_bonus": round(float(r.get("event_bonus", 0)), 4),
                }
                for i, r in enumerate(ranks)
            ],
            "root_cause": top_cause,
            "narrative": report_data["narrative"],
            "remediation_plan": remediation_plan,
            "evidence": evidence,
            "annotated_logs": annotated_logs,
        }

        # Add metric anomaly scores if available
        if metric_anomaly_scores:
            final_report["metric_anomaly_scores"] = {
                k: round(float(v), 4) for k, v in metric_anomaly_scores.items()
            }

        logger.info(
            "Pipeline run complete for %s — root cause: %s", incident_id, top_cause
        )
        return final_report

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------

    def _flatten_recommendations(self, plan: Dict[str, Any]) -> List[Dict[str, str]]:
        """Flatten tiered remediation plan into NLG-friendly recommendation list."""
        recs = []
        for a in plan.get("tier1_auto_actions", []):
            recs.append(
                {
                    "tier": "Tier 1",
                    "description": a.get("description", ""),
                    "command": a.get("command", ""),
                }
            )
        for step in plan.get("tier2_walkthrough", {}).get("steps", []):
            recs.append(
                {
                    "tier": "Tier 2",
                    "description": step.get("title", ""),
                }
            )
        for adv in plan.get("tier3_advisory", []):
            recs.append(
                {
                    "tier": "Tier 3",
                    "description": adv.get("recommendation", ""),
                }
            )
        return recs

    def _flatten_prevention(self, plan: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract structured prevention checklist from remediation plan."""
        checklist = plan.get("prevention_checklist", {})
        return {
            "immediate": checklist.get("immediate", []),
            "short_term": checklist.get("short_term", []),
            "long_term": checklist.get("long_term", []),
        }

    def _generate_summary(
        self,
        top_cause: str,
        causal_chain: List[Dict],
        anomaly_scores: Dict[str, float],
    ) -> str:
        """Generate a concise one-line summary of the incident."""
        affected = [
            label
            for label, score in anomaly_scores.items()
            if score > 0.1 and label != top_cause
        ]
        chain_desc = ""
        if causal_chain:
            chain_desc = " → ".join(
                [causal_chain[0]["from"]] + [step["to"] for step in causal_chain]
            )
            return (
                f"Detected causal chain: {chain_desc}. "
                f"Root cause identified as '{top_cause}' "
                f"affecting {len(affected)} other source(s)."
            )
        return (
            f"Anomaly detected in '{top_cause}' "
            f"with {len(affected)} other affected source(s)."
        )

    def _empty_report(
        self, incident_id: str, detected_at: str, reason: str
    ) -> Dict[str, Any]:
        """Return a minimal report when no analysis could be performed."""
        return {
            "incident_id": incident_id,
            "status": "complete",
            "detected_at": detected_at,
            "anomalous_metrics": [],
            "causal_graph": {"nodes": [], "edges": []},
            "ranked_causes": [],
            "root_cause": None,
            "narrative": f"# RCA Report\n\n**Incident:** {incident_id}\n\n{reason}",
            "remediation_plan": {},
            "evidence": [],
            "log_sources_analyzed": [],
            "total_records_analyzed": 0,
        }

    def _anomaly_only_report(
        self,
        incident_id: str,
        detected_at: str,
        anomaly_scores: Dict[str, float],
        all_logs: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Report anomalies without causal analysis (too few signals)."""
        top_cause = max(anomaly_scores, key=anomaly_scores.get)
        top_confidence = float(max(anomaly_scores.values()))
        remediation_key = self._map_to_remediation_key(top_cause)
        remediation_plan = self.remediation_engine.get_remediation_plan(
            remediation_key,
            confidence=top_confidence,
            context={"service": top_cause, "incident_id": incident_id},
        )

        evidence = []
        for label, score in anomaly_scores.items():
            records = all_logs.get(label, [])
            worst_msg = "Anomalous signal detected"
            worst_ts = None
            for rec in records:
                if rec.get("anomaly_score", 0) > score:
                    worst_msg = rec["raw_message"][:200]
                    worst_ts = rec.get("timestamp")
            evidence.append(
                {
                    "source": label,
                    "detail": worst_msg,
                    "score": round(score, 4),
                    "timestamp": worst_ts,
                }
            )

        report_data = {
            "incident_id": incident_id,
            "detected_at": detected_at,
            "summary": f"Anomaly detected in '{top_cause}'. Insufficient signals for causal analysis.",
            "causal_chain": [],
            "evidence": evidence,
            "root_cause": top_cause,
            "confidence": top_confidence,
            "anomaly_scores": {
                k: round(float(v), 4) for k, v in anomaly_scores.items()
            },
            "recommendations": self._flatten_recommendations(remediation_plan),
            "prevention": self._flatten_prevention(remediation_plan),
        }
        narrative = self.nlg_generator.generate_narrative(report_data)

        return {
            "incident_id": incident_id,
            "status": "complete",
            "detected_at": detected_at,
            "log_sources_analyzed": list(all_logs.keys()),
            "total_records_analyzed": sum(len(v) for v in all_logs.values()),
            "anomalous_metrics": [
                {"name": label, "score": score, "first_seen": detected_at}
                for label, score in sorted(
                    anomaly_scores.items(), key=lambda x: x[1], reverse=True
                )
            ],
            "causal_graph": {"nodes": [], "edges": []},
            "ranked_causes": [
                {"rank": i + 1, "cause": label, "confidence": round(score, 4)}
                for i, (label, score) in enumerate(
                    sorted(anomaly_scores.items(), key=lambda x: x[1], reverse=True)
                )
            ],
            "root_cause": top_cause,
            "narrative": narrative,
            "remediation_plan": remediation_plan,
            "evidence": evidence,
        }
