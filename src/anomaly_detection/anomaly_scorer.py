"""Re-exports the maintained detector.

This module used to define its own full copy of ``AnomalyDetector`` --
hardcoded checkpoint filename, 95th-percentile threshold instead of the
maintained 99th, no ``windows=``/``on_epoch`` support. It was never actually
instantiated anywhere (``ensemble_detector.py`` only used it as a type
annotation), so it sat importable with silently different, wrong behaviour:
anyone who did construct it directly would get a different detection
threshold than the rest of the pipeline uses, with no error to say so.

Kept as a re-export, not deleted outright, so the import in
``ensemble_detector.py`` (``from anomaly_detection.anomaly_scorer import
AnomalyDetector``) keeps working without every caller needing to change.
The one and only implementation now lives in ``models.lstm_autoencoder``.
"""

from models.lstm_autoencoder import AnomalyDetector

__all__ = ["AnomalyDetector"]
