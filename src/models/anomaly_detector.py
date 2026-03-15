import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import deque


class SimpleLogAnomalyDetector:
    """
    A simple log anomaly detector using TF-IDF and sequence profiling.
    """

    def __init__(self, window_size=60, threshold_percentile=99):
        self.window_size = window_size
        self.threshold_percentile = threshold_percentile
        self.vectorizer = TfidfVectorizer()
        self.history = []  # For training the baseline
        self.is_trained = False
        self.baseline_scores = []

    def train(self, training_logs):
        """
        Trains the TF-IDF vectorizer on healthy log messages.
        """
        if not training_logs:
            return

        self.vectorizer.fit(training_logs)
        tfidf_matrix = self.vectorizer.transform(training_logs)

        # Calculate mean TF-IDF vector for "normal" profile
        # Convert from np.matrix to np.ndarray for sklearn compatibility
        self.normal_profile = np.asarray(tfidf_matrix.mean(axis=0))

        # Calculate baseline scores (cosine distance from normal profile)
        from sklearn.metrics.pairwise import cosine_distances

        distances = cosine_distances(tfidf_matrix, self.normal_profile)
        self.baseline_scores = distances.flatten()
        self.threshold = np.percentile(self.baseline_scores, self.threshold_percentile)

        self.is_trained = True
        print(f"Log anomaly detector trained. Threshold: {self.threshold}")

    def score(self, log_messages):
        """
        Returns anomaly scores for a list of log messages.
        """
        if not self.is_trained or not log_messages:
            return np.zeros(len(log_messages))

        tfidf_matrix = self.vectorizer.transform(log_messages)
        from sklearn.metrics.pairwise import cosine_distances

        distances = cosine_distances(tfidf_matrix, self.normal_profile)

        return distances.flatten()


class MetricAnomalyDetector:
    """
    Simple statistical anomaly detector for numeric metrics.
    """

    def __init__(self, z_threshold=3):
        self.z_threshold = z_threshold
        self.stats = {}  # {metric_name: (mean, std)}

    def update_stats(self, metric_name, values):
        self.stats[metric_name] = (np.mean(values), np.std(values))

    def score(self, metric_name, value):
        if metric_name not in self.stats:
            return 0

        mean, std = self.stats[metric_name]
        if std == 0:
            return 0

        z_score = abs(value - mean) / std
        return min(1.0, z_score / self.z_threshold)


if __name__ == "__main__":
    # Test Log Anomaly Detector
    detector = SimpleLogAnomalyDetector()
    training = [
        "User login successful",
        "Data sync completed",
        "Heartbeat received",
        "Connection healthy",
    ]
    detector.train(training)

    test_logs = ["Connection refused to database", "User login successful"]
    scores = detector.score(test_logs)
    for log, score in zip(test_logs, scores):
        is_anomaly = score > detector.threshold
        print(f"Log: {log} | Score: {score:.4f} | Anomaly: {is_anomaly}")
