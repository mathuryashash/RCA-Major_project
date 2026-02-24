"""
rca-system.src.preprocessing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Public API for the preprocessing sub-package:

    DataPreprocessor       — normalisation + sliding window creation
    TimeSeriesAligner      — multi-source time-series alignment to a common grid
    PreprocessingPipeline  — full orchestrator (align -> normalise -> window)
"""

from .data_normalizer import DataPreprocessor
from .time_series_aligner import TimeSeriesAligner
from .pipeline import PreprocessingPipeline

__all__ = [
    'DataPreprocessor',
    'TimeSeriesAligner',
    'PreprocessingPipeline',
]
