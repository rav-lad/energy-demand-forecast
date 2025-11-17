"""
Baseline Models Package

Simple baseline models for price forecasting comparison.
Essential for validating that complex ML models add value.
"""

from .naive_baselines import (
    BaselineComparator,
    BaselineMetrics,
    BaselineModel,
    HistoricalMeanBaseline,
    MovingAverageBaseline,
    PersistenceBaseline,
    SeasonalMeanBaseline,
    SeasonalNaiveBaseline,
)

__all__ = [
    "BaselineModel",
    "BaselineMetrics",
    "BaselineComparator",
    "PersistenceBaseline",
    "HistoricalMeanBaseline",
    "SeasonalNaiveBaseline",
    "SeasonalMeanBaseline",
    "MovingAverageBaseline",
]
