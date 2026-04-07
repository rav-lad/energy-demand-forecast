"""LightGBM quantile regression wrapper for electricity price forecasting.

Importable from anywhere so joblib pickle/unpickle works correctly.
"""

from __future__ import annotations

import numpy as np
import lightgbm as lgb


class QuantileWrapper:
    """sklearn-compatible LightGBM quantile forecaster.

    Trains one LGBMRegressor per quantile and exposes:
      - predict(X)           → median (p50) — used by walk-forward validator
      - predict_interval(X)  → (lower, median, upper)
    """

    def __init__(self, quantiles: list[float] | None = None, **lgb_params):
        self.quantiles = sorted(quantiles) if quantiles else [0.05, 0.5, 0.95]
        self.lgb_params = lgb_params
        self.models: dict[float, lgb.LGBMRegressor] = {}

    def fit(self, X, y) -> "QuantileWrapper":
        for q in self.quantiles:
            params = dict(self.lgb_params, objective="quantile", alpha=q)
            m = lgb.LGBMRegressor(**params)
            m.fit(X, y)
            self.models[q] = m
        return self

    def predict(self, X) -> np.ndarray:
        """Return median (p50) prediction."""
        return self.models[0.5].predict(X)

    def predict_interval(self, X) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (lower, median, upper) arrays."""
        qs = self.quantiles  # already sorted
        return (
            self.models[qs[0]].predict(X),
            self.models[0.5].predict(X),
            self.models[qs[-1]].predict(X),
        )
