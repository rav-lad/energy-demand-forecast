"""Model monitoring: drift detection and performance tracking.

Tracks model metrics over time and triggers alerts when:
- Prediction accuracy degrades (MAE drift)
- Feature distributions shift (PSI)
- Trading performance deteriorates
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Compute Population Stability Index between two distributions.

    PSI < 0.1: no significant shift
    PSI 0.1-0.25: moderate shift
    PSI > 0.25: significant shift

    Args:
        expected: Baseline distribution (training data).
        actual: Current distribution (recent data).
        bins: Number of bins for discretization.

    Returns:
        PSI value.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)

    # Remove NaNs
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]

    if len(expected) < bins or len(actual) < bins:
        return 0.0

    # Create bins from expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]

    # Convert to proportions with smoothing
    expected_pct = (expected_counts + 1) / (len(expected) + bins)
    actual_pct = (actual_counts + 1) / (len(actual) + bins)

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi)


class ModelMonitor:
    """Monitors model performance and detects drift."""

    def __init__(
        self,
        mae_drift_threshold: float = 1.5,
        psi_threshold: float = 0.25,
        rolling_window_days: int = 7,
    ):
        """
        Args:
            mae_drift_threshold: Alert if rolling MAE > baseline * threshold.
            psi_threshold: Alert if any feature PSI exceeds this.
            rolling_window_days: Window for rolling metric computation.
        """
        self.mae_drift_threshold = mae_drift_threshold
        self.psi_threshold = psi_threshold
        self.rolling_window_days = rolling_window_days
        self.baseline_mae: Optional[float] = None
        self.baseline_features: Optional[pd.DataFrame] = None
        self.history: List[Dict] = []

    def set_baseline(self, mae: float, features_df: Optional[pd.DataFrame] = None):
        """Set baseline metrics from initial training period.

        Args:
            mae: Baseline MAE from walk-forward validation.
            features_df: Baseline feature distributions (training data sample).
        """
        self.baseline_mae = mae
        self.baseline_features = features_df
        logger.info(f"Baseline set: MAE={mae:.4f}")

    def check_prediction_drift(
        self, recent_predictions: pd.DataFrame
    ) -> Dict[str, any]:
        """Check if prediction accuracy has degraded.

        Args:
            recent_predictions: DataFrame with 'actual' and 'predicted' columns.

        Returns:
            Dictionary with drift status and metrics.
        """
        if self.baseline_mae is None:
            logger.warning("No baseline MAE set, skipping prediction drift check")
            return {"drift_detected": False, "reason": "no_baseline"}

        errors = np.abs(recent_predictions["actual"] - recent_predictions["predicted"])
        current_mae = errors.mean()
        ratio = current_mae / self.baseline_mae

        is_drifted = ratio > self.mae_drift_threshold

        result = {
            "drift_detected": is_drifted,
            "current_mae": current_mae,
            "baseline_mae": self.baseline_mae,
            "ratio": ratio,
            "threshold": self.mae_drift_threshold,
            "timestamp": datetime.now().isoformat(),
        }

        if is_drifted:
            logger.warning(
                f"PREDICTION DRIFT DETECTED: MAE={current_mae:.4f} "
                f"({ratio:.2f}x baseline {self.baseline_mae:.4f})"
            )

        self.history.append(result)
        return result

    def check_feature_drift(
        self, current_features: pd.DataFrame
    ) -> Dict[str, Dict]:
        """Check if feature distributions have shifted using PSI.

        Args:
            current_features: Current feature values.

        Returns:
            Dictionary mapping feature names to PSI results.
        """
        if self.baseline_features is None:
            return {}

        results = {}
        drifted_features = []

        for col in current_features.columns:
            if col not in self.baseline_features.columns:
                continue

            psi = compute_psi(
                self.baseline_features[col].values,
                current_features[col].values,
            )

            is_drifted = psi > self.psi_threshold
            results[col] = {
                "psi": psi,
                "drifted": is_drifted,
                "severity": "high" if psi > 0.25 else "moderate" if psi > 0.1 else "low",
            }

            if is_drifted:
                drifted_features.append(col)

        if drifted_features:
            logger.warning(
                f"FEATURE DRIFT DETECTED in {len(drifted_features)} features: "
                f"{drifted_features[:5]}"
            )

        return results

    def check_trading_performance(
        self, pnl_series: pd.Series, sharpe_floor: float = 0.0
    ) -> Dict:
        """Check if trading performance has deteriorated.

        Args:
            pnl_series: Recent daily PnL values.
            sharpe_floor: Minimum acceptable rolling Sharpe.

        Returns:
            Dictionary with trading performance status.
        """
        if len(pnl_series) < 10:
            return {"status": "insufficient_data"}

        rolling_sharpe = (
            (pnl_series.mean() / pnl_series.std()) * np.sqrt(252)
            if pnl_series.std() > 0
            else 0.0
        )
        cumulative_pnl = pnl_series.sum()
        max_dd = (pnl_series.cumsum() - pnl_series.cumsum().expanding().max()).min()

        return {
            "rolling_sharpe": rolling_sharpe,
            "cumulative_pnl": cumulative_pnl,
            "max_drawdown": max_dd,
            "below_floor": rolling_sharpe < sharpe_floor,
            "n_days": len(pnl_series),
        }

    def run_all_checks(
        self,
        recent_predictions: pd.DataFrame = None,
        current_features: pd.DataFrame = None,
        pnl_series: pd.Series = None,
        alert_service=None,
    ) -> Dict:
        """Run all monitoring checks and optionally send alerts.

        Args:
            recent_predictions: DataFrame with actual/predicted columns.
            current_features: Current feature values.
            pnl_series: Recent PnL series.
            alert_service: Optional AlertService for notifications.

        Returns:
            Combined results dictionary.
        """
        results = {"timestamp": datetime.now().isoformat()}

        if recent_predictions is not None:
            pred_drift = self.check_prediction_drift(recent_predictions)
            results["prediction_drift"] = pred_drift
            if pred_drift["drift_detected"] and alert_service:
                alert_service.alert_model_drift(
                    "MAE", pred_drift["current_mae"], pred_drift["baseline_mae"]
                )

        if current_features is not None:
            feat_drift = self.check_feature_drift(current_features)
            results["feature_drift"] = feat_drift
            n_drifted = sum(1 for v in feat_drift.values() if v.get("drifted"))
            if n_drifted > 0 and alert_service:
                alert_service.alert_model_drift(
                    "Feature PSI", float(n_drifted), 0.0
                )

        if pnl_series is not None:
            trading = self.check_trading_performance(pnl_series)
            results["trading_performance"] = trading

        return results
