"""Price spike classification for electricity markets.

Price spikes (extreme high prices) are critical events in electricity markets,
often caused by supply shortages, transmission constraints, or extreme weather.
Predicting spike probability enables risk management and trading opportunities.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


class PriceSpikeClassifier:
    """Binary classifier for electricity price spikes.

    A price spike is defined as a price exceeding a threshold (typically 95th
    or 99th percentile). Spike classification complements price forecasting by
    identifying high-risk periods requiring position adjustments.

    Attributes:
        threshold_percentile: Percentile for spike definition (default 95).
        model: Trained RandomForest classifier.
        threshold_price: Absolute price threshold (EUR/MWh).
    """

    def __init__(
        self,
        threshold_percentile: float = 95.0,
        n_estimators: int = 200,
        max_depth: int = 10,
        random_state: int = 42,
    ):
        """Initialize spike classifier.

        Args:
            threshold_percentile: Percentile above which price is a spike.
            n_estimators: Number of trees in random forest.
            max_depth: Maximum tree depth.
            random_state: Random seed for reproducibility.
        """
        self.threshold_percentile = threshold_percentile
        self.threshold_price = None
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",  # Handle imbalanced classes
            random_state=random_state,
            n_jobs=-1,
        )
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train spike classifier.

        Args:
            X: Feature DataFrame.
            y: Price series (continuous values).
        """
        # Calculate spike threshold
        self.threshold_price = np.percentile(y, self.threshold_percentile)

        # Create binary labels
        y_binary = (y > self.threshold_price).astype(int)

        # Train model
        self.model.fit(X, y_binary)
        self.is_fitted = True

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict spike probability.

        Args:
            X: Feature DataFrame.

        Returns:
            np.ndarray: Spike probability for each sample.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        return self.model.predict_proba(X)[:, 1]  # Probability of spike (class 1)

    def predict(self, X: pd.DataFrame, decision_threshold: float = 0.5) -> np.ndarray:
        """Predict spike occurrence.

        Args:
            X: Feature DataFrame.
            decision_threshold: Probability threshold for classification.

        Returns:
            np.ndarray: Binary predictions (0 = no spike, 1 = spike).
        """
        proba = self.predict_proba(X)
        return (proba >= decision_threshold).astype(int)

    def evaluate(
        self, X: pd.DataFrame, y: pd.Series, decision_threshold: float = 0.5
    ) -> Dict[str, float]:
        """Evaluate classifier performance.

        Args:
            X: Feature DataFrame.
            y: True prices.
            decision_threshold: Probability threshold for classification.

        Returns:
            dict: Performance metrics.
        """
        # Create binary labels
        y_true = (y > self.threshold_price).astype(int)

        # Predictions
        y_pred = self.predict(X, decision_threshold)
        y_proba = self.predict_proba(X)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Metrics
        metrics = {
            "accuracy": (tp + tn) / (tp + tn + fp + fn),
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
            "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "f1_score": (
                2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
            ),
            "roc_auc": roc_auc_score(y_true, y_proba),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "spike_rate": float(np.mean(y_true)),
        }

        return metrics

    def get_feature_importance(self) -> pd.Series:
        """Get feature importance for spike prediction.

        Returns:
            pd.Series: Feature importances sorted by value.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")

        importance = pd.Series(
            self.model.feature_importances_, index=self.model.feature_names_in_
        ).sort_values(ascending=False)

        return importance


def calculate_spike_regime_features(prices: pd.Series) -> pd.DataFrame:
    """Calculate features indicating spike regime.

    These features help predict spike probability:
    - Recent price volatility
    - Price momentum
    - Distance from historical mean
    - Recent spike frequency

    Args:
        prices: Price series.

    Returns:
        pd.DataFrame: Spike regime features.
    """
    df = pd.DataFrame(index=prices.index)

    # Volatility features (higher volatility → higher spike risk)
    df["volatility_24h"] = prices.rolling(24).std()
    df["volatility_168h"] = prices.rolling(168).std()
    df["volatility_ratio"] = df["volatility_24h"] / (df["volatility_168h"] + 1e-6)

    # Price level features
    df["price_zscore_24h"] = (prices - prices.rolling(24).mean()) / (
        prices.rolling(24).std() + 1e-6
    )
    df["price_zscore_168h"] = (prices - prices.rolling(168).mean()) / (
        prices.rolling(168).std() + 1e-6
    )

    # Momentum features (rapid price increases → spike risk)
    df["price_change_1h"] = prices.diff(1)
    df["price_change_3h"] = prices.diff(3)
    df["price_pct_change_1h"] = prices.pct_change(1)

    # Historical spike frequency (spikes cluster in time)
    threshold_95 = prices.rolling(720).quantile(0.95)  # 30-day rolling
    spike_indicator = (prices > threshold_95).astype(int)
    df["recent_spike_count_24h"] = spike_indicator.rolling(24).sum()
    df["recent_spike_count_168h"] = spike_indicator.rolling(168).sum()

    return df
