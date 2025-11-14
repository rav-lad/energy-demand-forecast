"""Market regime detection for electricity markets.

Electricity markets exhibit distinct regimes with different price dynamics:
- Base load regime (demand-driven, stable)
- Renewable flush regime (over-supply, low/negative prices)
- Scarcity regime (supply shortage, price spikes)
- Congestion regime (transmission constraints)

Regime-aware trading adapts strategy parameters to current market conditions.
"""

from enum import Enum
from typing import Dict, Optional

import numpy as np
import pandas as pd


class MarketRegime(Enum):
    """Electricity market regime classifications."""

    BASE_LOAD = "base_load"  # Normal demand-driven regime
    RENEWABLE_FLUSH = "renewable_flush"  # Oversupply from renewables
    SCARCITY = "scarcity"  # Supply shortage, high prices
    HIGH_VOLATILITY = "high_volatility"  # Uncertain/transition period


class ObservableRegimeDetector:
    """Observable-based regime detection.

    Uses market observables (price, renewable share, volatility) to classify
    current regime. Fast and interpretable, unlike HMM-based methods.

    Rules-based classification:
    - Renewable flush: renewable_share > 0.7 AND price < mean
    - Scarcity: price > 95th percentile OR net_load > 90th percentile
    - High volatility: recent_volatility > 2 × historical_volatility
    - Base load: default (none of above)
    """

    def __init__(
        self,
        renewable_flush_threshold: float = 0.70,
        scarcity_price_percentile: float = 95.0,
        volatility_multiplier: float = 2.0,
        lookback_window: int = 168,  # 1 week
    ):
        """Initialize regime detector.

        Args:
            renewable_flush_threshold: Renewable share threshold for flush regime.
            scarcity_price_percentile: Price percentile for scarcity regime.
            volatility_multiplier: Volatility threshold relative to historical.
            lookback_window: Hours for historical statistics.
        """
        self.renewable_flush_threshold = renewable_flush_threshold
        self.scarcity_price_percentile = scarcity_price_percentile
        self.volatility_multiplier = volatility_multiplier
        self.lookback_window = lookback_window

    def detect_regime(
        self,
        prices: pd.Series,
        renewable_share: Optional[pd.Series] = None,
        net_load: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Detect market regime at each timestamp.

        Args:
            prices: Electricity prices (EUR/MWh).
            renewable_share: Optional renewable share (0-1).
            net_load: Optional net load after renewables (MW).

        Returns:
            pd.Series: Regime classification for each timestamp.
        """
        regimes = pd.Series(index=prices.index, dtype=str)

        # Calculate rolling statistics
        price_mean = prices.rolling(self.lookback_window, min_periods=24).mean()
        price_p95 = prices.rolling(self.lookback_window, min_periods=24).quantile(
            self.scarcity_price_percentile / 100
        )

        # Volatility
        recent_vol = prices.rolling(24).std()
        hist_vol = prices.rolling(self.lookback_window, min_periods=24).std()
        vol_ratio = recent_vol / (hist_vol + 1e-6)

        # Initialize all as base load
        regimes[:] = MarketRegime.BASE_LOAD.value

        # Renewable flush regime
        if renewable_share is not None:
            flush_mask = (renewable_share > self.renewable_flush_threshold) & (
                prices < price_mean
            )
            regimes[flush_mask] = MarketRegime.RENEWABLE_FLUSH.value

        # Scarcity regime (overrides flush)
        scarcity_mask = prices > price_p95
        if net_load is not None:
            net_load_p90 = net_load.rolling(
                self.lookback_window, min_periods=24
            ).quantile(0.90)
            scarcity_mask = scarcity_mask | (net_load > net_load_p90)

        regimes[scarcity_mask] = MarketRegime.SCARCITY.value

        # High volatility regime (overrides base load only)
        high_vol_mask = (vol_ratio > self.volatility_multiplier) & (
            regimes == MarketRegime.BASE_LOAD.value
        )
        regimes[high_vol_mask] = MarketRegime.HIGH_VOLATILITY.value

        return regimes

    def get_regime_statistics(
        self, regimes: pd.Series, prices: pd.Series
    ) -> pd.DataFrame:
        """Calculate statistics for each regime.

        Args:
            regimes: Regime classifications.
            prices: Electricity prices.

        Returns:
            pd.DataFrame: Statistics by regime (mean, std, count, frequency).
        """
        df = pd.DataFrame({"regime": regimes, "price": prices})

        stats = df.groupby("regime").agg(
            {
                "price": [
                    "count",
                    "mean",
                    "std",
                    "min",
                    "max",
                    lambda x: np.percentile(x, 25),
                    lambda x: np.percentile(x, 75),
                ]
            }
        )

        stats.columns = ["count", "mean", "std", "min", "max", "p25", "p75"]
        stats["frequency"] = stats["count"] / len(df)
        stats["volatility"] = stats["std"] / stats["mean"]

        return stats.sort_values("mean", ascending=False)

    def get_regime_transitions(self, regimes: pd.Series) -> pd.DataFrame:
        """Calculate regime transition matrix.

        Args:
            regimes: Regime classifications.

        Returns:
            pd.DataFrame: Transition probability matrix.
        """
        # Create transition pairs
        from_regime = regimes[:-1].values
        to_regime = regimes[1:].values

        # Count transitions
        unique_regimes = regimes.unique()
        transition_counts = pd.DataFrame(
            0, index=unique_regimes, columns=unique_regimes
        )

        for from_r, to_r in zip(from_regime, to_regime):
            transition_counts.loc[from_r, to_r] += 1

        # Convert to probabilities
        transition_probs = transition_counts.div(
            transition_counts.sum(axis=1), axis=0
        ).fillna(0)

        return transition_probs


def calculate_regime_features(
    prices: pd.Series,
    renewable_share: Optional[pd.Series] = None,
    net_load: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Calculate features for regime detection.

    Args:
        prices: Electricity prices.
        renewable_share: Optional renewable share.
        net_load: Optional net load.

    Returns:
        pd.DataFrame: Regime-related features.
    """
    df = pd.DataFrame(index=prices.index)

    # Price-based features
    df["price_zscore_24h"] = (prices - prices.rolling(24).mean()) / (
        prices.rolling(24).std() + 1e-6
    )
    df["price_zscore_168h"] = (prices - prices.rolling(168).mean()) / (
        prices.rolling(168).std() + 1e-6
    )

    # Volatility features
    df["volatility_24h"] = prices.rolling(24).std()
    df["volatility_168h"] = prices.rolling(168).std()
    df["volatility_ratio"] = df["volatility_24h"] / (df["volatility_168h"] + 1e-6)

    # Momentum
    df["price_momentum_3h"] = prices.diff(3)
    df["price_momentum_24h"] = prices.diff(24)

    # Range features
    df["price_range_24h"] = prices.rolling(24).max() - prices.rolling(24).min()
    df["price_position_in_range"] = (prices - prices.rolling(24).min()) / (
        df["price_range_24h"] + 1e-6
    )

    # Renewable features (if available)
    if renewable_share is not None:
        df["renewable_share"] = renewable_share
        df["renewable_share_ma24"] = renewable_share.rolling(24).mean()
        df["high_renewable_flag"] = (renewable_share > 0.7).astype(int)

    # Net load features (if available)
    if net_load is not None:
        df["net_load"] = net_load
        df["net_load_zscore"] = (net_load - net_load.rolling(168).mean()) / (
            net_load.rolling(168).std() + 1e-6
        )
        df["net_load_high_flag"] = (
            net_load > net_load.rolling(168).quantile(0.90)
        ).astype(int)

    return df


def get_regime_conditional_stats(
    regimes: pd.Series, returns: pd.Series
) -> Dict[str, Dict[str, float]]:
    """Calculate return statistics conditional on regime.

    Useful for regime-adaptive trading strategies.

    Args:
        regimes: Regime classifications.
        returns: Strategy returns.

    Returns:
        dict: Statistics by regime (mean return, volatility, Sharpe).
    """
    df = pd.DataFrame({"regime": regimes, "returns": returns})

    stats = {}
    for regime in df["regime"].unique():
        regime_returns = df[df["regime"] == regime]["returns"]

        stats[regime] = {
            "mean_return": regime_returns.mean(),
            "volatility": regime_returns.std(),
            "sharpe": (
                regime_returns.mean() / regime_returns.std()
                if regime_returns.std() > 0
                else 0
            ),
            "count": len(regime_returns),
            "win_rate": (regime_returns > 0).mean(),
        }

    return stats
