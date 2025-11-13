"""
Demand-Price Arbitrage Trading Strategy

This strategy exploits the relationship between predicted energy demand,
renewable production, and electricity prices.

Strategy Logic:
- BUY signal: High demand predicted + low renewable production → prices likely to rise
- SELL signal: Low demand predicted + high renewable production → prices likely to fall
"""

import logging
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DemandPriceArbitrageStrategy:
    """
    Trading strategy based on demand forecasts and renewable production.

    This strategy generates trading signals by analyzing:
    1. Predicted demand vs historical distribution
    2. Renewable production forecast
    3. Residual load (demand - renewable production)
    """

    def __init__(
        self,
        buy_demand_threshold: float = 0.95,
        sell_demand_threshold: float = 0.25,
        renewable_high_threshold: float = 0.7,
        renewable_low_threshold: float = 0.3,
        base_position_size: float = 1000.0,
        use_confidence_scaling: bool = True,
    ):
        """
        Initialize strategy.

        Args:
            buy_demand_threshold: Percentile above which to consider buying (0-1)
            sell_demand_threshold: Percentile below which to consider selling (0-1)
            renewable_high_threshold: Renewable share threshold for sell signal (0-1)
            renewable_low_threshold: Renewable share threshold for buy signal (0-1)
            base_position_size: Base position size in MWh
            use_confidence_scaling: Scale position size by signal confidence
        """
        self.buy_demand_threshold = buy_demand_threshold
        self.sell_demand_threshold = sell_demand_threshold
        self.renewable_high_threshold = renewable_high_threshold
        self.renewable_low_threshold = renewable_low_threshold
        self.base_position_size = base_position_size
        self.use_confidence_scaling = use_confidence_scaling

        # Historical statistics (to be calculated from training data)
        self.demand_quantiles = None
        self.renewable_stats = None

        logger.info(f"Initialized Demand-Price Arbitrage Strategy")
        logger.info(f"  Buy threshold: {buy_demand_threshold:.2f} (demand percentile)")
        logger.info(f"  Sell threshold: {sell_demand_threshold:.2f} (demand percentile)")
        logger.info(f"  Renewable high: {renewable_high_threshold:.2f}")
        logger.info(f"  Renewable low: {renewable_low_threshold:.2f}")

    def calibrate(
        self,
        historical_demand: pd.Series,
        historical_renewable_share: Optional[pd.Series] = None,
    ):
        """
        Calibrate strategy parameters using historical data.

        Args:
            historical_demand: Historical demand data
            historical_renewable_share: Historical renewable production share (0-1)
        """
        logger.info("Calibrating strategy on historical data...")

        # Calculate demand quantiles
        self.demand_quantiles = {
            "q05": historical_demand.quantile(0.05),
            "q25": historical_demand.quantile(0.25),
            "q50": historical_demand.quantile(0.50),
            "q75": historical_demand.quantile(0.75),
            "q95": historical_demand.quantile(0.95),
            "mean": historical_demand.mean(),
            "std": historical_demand.std(),
        }

        logger.info(f"  Demand Q95: {self.demand_quantiles['q95']:.2f} MW")
        logger.info(f"  Demand Q25: {self.demand_quantiles['q25']:.2f} MW")

        # Calculate renewable statistics if available
        if historical_renewable_share is not None:
            self.renewable_stats = {
                "mean": historical_renewable_share.mean(),
                "std": historical_renewable_share.std(),
                "q25": historical_renewable_share.quantile(0.25),
                "q75": historical_renewable_share.quantile(0.75),
            }
            logger.info(f"  Renewable mean: {self.renewable_stats['mean']:.2%}")

    def generate_signals(
        self,
        predicted_demand: pd.Series,
        renewable_share: Optional[pd.Series] = None,
        actual_demand: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Generate trading signals based on predictions.

        Args:
            predicted_demand: Predicted demand values
            renewable_share: Renewable production share (0-1)
            actual_demand: Actual demand (optional, for validation)

        Returns:
            DataFrame with signals and metadata
        """
        if self.demand_quantiles is None:
            raise ValueError("Strategy not calibrated. Call calibrate() first.")

        logger.info(f"Generating signals for {len(predicted_demand)} periods")

        # Initialize signals DataFrame
        signals_df = pd.DataFrame(index=predicted_demand.index)
        signals_df["predicted_demand"] = predicted_demand
        signals_df["signal"] = 0  # 0 = HOLD, 1 = BUY, -1 = SELL

        # Calculate demand percentile for each prediction
        signals_df["demand_percentile"] = predicted_demand.rank(pct=True)

        # Add renewable data if available
        if renewable_share is not None:
            signals_df["renewable_share"] = renewable_share
        else:
            # Use mean if not available
            signals_df["renewable_share"] = (
                self.renewable_stats["mean"] if self.renewable_stats else 0.3
            )

        # Calculate residual load (demand - renewable production)
        # Assuming total capacity enables this calculation
        signals_df["residual_load"] = predicted_demand * (1 - signals_df["renewable_share"])

        # --- GENERATE BUY SIGNALS ---
        # Buy when: High demand + Low renewable production
        buy_condition = (signals_df["demand_percentile"] >= self.buy_demand_threshold) & (
            signals_df["renewable_share"] <= self.renewable_low_threshold
        )
        signals_df.loc[buy_condition, "signal"] = 1

        # --- GENERATE SELL SIGNALS ---
        # Sell when: Low demand + High renewable production
        sell_condition = (signals_df["demand_percentile"] <= self.sell_demand_threshold) & (
            signals_df["renewable_share"] >= self.renewable_high_threshold
        )
        signals_df.loc[sell_condition, "signal"] = -1

        # Calculate signal confidence (how extreme is the condition)
        signals_df["confidence"] = 0.0

        # Buy confidence: based on how high demand is and how low renewable is
        buy_mask = signals_df["signal"] == 1
        signals_df.loc[buy_mask, "confidence"] = (
            signals_df.loc[buy_mask, "demand_percentile"] - self.buy_demand_threshold
        ) / (1 - self.buy_demand_threshold) * 0.5 + (
            self.renewable_low_threshold - signals_df.loc[buy_mask, "renewable_share"]
        ) / self.renewable_low_threshold * 0.5

        # Sell confidence: based on how low demand is and how high renewable is
        sell_mask = signals_df["signal"] == -1
        signals_df.loc[sell_mask, "confidence"] = (
            self.sell_demand_threshold - signals_df.loc[sell_mask, "demand_percentile"]
        ) / self.sell_demand_threshold * 0.5 + (
            signals_df.loc[sell_mask, "renewable_share"] - self.renewable_high_threshold
        ) / (
            1 - self.renewable_high_threshold
        ) * 0.5

        # Clip confidence to [0, 1]
        signals_df["confidence"] = signals_df["confidence"].clip(0, 1)

        # Calculate position size
        if self.use_confidence_scaling:
            # Scale position size by confidence (0.5x to 1.5x base size)
            signals_df["position_size"] = self.base_position_size * (0.5 + signals_df["confidence"])
        else:
            signals_df["position_size"] = self.base_position_size

        # Add actual demand if available (for analysis)
        if actual_demand is not None:
            signals_df["actual_demand"] = actual_demand

        # Log signal statistics
        n_buy = (signals_df["signal"] == 1).sum()
        n_sell = (signals_df["signal"] == -1).sum()
        n_hold = (signals_df["signal"] == 0).sum()

        logger.info(f"Signals generated:")
        logger.info(f"  BUY:  {n_buy:>6} ({n_buy/len(signals_df)*100:.1f}%)")
        logger.info(f"  SELL: {n_sell:>6} ({n_sell/len(signals_df)*100:.1f}%)")
        logger.info(f"  HOLD: {n_hold:>6} ({n_hold/len(signals_df)*100:.1f}%)")

        if n_buy + n_sell > 0:
            avg_confidence = signals_df[signals_df["signal"] != 0]["confidence"].mean()
            logger.info(f"  Avg confidence: {avg_confidence:.2%}")

        return signals_df

    def analyze_signal_quality(self, signals_df: pd.DataFrame, actual_prices: pd.Series) -> Dict:
        """
        Analyze the quality of generated signals by checking if they
        preceded favorable price movements.

        Args:
            signals_df: DataFrame with signals (from generate_signals)
            actual_prices: Actual price series

        Returns:
            Dictionary with signal quality metrics
        """
        # Calculate forward returns
        forward_returns = actual_prices.pct_change(periods=1).shift(-1)

        signals_with_returns = signals_df.copy()
        signals_with_returns["forward_return"] = forward_returns

        # Analyze buy signals
        buy_signals = signals_with_returns[signals_with_returns["signal"] == 1]
        buy_accuracy = (
            (buy_signals["forward_return"] > 0).sum() / len(buy_signals)
            if len(buy_signals) > 0
            else 0
        )
        avg_buy_return = buy_signals["forward_return"].mean() if len(buy_signals) > 0 else 0

        # Analyze sell signals
        sell_signals = signals_with_returns[signals_with_returns["signal"] == -1]
        sell_accuracy = (
            (sell_signals["forward_return"] < 0).sum() / len(sell_signals)
            if len(sell_signals) > 0
            else 0
        )
        avg_sell_return = sell_signals["forward_return"].mean() if len(sell_signals) > 0 else 0

        metrics = {
            "buy_signals": len(buy_signals),
            "buy_accuracy": buy_accuracy * 100,
            "avg_buy_return": avg_buy_return * 100,
            "sell_signals": len(sell_signals),
            "sell_accuracy": sell_accuracy * 100,
            "avg_sell_return": avg_sell_return * 100,
            "total_accuracy": (
                (buy_accuracy * len(buy_signals) + sell_accuracy * len(sell_signals))
                / (len(buy_signals) + len(sell_signals))
                * 100
                if (len(buy_signals) + len(sell_signals)) > 0
                else 0
            ),
        }

        logger.info("\nSignal Quality Analysis:")
        logger.info(f"  Buy Signal Accuracy: {metrics['buy_accuracy']:.1f}%")
        logger.info(f"  Avg Buy Return: {metrics['avg_buy_return']:.2f}%")
        logger.info(f"  Sell Signal Accuracy: {metrics['sell_accuracy']:.1f}%")
        logger.info(f"  Avg Sell Return: {metrics['avg_sell_return']:.2f}%")
        logger.info(f"  Overall Accuracy: {metrics['total_accuracy']:.1f}%")

        return metrics


if __name__ == "__main__":
    # Example usage
    print("Demand-Price Arbitrage Strategy - Example\n")

    # Create synthetic data
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")

    # Synthetic demand with seasonality
    t = np.arange(len(dates))
    demand = 5000 + 1000 * np.sin(2 * np.pi * t / 365) + np.random.randn(len(dates)) * 200

    # Synthetic renewable share
    renewable = (
        0.3 + 0.2 * np.sin(2 * np.pi * t / 365 + np.pi / 2) + np.random.randn(len(dates)) * 0.05
    )
    renewable = renewable.clip(0, 1)

    # Synthetic prices (correlated with demand, anti-correlated with renewables)
    prices = 50 + (demand - demand.mean()) / 100 - renewable * 20 + np.random.randn(len(dates)) * 5

    demand_series = pd.Series(demand, index=dates)
    renewable_series = pd.Series(renewable, index=dates)
    price_series = pd.Series(prices, index=dates)

    # Initialize and calibrate strategy
    strategy = DemandPriceArbitrageStrategy(
        buy_demand_threshold=0.90,
        sell_demand_threshold=0.10,
        renewable_high_threshold=0.6,
        renewable_low_threshold=0.2,
    )

    # Use first 6 months for calibration
    calibration_end = pd.Timestamp("2023-06-30")
    strategy.calibrate(demand_series[:calibration_end], renewable_series[:calibration_end])

    # Generate signals for full period
    signals_df = strategy.generate_signals(demand_series, renewable_series)

    # Analyze signal quality
    metrics = strategy.analyze_signal_quality(signals_df, price_series)

    # Display sample signals
    print("\nSample Signals:")
    print(signals_df[signals_df["signal"] != 0].head(10))
