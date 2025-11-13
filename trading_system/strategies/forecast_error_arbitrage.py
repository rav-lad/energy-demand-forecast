"""
Forecast Error Arbitrage Strategy for Electricity Markets

This strategy exploits the difference between official market forecasts (ENTSO-E)
and proprietary ML model forecasts. When market participants are systematically
wrong about demand, prices will adjust, creating trading opportunities.

Core Insight:
If official forecast underestimates demand → Prices will rise when actual revealed
If official forecast overestimates demand → Prices will fall when actual revealed

Author: Quant Research Team
Date: 2024-11-12
"""

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class ForecastArbitrageConfig:
    """Configuration for Forecast Error Arbitrage Strategy."""

    forecast_error_threshold: float = 200.0  # MW threshold for signal generation
    forecast_error_percentile: float = 0.75  # Use 75th percentile if threshold None
    confidence_threshold: float = 0.6  # Minimum model confidence (0-1)
    max_position_size: float = 1000.0  # Maximum position size (MW equivalent)
    position_scaling: str = "linear"  # 'linear', 'sqrt', or 'log'
    hold_period: int = 1  # Days to hold position (day-ahead market)
    transaction_cost_bps: float = 10.0  # Transaction cost (basis points)
    slippage_model: str = "linear"  # 'linear', 'sqrt', or 'impact'
    slippage_coefficient: float = 0.0001  # Slippage per MW
    stop_loss_pct: float = 0.05  # Stop-loss at 5% loss
    take_profit_pct: float = 0.10  # Take profit at 10% gain


class ForecastErrorArbitrageStrategy:
    """
    Forecast Error Arbitrage Strategy.

    Exploits systematic differences between market consensus forecasts
    and proprietary ML forecasts to generate alpha.

    Signal Generation Logic:
    1. Calculate forecast error: our_forecast - market_forecast
    2. If error > threshold and confidence high → Take position
    3. Direction: positive error → BUY (expect price rise)
                 negative error → SELL (expect price fall)
    4. Position size: scaled by error magnitude and confidence

    Risk Management:
    - Maximum position limits
    - Stop-loss on adverse moves
    - Take-profit at target returns
    - Transaction cost and slippage modeling

    Example:
        >>> config = ForecastArbitrageConfig(forecast_error_threshold=200)
        >>> strategy = ForecastErrorArbitrageStrategy(config)
        >>> signals = strategy.generate_signals(
        ...     our_forecasts=ml_forecasts,
        ...     market_forecasts=entsoe_forecasts,
        ...     actual_demand=actual_data
        ... )
    """

    def __init__(self, config: Optional[ForecastArbitrageConfig] = None):
        """
        Initialize Forecast Error Arbitrage Strategy.

        Args:
            config: Strategy configuration. Uses defaults if None.
        """
        self.config = config or ForecastArbitrageConfig()
        self.is_calibrated = False
        self.error_threshold = None
        self.historical_ic = None  # Information Coefficient

        logger.info(f"Initialized Forecast Arbitrage Strategy with config: {self.config}")

    def calculate_information_coefficient(self, forecasts: pd.Series, actuals: pd.Series) -> float:
        """
        Calculate Information Coefficient (IC): correlation between forecasts and actuals.

        IC measures forecast skill:
        - IC > 0.05: Good
        - IC > 0.10: Excellent
        - IC > 0.15: Exceptional

        Args:
            forecasts: Forecast time series
            actuals: Actual values time series

        Returns:
            IC value (Spearman correlation)
        """
        # Align indices
        common_idx = forecasts.index.intersection(actuals.index)
        forecasts_aligned = forecasts.loc[common_idx]
        actuals_aligned = actuals.loc[common_idx]

        # Calculate Spearman correlation (rank-based, more robust)
        ic, p_value = stats.spearmanr(forecasts_aligned, actuals_aligned)

        logger.info(f"Information Coefficient: {ic:.4f} (p-value: {p_value:.4e})")

        return ic

    def calibrate(
        self,
        our_forecasts: pd.Series,
        market_forecasts: pd.Series,
        actual_demand: pd.Series,
        prices: Optional[pd.Series] = None,
    ) -> None:
        """
        Calibrate strategy on historical data.

        Estimates:
        - Optimal forecast error threshold
        - Information Coefficient of our forecasts
        - Price sensitivity to forecast errors

        Args:
            our_forecasts: Our ML model forecasts
            market_forecasts: Official market forecasts (ENTSO-E)
            actual_demand: Actual realized demand
            prices: Historical prices (optional, for price sensitivity)
        """
        logger.info(f"Calibrating strategy on {len(actual_demand)} data points...")

        # Calculate forecast errors
        forecast_errors = our_forecasts - market_forecasts
        our_errors = our_forecasts - actual_demand
        market_errors = market_forecasts - actual_demand

        # Calculate ICs
        ic_ours = self.calculate_information_coefficient(our_forecasts, actual_demand)
        ic_market = self.calculate_information_coefficient(market_forecasts, actual_demand)

        self.historical_ic = ic_ours

        logger.info(f"Our IC: {ic_ours:.4f}, Market IC: {ic_market:.4f}")

        if ic_ours <= ic_market:
            logger.warning(
                f"Our forecasts (IC={ic_ours:.4f}) not better than market "
                f"(IC={ic_market:.4f}). Strategy may not be profitable."
            )

        # Determine optimal threshold
        if self.config.forecast_error_threshold is None:
            # Use percentile-based threshold
            self.error_threshold = np.abs(forecast_errors).quantile(
                self.config.forecast_error_percentile
            )
        else:
            self.error_threshold = self.config.forecast_error_threshold

        logger.info(f"Forecast error threshold: {self.error_threshold:.2f} MW")

        # Analyze price sensitivity (if prices provided)
        if prices is not None:
            self._analyze_price_sensitivity(forecast_errors, prices)

        self.is_calibrated = True
        logger.info("Calibration complete.")

    def _analyze_price_sensitivity(self, forecast_errors: pd.Series, prices: pd.Series) -> None:
        """
        Analyze how forecast errors affect price changes.

        Args:
            forecast_errors: Difference between our and market forecasts
            prices: Historical price series
        """
        # Price changes
        price_changes = prices.diff()

        # Align data
        common_idx = forecast_errors.index.intersection(price_changes.index)
        errors_aligned = forecast_errors.loc[common_idx]
        changes_aligned = price_changes.loc[common_idx]

        # Calculate correlation
        corr, p_value = stats.pearsonr(errors_aligned, changes_aligned)

        logger.info(
            f"Forecast error - Price change correlation: {corr:.4f} " f"(p-value: {p_value:.4e})"
        )

        if corr > 0.1:
            logger.info("Positive correlation detected: " "Forecast errors predict price direction")
        else:
            logger.warning("Weak correlation: " "Forecast errors may not strongly predict prices")

    def calculate_position_size(self, forecast_error: float, confidence: float = 1.0) -> float:
        """
        Calculate position size based on forecast error magnitude and confidence.

        Position sizing methods:
        - 'linear': size ∝ error
        - 'sqrt': size ∝ sqrt(|error|) (less aggressive)
        - 'log': size ∝ log(1 + |error|) (even less aggressive)

        Args:
            forecast_error: Difference between forecasts (MW)
            confidence: Model confidence (0-1)

        Returns:
            Position size (MW equivalent)
        """
        error_magnitude = abs(forecast_error)

        # Base size from error magnitude
        if self.config.position_scaling == "linear":
            base_size = error_magnitude
        elif self.config.position_scaling == "sqrt":
            base_size = np.sqrt(error_magnitude) * 10  # Scale factor
        elif self.config.position_scaling == "log":
            base_size = np.log1p(error_magnitude) * 50  # Scale factor
        else:
            raise ValueError(f"Unknown scaling method: {self.config.position_scaling}")

        # Scale by confidence
        scaled_size = base_size * confidence

        # Cap at maximum
        position_size = min(scaled_size, self.config.max_position_size)

        return position_size

    def calculate_slippage(self, position_size: float, price: float) -> float:
        """
        Calculate slippage cost based on position size.

        Slippage models:
        - 'linear': slippage ∝ size
        - 'sqrt': slippage ∝ sqrt(size) (market impact)
        - 'impact': slippage ∝ size^1.5 (more realistic for large orders)

        Args:
            position_size: Position size (MW)
            price: Current price (EUR/MWh)

        Returns:
            Slippage cost (EUR)
        """
        if self.config.slippage_model == "linear":
            slippage_pct = self.config.slippage_coefficient * position_size
        elif self.config.slippage_model == "sqrt":
            slippage_pct = self.config.slippage_coefficient * np.sqrt(position_size)
        elif self.config.slippage_model == "impact":
            slippage_pct = self.config.slippage_coefficient * (position_size**1.5)
        else:
            slippage_pct = 0.0

        slippage_cost = price * slippage_pct

        return slippage_cost

    def generate_signals(
        self,
        our_forecasts: pd.Series,
        market_forecasts: pd.Series,
        prices: pd.Series,
        model_confidence: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Generate trading signals based on forecast errors.

        Args:
            our_forecasts: Our ML model forecasts (MW)
            market_forecasts: Official market forecasts (MW)
            prices: Historical prices (EUR/MWh)
            model_confidence: Optional confidence scores (0-1)

        Returns:
            DataFrame with columns:
                - our_forecast: Our forecast
                - market_forecast: Market forecast
                - forecast_error: Difference
                - price: Price
                - confidence: Model confidence
                - signal: Trading signal (-1, 0, 1)
                - position_size: Position size (MW)
                - entry_price: Entry price
                - transaction_cost: Transaction cost (EUR)
                - slippage: Slippage cost (EUR)
        """
        if not self.is_calibrated:
            logger.warning("Strategy not calibrated. Running calibration...")
            # Need actual demand for calibration
            raise ValueError(
                "Strategy must be calibrated before generating signals. "
                "Call calibrate() with historical data first."
            )

        # Create DataFrame
        df = pd.DataFrame(
            {"our_forecast": our_forecasts, "market_forecast": market_forecasts, "price": prices}
        )

        # Add confidence (default to 1.0 if not provided)
        if model_confidence is not None:
            df["confidence"] = model_confidence
        else:
            df["confidence"] = 1.0

        # Calculate forecast error
        df["forecast_error"] = df["our_forecast"] - df["market_forecast"]

        # Generate signals
        df["signal"] = 0

        # BUY signal: our forecast > market forecast + threshold
        buy_condition = (df["forecast_error"] > self.error_threshold) & (
            df["confidence"] >= self.config.confidence_threshold
        )
        df.loc[buy_condition, "signal"] = 1

        # SELL signal: our forecast < market forecast - threshold
        sell_condition = (df["forecast_error"] < -self.error_threshold) & (
            df["confidence"] >= self.config.confidence_threshold
        )
        df.loc[sell_condition, "signal"] = -1

        # Calculate position sizes
        df["position_size"] = df.apply(
            lambda row: (
                self.calculate_position_size(row["forecast_error"], row["confidence"])
                if row["signal"] != 0
                else 0
            ),
            axis=1,
        )

        # Entry prices (with bid-ask spread)
        df["entry_price"] = df["price"]

        # Calculate costs
        # Transaction cost (basis points)
        df["transaction_cost"] = (
            df["position_size"] * df["entry_price"] * (self.config.transaction_cost_bps / 10000)
        )

        # Slippage
        df["slippage"] = df.apply(
            lambda row: (
                self.calculate_slippage(row["position_size"], row["entry_price"])
                if row["signal"] != 0
                else 0
            ),
            axis=1,
        )

        # Total cost
        df["total_cost"] = df["transaction_cost"] + df["slippage"]

        logger.info(
            f"Generated signals: {(df['signal'] == 1).sum()} buys, "
            f"{(df['signal'] == -1).sum()} sells, "
            f"Avg position size: {df[df['signal'] != 0]['position_size'].mean():.2f} MW"
        )

        return df

    def calculate_performance_metrics(
        self, signals_df: pd.DataFrame, actual_demand: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate strategy performance metrics.

        Args:
            signals_df: DataFrame from generate_signals()
            actual_demand: Actual realized demand

        Returns:
            Dictionary with performance metrics
        """
        df = signals_df.copy()

        # Align actual demand
        df["actual_demand"] = actual_demand

        # Calculate realized forecast error
        df["realized_error"] = df["actual_demand"] - df["market_forecast"]

        # Check if signal was correct
        # Correct if: signal direction matches realized error direction
        df["correct_signal"] = (df["signal"] * df["realized_error"]) > 0

        # Hit rate (percentage of correct signals)
        signals_only = df[df["signal"] != 0]
        hit_rate = signals_only["correct_signal"].mean() if len(signals_only) > 0 else 0

        # Simulate PnL (simplified)
        # Assume price moves proportionally to forecast error
        # PnL = signal * position_size * (realized_error / market_forecast) * entry_price
        df["price_impact"] = (df["realized_error"] / df["market_forecast"]) * df["entry_price"]
        df["price_impact"] = df["price_impact"].fillna(0)

        df["gross_pnl"] = df["signal"] * df["position_size"] * df["price_impact"]
        df["net_pnl"] = df["gross_pnl"] - df["total_cost"]

        # Cumulative PnL
        df["cumulative_pnl"] = df["net_pnl"].cumsum()

        # Calculate metrics
        total_pnl = df["net_pnl"].sum()
        total_trades = (df["signal"] != 0).sum()

        # Sharpe ratio (annualized)
        returns = df["net_pnl"] / (df["position_size"] * df["entry_price"] + 1)
        returns = returns.replace([np.inf, -np.inf], 0)
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        # Win rate
        profitable_trades = (df["net_pnl"] > 0).sum()
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0

        # Average PnL per trade
        avg_pnl_per_trade = df[df["signal"] != 0]["net_pnl"].mean()

        # Maximum drawdown
        cumulative = df["cumulative_pnl"]
        running_max = cumulative.expanding().max()
        drawdown = cumulative - running_max
        max_drawdown = drawdown.min()

        # Information Coefficient (if available)
        ic = self.historical_ic if self.historical_ic is not None else np.nan

        metrics = {
            "total_pnl": total_pnl,
            "num_trades": total_trades,
            "hit_rate": hit_rate,
            "win_rate": win_rate,
            "sharpe_ratio": sharpe,
            "avg_pnl_per_trade": avg_pnl_per_trade,
            "max_drawdown": max_drawdown,
            "information_coefficient": ic,
            "avg_position_size": df[df["signal"] != 0]["position_size"].mean(),
            "total_transaction_costs": df["transaction_cost"].sum(),
            "total_slippage": df["slippage"].sum(),
        }

        return metrics

    def calculate_alpha_decay(
        self,
        our_forecasts: pd.Series,
        market_forecasts: pd.Series,
        actual_demand: pd.Series,
        max_horizon: int = 7,
    ) -> pd.DataFrame:
        """
        Calculate alpha decay: how long does forecast advantage persist?

        Alpha decay measures how quickly the information advantage
        from forecast errors dissipates over time.

        Args:
            our_forecasts: Our forecasts
            market_forecasts: Market forecasts
            actual_demand: Actual demand
            max_horizon: Maximum days to test

        Returns:
            DataFrame with IC at different horizons
        """
        forecast_errors = our_forecasts - market_forecasts
        actual_errors = actual_demand - market_forecasts

        results = []

        for horizon in range(0, max_horizon + 1):
            # Shift actual errors by horizon
            future_errors = actual_errors.shift(-horizon)

            # Calculate IC between current forecast error and future actual error
            common_idx = forecast_errors.index.intersection(future_errors.index)
            if len(common_idx) < 30:  # Need minimum data
                continue

            corr, p_value = stats.spearmanr(
                forecast_errors.loc[common_idx], future_errors.loc[common_idx]
            )

            results.append({"horizon": horizon, "ic": corr, "p_value": p_value})

        alpha_decay_df = pd.DataFrame(results)

        logger.info(
            f"Alpha decay calculated. IC at horizon 0: {alpha_decay_df.iloc[0]['ic']:.4f}, "
            f"IC at horizon {max_horizon}: {alpha_decay_df.iloc[-1]['ic']:.4f}"
        )

        return alpha_decay_df


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Generate synthetic data
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2024-01-01", freq="D")
    n = len(dates)

    # Actual demand
    actual_demand = pd.Series(
        4000 + 1000 * np.sin(np.arange(n) * 2 * np.pi / 365) + np.random.normal(0, 200, n),
        index=dates,
    )

    # Market forecasts (less accurate)
    market_forecasts = actual_demand + np.random.normal(0, 300, n)

    # Our forecasts (more accurate)
    our_forecasts = actual_demand + np.random.normal(0, 150, n)

    # Prices (correlated with demand)
    prices = pd.Series(50 + (actual_demand - 4000) / 50 + np.random.normal(0, 5, n), index=dates)

    # Split data
    split_idx = int(n * 0.7)
    train_actual = actual_demand.iloc[:split_idx]
    train_market = market_forecasts.iloc[:split_idx]
    train_ours = our_forecasts.iloc[:split_idx]
    train_prices = prices.iloc[:split_idx]

    test_actual = actual_demand.iloc[split_idx:]
    test_market = market_forecasts.iloc[split_idx:]
    test_ours = our_forecasts.iloc[split_idx:]
    test_prices = prices.iloc[split_idx:]

    # Initialize and calibrate strategy
    strategy = ForecastErrorArbitrageStrategy()
    strategy.calibrate(train_ours, train_market, train_actual, train_prices)

    # Generate signals on test set
    signals = strategy.generate_signals(test_ours, test_market, test_prices)

    # Calculate performance
    metrics = strategy.calculate_performance_metrics(signals, test_actual)

    print("\n" + "=" * 60)
    print("FORECAST ERROR ARBITRAGE - PERFORMANCE METRICS")
    print("=" * 60)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:30s}: {value:12.4f}")
        else:
            print(f"{key:30s}: {value}")

    # Alpha decay analysis
    alpha_decay = strategy.calculate_alpha_decay(test_ours, test_market, test_actual, max_horizon=7)
    print("\n" + "=" * 60)
    print("ALPHA DECAY ANALYSIS")
    print("=" * 60)
    print(alpha_decay.to_string(index=False))
