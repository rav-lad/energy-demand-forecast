"""
Mean Reversion Trading Strategy for Electricity Prices

This module implements a statistical arbitrage strategy based on mean reversion
in electricity prices. The strategy assumes that electricity prices revert to
their mean over time, creating trading opportunities when prices deviate
significantly from equilibrium.

Author: Quant Research Team
Date: 2024-11-12
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class MeanReversionConfig:
    """Configuration for Mean Reversion Strategy."""

    lookback_window: int = 30  # Days for rolling mean/std calculation
    entry_z_score: float = 2.0  # Z-score threshold for entry
    exit_z_score: float = 0.5  # Z-score threshold for exit
    half_life_window: int = 90  # Days for half-life estimation
    min_half_life: float = 1.0  # Minimum acceptable half-life (days)
    max_half_life: float = 30.0  # Maximum acceptable half-life (days)
    position_size: float = 1.0  # Base position size
    stop_loss_z: float = 3.5  # Stop-loss Z-score threshold


class MeanReversionStrategy:
    """
    Mean Reversion Strategy for Electricity Prices.

    The strategy identifies overbought/oversold conditions using Z-scores
    and takes contrarian positions expecting price reversion to mean.

    Key Features:
    - Dynamic half-life estimation (Ornstein-Uhlenbeck process)
    - Adaptive position sizing based on mean reversion strength
    - Z-score based entry/exit signals
    - Stop-loss protection

    Mathematical Framework:
    - Price follows OU process: dX_t = θ(μ - X_t)dt + σdW_t
    - Half-life: τ = ln(2)/θ
    - Z-score: (X_t - μ) / σ

    Example:
        >>> config = MeanReversionConfig(entry_z_score=2.0)
        >>> strategy = MeanReversionStrategy(config)
        >>> signals = strategy.generate_signals(prices_df)
    """

    def __init__(self, config: Optional[MeanReversionConfig] = None):
        """
        Initialize Mean Reversion Strategy.

        Args:
            config: Strategy configuration. Uses defaults if None.
        """
        self.config = config or MeanReversionConfig()
        self.is_calibrated = False
        self.half_life = None
        self.mean_reversion_speed = None

        logger.info(f"Initialized Mean Reversion Strategy with config: {self.config}")

    def calculate_half_life(self, prices: pd.Series, method: str = "ols") -> Tuple[float, float]:
        """
        Calculate half-life of mean reversion using Ornstein-Uhlenbeck process.

        The half-life τ represents the time it takes for a price deviation
        to decay to half its original size.

        Model: ΔX_t = α + β*X_{t-1} + ε_t
        Where β = -θ (mean reversion speed)
        Half-life: τ = ln(2) / θ

        Args:
            prices: Price time series
            method: Estimation method ('ols' or 'mle')

        Returns:
            (half_life, mean_reversion_speed): Tuple of half-life and theta

        Raises:
            ValueError: If prices are insufficient or method is invalid
        """
        if len(prices) < self.config.half_life_window:
            raise ValueError(
                f"Insufficient data for half-life calculation. "
                f"Need {self.config.half_life_window}, got {len(prices)}"
            )

        # Use last N days
        prices_recent = prices.iloc[-self.config.half_life_window :]

        if method == "ols":
            # OLS regression: ΔX_t = α + β*X_{t-1} + ε
            lag_prices = prices_recent.shift(1).dropna()
            price_changes = prices_recent.diff().dropna()

            # Align indices
            common_idx = lag_prices.index.intersection(price_changes.index)
            X = lag_prices.loc[common_idx].values
            y = price_changes.loc[common_idx].values

            # Add intercept
            X_with_const = np.column_stack([np.ones_like(X), X])

            # OLS: β = (X'X)^(-1)X'y
            try:
                beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
                mean_reversion_speed = -beta[1]  # θ = -β
            except np.linalg.LinAlgError:
                logger.warning("OLS failed, using default half-life")
                return 10.0, 0.0693  # Default: 10 days half-life

        elif method == "mle":
            # Maximum Likelihood Estimation (more robust but slower)
            # Not implemented yet, fallback to OLS
            logger.warning("MLE not implemented, using OLS")
            return self.calculate_half_life(prices, method="ols")
        else:
            raise ValueError(f"Unknown method: {method}. Use 'ols' or 'mle'")

        # Calculate half-life
        if mean_reversion_speed > 0:
            half_life = np.log(2) / mean_reversion_speed
        else:
            logger.warning(
                f"Negative mean reversion speed: {mean_reversion_speed}. "
                f"Prices may not be mean-reverting."
            )
            half_life = np.inf

        # Clip to reasonable range
        half_life = np.clip(half_life, self.config.min_half_life, self.config.max_half_life)

        logger.info(
            f"Half-life: {half_life:.2f} days, " f"Mean reversion speed: {mean_reversion_speed:.4f}"
        )

        return half_life, mean_reversion_speed

    def calibrate(self, prices: pd.Series) -> None:
        """
        Calibrate strategy parameters on historical data.

        Estimates:
        - Half-life of mean reversion
        - Mean reversion speed (theta)

        Args:
            prices: Historical price series for calibration
        """
        logger.info(f"Calibrating strategy on {len(prices)} data points...")

        # Calculate half-life
        self.half_life, self.mean_reversion_speed = self.calculate_half_life(prices)

        self.is_calibrated = True

        logger.info(
            f"Calibration complete. Half-life: {self.half_life:.2f} days, "
            f"θ: {self.mean_reversion_speed:.4f}"
        )

    def generate_signals(
        self, prices: pd.Series, additional_features: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate trading signals based on Z-scores.

        Signal Logic:
        - BUY (1): Z-score < -entry_z_score (price significantly below mean)
        - SELL (-1): Z-score > entry_z_score (price significantly above mean)
        - EXIT (0): |Z-score| < exit_z_score (price near mean)
        - STOP-LOSS: |Z-score| > stop_loss_z (extreme deviation)

        Args:
            prices: Price time series
            additional_features: Optional features for position sizing

        Returns:
            DataFrame with columns:
                - price: Original prices
                - rolling_mean: Rolling mean
                - rolling_std: Rolling standard deviation
                - z_score: Z-score values
                - signal: Trading signal (-1, 0, 1)
                - position: Actual position after signal logic
                - position_size: Size of position
        """
        if not self.is_calibrated:
            logger.warning("Strategy not calibrated. Running calibration first...")
            self.calibrate(prices)

        df = pd.DataFrame({"price": prices})

        # Calculate rolling statistics
        df["rolling_mean"] = (
            df["price"].rolling(window=self.config.lookback_window, min_periods=1).mean()
        )

        df["rolling_std"] = (
            df["price"].rolling(window=self.config.lookback_window, min_periods=1).std()
        )

        # Calculate Z-score
        df["z_score"] = (df["price"] - df["rolling_mean"]) / df["rolling_std"]
        df["z_score"] = df["z_score"].fillna(0)

        # Generate raw signals
        df["raw_signal"] = 0

        # Entry signals
        df.loc[df["z_score"] < -self.config.entry_z_score, "raw_signal"] = 1  # BUY
        df.loc[df["z_score"] > self.config.entry_z_score, "raw_signal"] = -1  # SELL

        # Exit signals (flatten position when near mean)
        df.loc[
            (df["z_score"].abs() < self.config.exit_z_score) & (df["raw_signal"] == 0), "raw_signal"
        ] = 0

        # Stop-loss (force exit on extreme moves)
        df["stop_loss"] = df["z_score"].abs() > self.config.stop_loss_z

        # Convert signals to positions (with state management)
        df["signal"] = 0
        df["position"] = 0

        current_position = 0
        for i in range(len(df)):
            raw_sig = df["raw_signal"].iloc[i]
            stop_loss = df["stop_loss"].iloc[i]
            z_score = df["z_score"].iloc[i]

            # Stop-loss: force exit
            if stop_loss:
                current_position = 0
                df["signal"].iloc[i] = -np.sign(current_position) if current_position != 0 else 0

            # Exit when Z-score crosses zero (mean)
            elif abs(z_score) < self.config.exit_z_score and current_position != 0:
                df["signal"].iloc[i] = -np.sign(current_position)  # Exit signal
                current_position = 0

            # Entry signals
            elif raw_sig != 0 and current_position == 0:
                df["signal"].iloc[i] = raw_sig
                current_position = raw_sig

            # Hold position
            else:
                df["signal"].iloc[i] = 0

            df["position"].iloc[i] = current_position

        # Position sizing based on mean reversion strength
        # Stronger mean reversion (lower half-life) → larger positions
        if self.half_life is not None and self.half_life < np.inf:
            mean_reversion_strength = 1.0 / self.half_life
            size_multiplier = np.clip(mean_reversion_strength * 10, 0.5, 2.0)
        else:
            size_multiplier = 1.0

        df["position_size"] = df["position"].abs() * self.config.position_size * size_multiplier

        logger.info(
            f"Generated signals: {(df['signal'] == 1).sum()} buys, "
            f"{(df['signal'] == -1).sum()} sells"
        )

        return df

    def calculate_performance_metrics(
        self, signals_df: pd.DataFrame, transaction_cost: float = 0.001
    ) -> Dict[str, float]:
        """
        Calculate strategy performance metrics.

        Args:
            signals_df: DataFrame from generate_signals()
            transaction_cost: Transaction cost per trade (default 0.1%)

        Returns:
            Dictionary with performance metrics:
                - total_return: Total return
                - sharpe_ratio: Annualized Sharpe ratio
                - sortino_ratio: Annualized Sortino ratio
                - max_drawdown: Maximum drawdown
                - win_rate: Percentage of profitable trades
                - avg_holding_period: Average days per trade
                - turnover: Annual turnover rate
        """
        df = signals_df.copy()

        # Calculate returns
        df["price_return"] = df["price"].pct_change()
        df["strategy_return"] = df["position"].shift(1) * df["price_return"]

        # Apply transaction costs
        df["trades"] = df["signal"].abs()
        df["transaction_costs"] = df["trades"] * transaction_cost
        df["net_return"] = df["strategy_return"] - df["transaction_costs"]

        # Cumulative returns
        df["cumulative_return"] = (1 + df["net_return"]).cumprod()

        # Calculate metrics
        total_return = df["cumulative_return"].iloc[-1] - 1

        # Sharpe ratio (annualized, assuming daily data)
        returns = df["net_return"].dropna()
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else returns.std()
        sortino = returns.mean() / downside_std * np.sqrt(252) if downside_std > 0 else 0

        # Maximum drawdown
        cumulative = df["cumulative_return"]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Win rate
        profitable_trades = (returns > 0).sum()
        total_trades = (df["trades"] > 0).sum()
        win_rate = profitable_trades / total_trades if total_trades > 0 else 0

        # Average holding period
        position_changes = df["position"].diff().abs() > 0
        num_position_changes = position_changes.sum()
        avg_holding_period = len(df) / num_position_changes if num_position_changes > 0 else 0

        # Turnover (annualized)
        turnover = (df["trades"].sum() / len(df)) * 252

        metrics = {
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "avg_holding_period": avg_holding_period,
            "turnover": turnover,
            "num_trades": total_trades,
            "half_life": self.half_life if self.half_life is not None else np.nan,
        }

        return metrics


def optimize_parameters(
    prices: pd.Series, param_grid: Optional[Dict] = None
) -> Tuple[MeanReversionConfig, Dict[str, float]]:
    """
    Optimize strategy parameters using grid search.

    Args:
        prices: Historical price series
        param_grid: Parameter grid for optimization. If None, uses defaults.

    Returns:
        (best_config, best_metrics): Tuple of optimal config and its metrics
    """
    if param_grid is None:
        param_grid = {
            "entry_z_score": [1.5, 2.0, 2.5],
            "exit_z_score": [0.3, 0.5, 0.7],
            "lookback_window": [20, 30, 40],
        }

    best_sharpe = -np.inf
    best_config = None
    best_metrics = None

    # Split data: 70% train, 30% test
    split_idx = int(len(prices) * 0.7)
    train_prices = prices.iloc[:split_idx]
    test_prices = prices.iloc[split_idx:]

    logger.info(f"Optimizing parameters on {len(train_prices)} training samples...")

    # Grid search
    from itertools import product

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    for params in product(*param_values):
        param_dict = dict(zip(param_names, params))

        # Create config
        config = MeanReversionConfig(**param_dict)

        # Train strategy
        strategy = MeanReversionStrategy(config)
        strategy.calibrate(train_prices)

        # Test on out-of-sample data
        signals = strategy.generate_signals(test_prices)
        metrics = strategy.calculate_performance_metrics(signals)

        # Select based on Sharpe ratio
        if metrics["sharpe_ratio"] > best_sharpe:
            best_sharpe = metrics["sharpe_ratio"]
            best_config = config
            best_metrics = metrics

    logger.info(f"Optimization complete. Best Sharpe: {best_sharpe:.3f}, " f"Config: {best_config}")

    return best_config, best_metrics


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Generate synthetic price data (replace with real data)
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2024-01-01", freq="D")

    # Mean-reverting process
    prices = pd.Series(50 + np.cumsum(np.random.randn(len(dates)) * 0.5), index=dates)
    prices = prices + 10 * np.sin(np.arange(len(dates)) * 2 * np.pi / 365)  # Seasonality

    # Initialize and calibrate strategy
    strategy = MeanReversionStrategy()
    strategy.calibrate(prices)

    # Generate signals
    signals = strategy.generate_signals(prices)

    # Calculate performance
    metrics = strategy.calculate_performance_metrics(signals)

    print("\n" + "=" * 60)
    print("MEAN REVERSION STRATEGY - PERFORMANCE METRICS")
    print("=" * 60)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:25s}: {value:10.4f}")
        else:
            print(f"{key:25s}: {value}")

    print(f"\nSignals generated: {len(signals)}")
    print(f"Buy signals: {(signals['signal'] == 1).sum()}")
    print(f"Sell signals: {(signals['signal'] == -1).sum()}")
