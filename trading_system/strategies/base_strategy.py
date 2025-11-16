"""Base Strategy Abstract Class

This module defines the abstract base class for all trading strategies,
providing a common interface and shared metric calculation methods.

Benefits:
- Reduces code duplication across strategies
- Ensures consistent metric calculations
- Enforces interface compliance
- Easier to add new strategies

Usage:
    class MyStrategy(BaseStrategy):
        def generate_signals(self, **kwargs) -> pd.DataFrame:
            # Implement strategy-specific logic
            ...
            return signals_df

        def get_strategy_name(self) -> str:
            return "MyStrategy"
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np
import pandas as pd


# Constants for metric calculations
EPSILON_MAPE = 1e-8  # Standard epsilon for MAPE calculation
HOURS_PER_YEAR = 365 * 24  # For annualizing hourly data
MIN_STD_FOR_SHARPE = 1e-10  # Minimum std to avoid division by zero


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies.

    All trading strategies should inherit from this class and implement:
    - generate_signals(): Core strategy logic
    - get_strategy_name(): Strategy identifier

    Provides standardized methods for:
    - Performance metrics calculation (Sharpe, Sortino, max DD)
    - Risk metrics (VaR, CVaR)
    - Transaction cost application
    """

    @abstractmethod
    def generate_signals(self, **kwargs) -> pd.DataFrame:
        """Generate trading signals.

        Must return DataFrame with at minimum:
        - 'signal': -1 (sell), 0 (neutral), 1 (buy)
        - 'position_size': Position size as fraction of capital

        Args:
            **kwargs: Strategy-specific parameters

        Returns:
            DataFrame with signals and positions
        """
        pass

    @abstractmethod
    def get_strategy_name(self) -> str:
        """Return strategy name for logging and reporting.

        Returns:
            Strategy name (e.g., "MeanReversion", "ForecastArbitrage")
        """
        pass

    def calculate_performance_metrics(
        self,
        signals_df: pd.DataFrame,
        prices: pd.Series,
        transaction_cost: float = 0.001,
    ) -> Dict[str, float]:
        """Calculate standardized performance metrics.

        Args:
            signals_df: DataFrame with 'signal' and 'position_size' columns
            prices: Price series
            transaction_cost: Transaction cost as fraction (default 0.1%)

        Returns:
            Dictionary with metrics:
            - sharpe_ratio: Annualized Sharpe ratio
            - sortino_ratio: Annualized Sortino ratio
            - max_drawdown: Maximum drawdown (%)
            - total_return: Total return (%)
            - annual_return: Annualized return (%)
            - win_rate: Percentage of profitable trades
            - profit_factor: Gross profit / gross loss
            - total_trades: Number of trades executed
            - avg_trade_duration: Average holding period (hours)
        """
        df = signals_df.copy()
        df["price"] = prices.values[: len(df)]

        # Calculate returns
        df["position"] = df["signal"] * df["position_size"]
        df["market_return"] = df["price"].pct_change()
        df["strategy_return"] = df["position"].shift(1) * df["market_return"]

        # Apply transaction costs
        df["trades"] = df["position"].diff().abs()
        df["transaction_costs"] = df["trades"] * transaction_cost
        df["net_return"] = df["strategy_return"] - df["transaction_costs"]

        # Cumulative returns
        df["cumulative_return"] = (1 + df["net_return"]).cumprod()

        # Total return
        total_return = df["cumulative_return"].iloc[-1] - 1

        # Annualized return (hourly data assumed)
        n_periods = len(df)
        annual_return = (1 + total_return) ** (HOURS_PER_YEAR / n_periods) - 1

        # Sharpe ratio (annualized for hourly data)
        returns = df["net_return"].dropna()
        mean_return = returns.mean()
        std_return = returns.std()

        if std_return > MIN_STD_FOR_SHARPE:
            sharpe = mean_return / std_return * np.sqrt(HOURS_PER_YEAR)
        else:
            sharpe = 0.0

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            downside_std = downside_returns.std()
            if downside_std > MIN_STD_FOR_SHARPE:
                sortino = mean_return / downside_std * np.sqrt(HOURS_PER_YEAR)
            else:
                sortino = 0.0
        else:
            sortino = sharpe  # No downside, use Sharpe

        # Maximum drawdown
        cumulative = df["cumulative_return"]
        running_max = cumulative.expanding().max()
        drawdown = (cumulative / running_max - 1) * 100
        max_drawdown = drawdown.min()

        # Win rate and profit factor
        trade_returns = df[df["trades"] > 0]["net_return"]
        if len(trade_returns) > 0:
            winning_trades = trade_returns[trade_returns > 0]
            losing_trades = trade_returns[trade_returns < 0]

            win_rate = len(winning_trades) / len(trade_returns) * 100

            gross_profit = winning_trades.sum() if len(winning_trades) > 0 else 0
            gross_loss = abs(losing_trades.sum()) if len(losing_trades) > 0 else 1e-10
            profit_factor = gross_profit / gross_loss
        else:
            win_rate = 0.0
            profit_factor = 0.0

        # Trade statistics
        total_trades = int(df["trades"].sum() / 2)  # Divide by 2 (entry + exit)

        # Average trade duration (in hours)
        position_changes = df["position"].diff().abs() > 0
        trade_starts = df.index[position_changes].tolist()
        trade_durations = []
        for i in range(len(trade_starts) - 1):
            duration = trade_starts[i + 1] - trade_starts[i]
            trade_durations.append(duration)

        avg_trade_duration = np.mean(trade_durations) if trade_durations else 0

        return {
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_drawdown,
            "total_return": total_return * 100,
            "annual_return": annual_return * 100,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "total_trades": total_trades,
            "avg_trade_duration": avg_trade_duration,
        }

    def calculate_risk_metrics(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95,
    ) -> Dict[str, float]:
        """Calculate risk metrics (VaR, CVaR).

        Args:
            returns: Series of returns
            confidence_level: Confidence level for VaR/CVaR (default 95%)

        Returns:
            Dictionary with:
            - var_95: Value at Risk at 95% confidence (negative = loss)
            - cvar_95: Conditional VaR (Expected Shortfall)
            - volatility: Annualized volatility
            - skewness: Return distribution skewness
            - kurtosis: Return distribution kurtosis (excess)
        """
        returns_clean = returns.dropna()

        # VaR (Value at Risk) - 5th percentile for 95% confidence
        alpha = 1 - confidence_level
        var = np.percentile(returns_clean, alpha * 100)

        # CVaR (Conditional VaR / Expected Shortfall)
        # Average of returns below VaR threshold
        cvar = returns_clean[returns_clean <= var].mean()

        # Volatility (annualized)
        volatility = returns_clean.std() * np.sqrt(HOURS_PER_YEAR)

        # Higher moments
        skewness = returns_clean.skew()
        kurtosis = returns_clean.kurtosis()  # Excess kurtosis

        return {
            "var_95": var * 100,  # As percentage
            "cvar_95": cvar * 100,
            "volatility_annual": volatility * 100,
            "skewness": skewness,
            "kurtosis": kurtosis,
        }

    def calculate_forecast_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        """Calculate forecast accuracy metrics.

        Standardized across all strategies for consistency.

        Args:
            y_true: Actual values
            y_pred: Predicted values

        Returns:
            Dictionary with RMSE, MAE, R², MAPE, DA
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # MAPE with consistent epsilon
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + EPSILON_MAPE))) * 100

        # Directional accuracy
        direction_actual = np.sign(np.diff(y_true))
        direction_pred = np.sign(np.diff(y_pred))
        da = np.mean(direction_actual == direction_pred) * 100

        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mape": mape,
            "directional_accuracy": da,
        }

    def backtest(
        self,
        prices: pd.Series,
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.001,
        **strategy_kwargs,
    ) -> Dict:
        """Run complete backtest with signals, execution, and metrics.

        Args:
            prices: Price series
            initial_capital: Starting capital (EUR)
            transaction_cost: Transaction cost fraction
            **strategy_kwargs: Strategy-specific parameters for generate_signals()

        Returns:
            Dictionary with:
            - signals: DataFrame with signals and positions
            - metrics: Performance metrics dictionary
            - risk_metrics: Risk metrics dictionary
            - equity_curve: Cumulative equity curve
        """
        # Generate signals
        signals = self.generate_signals(prices=prices, **strategy_kwargs)

        # Calculate metrics
        metrics = self.calculate_performance_metrics(
            signals_df=signals,
            prices=prices,
            transaction_cost=transaction_cost,
        )

        # Calculate equity curve
        df = signals.copy()
        df["price"] = prices.values[: len(df)]
        df["position"] = df["signal"] * df["position_size"]
        df["market_return"] = df["price"].pct_change()
        df["strategy_return"] = df["position"].shift(1) * df["market_return"]
        df["trades"] = df["position"].diff().abs()
        df["transaction_costs"] = df["trades"] * transaction_cost
        df["net_return"] = df["strategy_return"] - df["transaction_costs"]
        df["cumulative_return"] = (1 + df["net_return"]).cumprod()
        df["equity"] = initial_capital * df["cumulative_return"]

        equity_curve = df["equity"]

        # Risk metrics
        risk_metrics = self.calculate_risk_metrics(df["net_return"])

        return {
            "signals": signals,
            "metrics": metrics,
            "risk_metrics": risk_metrics,
            "equity_curve": equity_curve,
            "strategy_name": self.get_strategy_name(),
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__}(name='{self.get_strategy_name()}')"
