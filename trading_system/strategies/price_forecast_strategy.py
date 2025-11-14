"""Price forecast-based trading strategy.

Trades the spread between ML price forecasts and market prices.
Simple but effective: if forecast > market + threshold → BUY
                      if forecast < market - threshold → SELL
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd


class PriceForecastStrategy:
    """Trade based on price forecast deviations from market.

    This strategy monetizes forecast accuracy by trading when the model's
    prediction significantly differs from current market price.

    Signal generation:
    - BUY when: forecast - market_price > entry_threshold
    - SELL when: forecast - market_price < -entry_threshold
    - EXIT when: abs(forecast - market_price) < exit_threshold

    Parameters control risk-reward trade-off:
    - Higher entry_threshold → fewer trades, higher conviction
    - Lower exit_threshold → quick profit-taking
    """

    def __init__(
        self,
        entry_threshold: float = 5.0,
        exit_threshold: float = 2.0,
        position_size: float = 0.02,
        max_position: float = 0.10,
        confidence_scaling: bool = True,
        max_holding_period: int = 72,
    ):
        """Initialize price forecast strategy.

        Args:
            entry_threshold: Minimum forecast deviation for entry (EUR/MWh).
            exit_threshold: Forecast deviation for exit (EUR/MWh).
            position_size: Base position size (fraction of capital).
            max_position: Maximum position size (fraction of capital).
            confidence_scaling: Scale position by forecast confidence.
            max_holding_period: Maximum hours to hold a position (default 72).
        """
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.position_size = position_size
        self.max_position = max_position
        self.confidence_scaling = confidence_scaling
        self.max_holding_period = max_holding_period

    def generate_signals(
        self,
        market_prices: pd.Series,
        forecasts: pd.Series,
        forecast_std: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Generate trading signals from price forecasts.

        Args:
            market_prices: Current market prices (EUR/MWh).
            forecasts: ML price forecasts (EUR/MWh).
            forecast_std: Optional forecast uncertainty (std dev).

        Returns:
            pd.DataFrame: Signals with columns:
                - signal: -1 (short), 0 (flat), +1 (long)
                - position_size: Recommended position size
                - forecast_error: forecast - market_price
                - confidence: Signal confidence (0-1)
        """
        df = pd.DataFrame(
            {"market_price": market_prices, "forecast": forecasts}, index=forecasts.index
        )

        # Calculate forecast error
        df["forecast_error"] = df["forecast"] - df["market_price"]

        # Calculate confidence (inverse of uncertainty)
        if forecast_std is not None:
            # Higher std → lower confidence
            df["forecast_std"] = forecast_std
            df["confidence"] = 1 / (1 + forecast_std / df["market_price"])
        else:
            # Default confidence based on error magnitude
            df["confidence"] = np.tanh(
                np.abs(df["forecast_error"]) / self.entry_threshold
            )

        # Initialize signals
        df["raw_signal"] = 0

        # Entry signals
        buy_signal = df["forecast_error"] > self.entry_threshold
        sell_signal = df["forecast_error"] < -self.entry_threshold

        df.loc[buy_signal, "raw_signal"] = 1
        df.loc[sell_signal, "raw_signal"] = -1

        # Apply position persistence (hold until exit threshold or max holding period)
        df["signal"] = df["raw_signal"]
        position = 0
        position_age = 0

        for i in range(len(df)):
            if df["raw_signal"].iloc[i] != 0:
                # New signal
                position = df["raw_signal"].iloc[i]
                position_age = 0
            elif position != 0:
                # Holding position
                position_age += 1

                # Exit if forecast converged to market
                if abs(df["forecast_error"].iloc[i]) < self.exit_threshold:
                    position = 0
                    position_age = 0
                # Exit if max holding period exceeded
                elif position_age >= self.max_holding_period:
                    position = 0
                    position_age = 0

            df["signal"].iloc[i] = position

        # Calculate position sizing
        if self.confidence_scaling:
            # Scale by confidence
            df["position_size"] = (
                df["confidence"] * self.position_size * np.sign(df["signal"])
            )
        else:
            # Fixed sizing
            df["position_size"] = self.position_size * np.sign(df["signal"])

        # Cap at maximum position
        df["position_size"] = np.clip(df["position_size"], -self.max_position, self.max_position)

        return df[["signal", "position_size", "forecast_error", "confidence"]]

    def backtest(
        self,
        market_prices: pd.Series,
        forecasts: pd.Series,
        forecast_std: Optional[pd.Series] = None,
        transaction_cost: float = 0.001,
    ) -> Dict[str, float]:
        """Backtest strategy performance.

        Args:
            market_prices: Historical market prices.
            forecasts: Historical price forecasts.
            forecast_std: Optional forecast uncertainty.
            transaction_cost: Transaction cost (fraction of trade value).

        Returns:
            dict: Performance metrics.
        """
        # Generate signals
        signals = self.generate_signals(market_prices, forecasts, forecast_std)

        # Calculate returns
        price_returns = market_prices.pct_change()
        strategy_returns = signals["position_size"].shift(1) * price_returns

        # Transaction costs
        position_changes = signals["position_size"].diff().abs()
        costs = position_changes * transaction_cost
        net_returns = strategy_returns - costs

        # Performance metrics
        total_return = (1 + net_returns).prod() - 1
        annual_return = (1 + total_return) ** (365 * 24 / len(net_returns)) - 1

        sharpe = (
            net_returns.mean() / net_returns.std() * np.sqrt(365 * 24)
            if net_returns.std() > 0
            else 0
        )

        # Drawdown
        cumulative = (1 + net_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Trade statistics
        trades = (signals["signal"].diff() != 0).sum()
        winning_trades = (net_returns > 0).sum()
        win_rate = winning_trades / trades if trades > 0 else 0

        metrics = {
            "total_return": total_return,
            "annual_return": annual_return,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "num_trades": int(trades),
            "win_rate": win_rate,
            "avg_return_per_trade": net_returns.mean(),
            "forecast_accuracy": (
                (np.sign(signals["forecast_error"]) == np.sign(price_returns)).mean()
            ),
        }

        return metrics


class RegimeAdaptiveStrategy(PriceForecastStrategy):
    """Price forecast strategy with regime-dependent parameters.

    Adapts entry/exit thresholds based on market regime:
    - Scarcity regime: Wider thresholds (more volatile)
    - Renewable flush: Tighter thresholds (more predictable)
    - Base load: Default thresholds
    """

    def __init__(
        self,
        base_entry_threshold: float = 5.0,
        base_exit_threshold: float = 2.0,
        position_size: float = 0.02,
        max_position: float = 0.10,
        regime_multipliers: Optional[Dict[str, float]] = None,
        max_holding_period: int = 72,
    ):
        """Initialize regime-adaptive strategy.

        Args:
            base_entry_threshold: Base entry threshold (EUR/MWh).
            base_exit_threshold: Base exit threshold (EUR/MWh).
            position_size: Base position size.
            max_position: Maximum position size.
            regime_multipliers: Threshold multipliers by regime.
            max_holding_period: Maximum hours to hold a position (default 72).
        """
        super().__init__(
            entry_threshold=base_entry_threshold,
            exit_threshold=base_exit_threshold,
            position_size=position_size,
            max_position=max_position,
            max_holding_period=max_holding_period,
        )

        self.base_entry_threshold = base_entry_threshold
        self.base_exit_threshold = base_exit_threshold

        # Default regime multipliers
        if regime_multipliers is None:
            regime_multipliers = {
                "base_load": 1.0,  # Default
                "renewable_flush": 0.7,  # Tighter (more predictable)
                "scarcity": 1.5,  # Wider (more volatile)
                "high_volatility": 1.3,  # Wider (less certain)
            }

        self.regime_multipliers = regime_multipliers

    def generate_signals(
        self,
        market_prices: pd.Series,
        forecasts: pd.Series,
        regimes: pd.Series,
        forecast_std: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Generate regime-adaptive signals.

        Args:
            market_prices: Market prices.
            forecasts: Price forecasts.
            regimes: Market regime classifications.
            forecast_std: Optional forecast uncertainty.

        Returns:
            pd.DataFrame: Trading signals with regime-adaptive thresholds.
        """
        df = pd.DataFrame(
            {
                "market_price": market_prices,
                "forecast": forecasts,
                "regime": regimes,
            }
        )

        # Adaptive thresholds by regime
        df["entry_threshold"] = df["regime"].map(self.regime_multipliers).fillna(
            1.0
        ) * self.base_entry_threshold
        df["exit_threshold"] = df["regime"].map(self.regime_multipliers).fillna(
            1.0
        ) * self.base_exit_threshold

        # Calculate signals row by row with adaptive thresholds
        signals_list = []

        for i in range(len(df)):
            # Temporarily set instance thresholds
            self.entry_threshold = df["entry_threshold"].iloc[i]
            self.exit_threshold = df["exit_threshold"].iloc[i]

            # Generate signal for this timestamp
            signal_df = super().generate_signals(
                market_prices.iloc[i : i + 1],
                forecasts.iloc[i : i + 1],
                forecast_std.iloc[i : i + 1] if forecast_std is not None else None,
            )

            signals_list.append(signal_df.iloc[0])

        # Reset thresholds
        self.entry_threshold = self.base_entry_threshold
        self.exit_threshold = self.base_exit_threshold

        return pd.DataFrame(signals_list, index=df.index)
