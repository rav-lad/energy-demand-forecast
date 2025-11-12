"""
Backtesting Engine for Energy Trading Strategies

This engine simulates trading strategies on historical data and calculates
performance metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade."""

    timestamp: pd.Timestamp
    action: str  # 'BUY' or 'SELL'
    quantity: float  # MWh
    price: float  # EUR/MWh
    region: str = "FR"
    trade_id: Optional[int] = None

    @property
    def value(self) -> float:
        """Total value of the trade."""
        return self.quantity * self.price


@dataclass
class Position:
    """Represents an open position."""

    entry_trade: Trade
    exit_trade: Optional[Trade] = None
    status: str = "OPEN"  # 'OPEN' or 'CLOSED'

    @property
    def pnl(self) -> float:
        """Profit and loss for this position."""
        if not self.exit_trade:
            return 0.0

        if self.entry_trade.action == "BUY":
            # Long position: profit = (exit_price - entry_price) * quantity
            return (
                self.exit_trade.price - self.entry_trade.price
            ) * self.entry_trade.quantity
        else:
            # Short position: profit = (entry_price - exit_price) * quantity
            return (
                self.entry_trade.price - self.exit_trade.price
            ) * self.entry_trade.quantity

    @property
    def return_pct(self) -> float:
        """Return percentage for this position."""
        if not self.exit_trade:
            return 0.0
        return (self.pnl / self.entry_trade.value) * 100

    @property
    def duration(self) -> pd.Timedelta:
        """Duration of the position."""
        if not self.exit_trade:
            return pd.Timedelta(0)
        return self.exit_trade.timestamp - self.entry_trade.timestamp


class BacktestEngine:
    """
    Backtesting engine for energy trading strategies.

    Attributes:
        initial_capital: Starting capital in EUR
        capital: Current available capital
        positions: List of all positions (open and closed)
        trades: List of all trades
        equity_curve: Time series of portfolio value
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
        max_position_size: float = 10000,
        max_positions: int = 5,
    ):
        """
        Initialize backtesting engine.

        Args:
            initial_capital: Initial capital in EUR
            transaction_cost: Transaction cost as fraction (e.g., 0.001 = 0.1%)
            slippage: Slippage as fraction
            max_position_size: Maximum position size in MWh
            max_positions: Maximum number of concurrent positions
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.max_position_size = max_position_size
        self.max_positions = max_positions

        # Trading records
        self.positions: List[Position] = []
        self.trades: List[Trade] = []
        self.equity_curve: pd.Series = pd.Series(dtype=float)

        # Metrics
        self.total_return = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0

        logger.info(
            f"Backtest engine initialized with capital: {initial_capital:,.2f} EUR"
        )

    @property
    def open_positions(self) -> List[Position]:
        """Get list of open positions."""
        return [p for p in self.positions if p.status == "OPEN"]

    @property
    def closed_positions(self) -> List[Position]:
        """Get list of closed positions."""
        return [p for p in self.positions if p.status == "CLOSED"]

    @property
    def current_exposure(self) -> float:
        """Total exposure from open positions."""
        return sum(p.entry_trade.value for p in self.open_positions)

    def can_open_position(self, quantity: float, price: float) -> bool:
        """
        Check if a new position can be opened.

        Args:
            quantity: Position size in MWh
            price: Price in EUR/MWh

        Returns:
            True if position can be opened
        """
        # Check max positions limit
        if len(self.open_positions) >= self.max_positions:
            logger.debug("Maximum positions limit reached")
            return False

        # Check max position size
        if quantity > self.max_position_size:
            logger.debug(
                f"Position size {quantity} exceeds maximum {self.max_position_size}"
            )
            return False

        # Check available capital
        required_capital = (
            quantity * price * (1 + self.transaction_cost + self.slippage)
        )
        if required_capital > self.capital:
            logger.debug(
                f"Insufficient capital: need {required_capital:,.2f}, have {self.capital:,.2f}"
            )
            return False

        return True

    def execute_trade(
        self,
        timestamp: pd.Timestamp,
        action: str,
        quantity: float,
        price: float,
        region: str = "FR",
    ) -> Optional[Trade]:
        """
        Execute a trade.

        Args:
            timestamp: Trade timestamp
            action: 'BUY' or 'SELL'
            quantity: Quantity in MWh
            price: Price in EUR/MWh
            region: Trading region

        Returns:
            Trade object if successful, None otherwise
        """
        # Apply slippage
        effective_price = (
            price * (1 + self.slippage)
            if action == "BUY"
            else price * (1 - self.slippage)
        )

        # Check if trade is feasible
        if action == "BUY" and not self.can_open_position(quantity, effective_price):
            return None

        # Create trade
        trade = Trade(
            timestamp=timestamp,
            action=action,
            quantity=quantity,
            price=effective_price,
            region=region,
            trade_id=len(self.trades),
        )

        # Calculate costs
        transaction_value = trade.value
        costs = transaction_value * self.transaction_cost

        # Update capital
        if action == "BUY":
            self.capital -= transaction_value + costs
            # Open new position
            position = Position(entry_trade=trade)
            self.positions.append(position)
            logger.debug(
                f"Opened BUY position: {quantity} MWh @ {effective_price:.2f} EUR/MWh"
            )

        else:  # SELL
            self.capital += transaction_value - costs
            # Close matching open position (FIFO)
            open_buy_positions = [
                p for p in self.open_positions if p.entry_trade.action == "BUY"
            ]
            if open_buy_positions:
                position = open_buy_positions[0]
                position.exit_trade = trade
                position.status = "CLOSED"

                # Update metrics
                if position.pnl > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1

                logger.debug(
                    f"Closed position: PnL = {position.pnl:.2f} EUR ({position.return_pct:.2f}%)"
                )

        self.trades.append(trade)
        self.total_trades += 1

        return trade

    def process_signals(
        self,
        data: pd.DataFrame,
        signals: pd.Series,
        price_column: str = "price_FR",
        quantity_column: Optional[str] = None,
    ):
        """
        Process trading signals on historical data.

        Args:
            data: DataFrame with historical price data
            signals: Series with trading signals (1 = BUY, -1 = SELL, 0 = HOLD)
            price_column: Column name for prices
            quantity_column: Column name for position sizes (optional)
        """
        logger.info(f"Processing signals for {len(data)} periods")

        equity = []

        for timestamp, row in data.iterrows():
            signal = signals.loc[timestamp] if timestamp in signals.index else 0
            price = row[price_column]

            # Determine position size
            if quantity_column and quantity_column in row:
                quantity = row[quantity_column]
            else:
                quantity = min(1000, self.max_position_size)  # Default 1000 MWh

            # Execute trade based on signal
            if signal == 1:  # BUY signal
                if len(self.open_positions) == 0:  # Only open if no open positions
                    self.execute_trade(timestamp, "BUY", quantity, price)

            elif signal == -1:  # SELL signal
                if len(self.open_positions) > 0:  # Only close if have open positions
                    self.execute_trade(timestamp, "SELL", quantity, price)

            # Calculate current equity
            current_equity = self.capital
            for pos in self.open_positions:
                # Mark-to-market for open positions
                unrealized_pnl = (
                    price - pos.entry_trade.price
                ) * pos.entry_trade.quantity
                current_equity += pos.entry_trade.value + unrealized_pnl

            equity.append(current_equity)

        # Store equity curve
        self.equity_curve = pd.Series(equity, index=data.index)
        self.total_return = (
            (self.equity_curve.iloc[-1] - self.initial_capital)
            / self.initial_capital
            * 100
        )

        logger.info(
            f"Backtest completed: {self.total_trades} trades, {self.total_return:.2f}% return"
        )

    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics.

        Returns:
            Dictionary of performance metrics
        """
        if len(self.closed_positions) == 0:
            logger.warning("No closed positions to calculate metrics")
            return {}

        # Returns
        returns = self.equity_curve.pct_change().dropna()

        # Basic metrics
        total_return = self.total_return
        final_capital = self.equity_curve.iloc[-1]

        # Win rate
        win_rate = (
            self.winning_trades / len(self.closed_positions) * 100
            if self.closed_positions
            else 0
        )

        # Profit factor
        gross_profit = sum(p.pnl for p in self.closed_positions if p.pnl > 0)
        gross_loss = abs(sum(p.pnl for p in self.closed_positions if p.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

        # Sharpe ratio (assuming 252 trading days per year)
        sharpe_ratio = (
            returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        )

        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        sortino_ratio = (
            returns.mean() / downside_returns.std() * np.sqrt(252)
            if len(downside_returns) > 0 and downside_returns.std() > 0
            else 0
        )

        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100

        # Average trade metrics
        avg_trade_pnl = np.mean([p.pnl for p in self.closed_positions])
        avg_win = (
            np.mean([p.pnl for p in self.closed_positions if p.pnl > 0])
            if self.winning_trades > 0
            else 0
        )
        avg_loss = (
            np.mean([p.pnl for p in self.closed_positions if p.pnl < 0])
            if self.losing_trades > 0
            else 0
        )

        # Average trade duration
        avg_duration = np.mean(
            [p.duration.total_seconds() / 3600 for p in self.closed_positions]
        )  # hours

        # Calmar ratio (return / max drawdown)
        calmar_ratio = abs(total_return / max_drawdown) if max_drawdown != 0 else 0

        metrics = {
            "initial_capital": self.initial_capital,
            "final_capital": final_capital,
            "total_return": total_return,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "calmar_ratio": calmar_ratio,
            "avg_trade_pnl": avg_trade_pnl,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "avg_trade_duration_hours": avg_duration,
            "gross_profit": gross_profit,
            "gross_loss": gross_loss,
        }

        return metrics

    def print_results(self):
        """Print backtest results in a formatted table."""
        metrics = self.calculate_metrics()

        if not metrics:
            print("No results to display")
            return

        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)

        print("\nCapital:")
        print(f"  Initial Capital:     {metrics['initial_capital']:>15,.2f} EUR")
        print(f"  Final Capital:       {metrics['final_capital']:>15,.2f} EUR")
        print(f"  Total Return:        {metrics['total_return']:>15.2f} %")

        print("\nTrade Statistics:")
        print(f"  Total Trades:        {metrics['total_trades']:>15}")
        print(f"  Winning Trades:      {metrics['winning_trades']:>15}")
        print(f"  Losing Trades:       {metrics['losing_trades']:>15}")
        print(f"  Win Rate:            {metrics['win_rate']:>15.2f} %")
        print(f"  Profit Factor:       {metrics['profit_factor']:>15.2f}")

        print("\nRisk Metrics:")
        print(f"  Sharpe Ratio:        {metrics['sharpe_ratio']:>15.2f}")
        print(f"  Sortino Ratio:       {metrics['sortino_ratio']:>15.2f}")
        print(f"  Max Drawdown:        {metrics['max_drawdown']:>15.2f} %")
        print(f"  Calmar Ratio:        {metrics['calmar_ratio']:>15.2f}")

        print("\nTrade Performance:")
        print(f"  Avg Trade PnL:       {metrics['avg_trade_pnl']:>15,.2f} EUR")
        print(f"  Avg Win:             {metrics['avg_win']:>15,.2f} EUR")
        print(f"  Avg Loss:            {metrics['avg_loss']:>15,.2f} EUR")
        print(
            f"  Avg Duration:        {metrics['avg_trade_duration_hours']:>15.1f} hours"
        )

        print("\nGross Figures:")
        print(f"  Gross Profit:        {metrics['gross_profit']:>15,.2f} EUR")
        print(f"  Gross Loss:          {metrics['gross_loss']:>15,.2f} EUR")

        print("=" * 60 + "\n")


if __name__ == "__main__":
    # Example usage
    print("Backtesting Engine - Example")

    # Create sample data
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    np.random.seed(42)
    prices = 50 + np.random.randn(len(dates)).cumsum() * 2

    data = pd.DataFrame({"price_FR": prices}, index=dates)

    # Create simple signals (buy when price < 45, sell when price > 55)
    signals = pd.Series(0, index=dates)
    signals[prices < 45] = 1
    signals[prices > 55] = -1

    # Run backtest
    engine = BacktestEngine(initial_capital=100000)
    engine.process_signals(data, signals, price_column="price_FR")

    # Print results
    engine.print_results()
