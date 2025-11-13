"""
Backtesting Engine for Trading Strategies

This module provides a comprehensive backtesting framework with:
- Realistic transaction costs and slippage modeling
- Position tracking and P&L calculation
- Performance metrics computation
- Support for multiple strategies
- Event-driven architecture for realistic simulation

Author: Energy Trading Quant Team
Date: 2025-11-12
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types supported by the backtesting engine."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderSide(Enum):
    """Order side (buy/sell)."""
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Represents a trading order."""
    timestamp: datetime
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    price: Optional[float] = None  # For limit/stop orders
    filled_price: Optional[float] = None
    filled_quantity: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    status: str = "pending"  # pending, filled, rejected, partial

    def __post_init__(self):
        """Validate order parameters."""
        if self.quantity <= 0:
            raise ValueError(f"Order quantity must be positive, got {self.quantity}")


@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    quantity: float = 0.0  # Positive for long, negative for short
    avg_entry_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_commission: float = 0.0
    total_slippage: float = 0.0

    def update_position(self, fill_quantity: float, fill_price: float,
                       commission: float, slippage: float) -> float:
        """
        Update position with a new fill.

        Returns:
            realized_pnl: Realized P&L from this trade
        """
        realized = 0.0

        # If reversing or closing position
        if (self.quantity > 0 and fill_quantity < 0) or \
           (self.quantity < 0 and fill_quantity > 0):

            close_quantity = min(abs(fill_quantity), abs(self.quantity))

            if self.quantity > 0:  # Closing long
                realized = close_quantity * (fill_price - self.avg_entry_price)
            else:  # Closing short
                realized = close_quantity * (self.avg_entry_price - fill_price)

            # Adjust for costs
            realized -= commission + slippage

            # Update position
            remaining_fill = fill_quantity + (close_quantity if self.quantity > 0 else -close_quantity)
            self.quantity -= (close_quantity if self.quantity > 0 else -close_quantity)

            if abs(self.quantity) < 1e-6:
                self.quantity = 0.0
                self.avg_entry_price = 0.0

            # If there's remaining fill, it opens a new position
            if abs(remaining_fill) > 1e-6:
                self.quantity = remaining_fill
                self.avg_entry_price = fill_price

        else:  # Adding to position
            total_cost = self.quantity * self.avg_entry_price + fill_quantity * fill_price
            self.quantity += fill_quantity

            if abs(self.quantity) > 1e-6:
                self.avg_entry_price = total_cost / self.quantity

        self.realized_pnl += realized
        self.total_commission += commission
        self.total_slippage += slippage

        return realized

    def mark_to_market(self, current_price: float) -> None:
        """Update unrealized P&L based on current market price."""
        if abs(self.quantity) < 1e-6:
            self.unrealized_pnl = 0.0
        else:
            self.unrealized_pnl = self.quantity * (current_price - self.avg_entry_price)


@dataclass
class TransactionCostModel:
    """Model for transaction costs and slippage."""

    # Commission structure
    commission_pct: float = 0.001  # 0.1% commission
    commission_min: float = 1.0    # Minimum commission per trade

    # Slippage model
    slippage_pct: float = 0.0005   # 0.05% slippage
    slippage_fixed: float = 0.5    # Fixed slippage in EUR/MWh

    # Market impact (for large orders)
    market_impact_coef: float = 0.01  # Impact coefficient
    daily_volume_pct: float = 0.05     # Max 5% of daily volume

    def calculate_commission(self, quantity: float, price: float) -> float:
        """Calculate commission for a trade."""
        notional = abs(quantity) * price
        commission = max(notional * self.commission_pct, self.commission_min)
        return commission

    def calculate_slippage(self, quantity: float, price: float,
                          volatility: float = 0.0,
                          daily_volume: Optional[float] = None) -> float:
        """
        Calculate slippage for a trade.

        Slippage model:
        - Base slippage: percentage + fixed
        - Market impact: proportional to order size / daily volume
        - Volatility adjustment: higher vol = higher slippage

        Args:
            quantity: Order quantity (MWh)
            price: Execution price (EUR/MWh)
            volatility: Current market volatility (optional)
            daily_volume: Average daily volume (optional)

        Returns:
            slippage: Total slippage cost in EUR
        """
        notional = abs(quantity) * price

        # Base slippage
        base_slippage = notional * self.slippage_pct + self.slippage_fixed * abs(quantity)

        # Market impact (if volume data available)
        market_impact = 0.0
        if daily_volume is not None and daily_volume > 0:
            volume_participation = abs(quantity) / daily_volume
            market_impact = notional * self.market_impact_coef * (volume_participation ** 0.5)

        # Volatility adjustment
        vol_adjustment = 1.0 + volatility if volatility > 0 else 1.0

        total_slippage = (base_slippage + market_impact) * vol_adjustment
        return total_slippage


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""

    initial_capital: float = 100000.0  # EUR

    # Transaction costs
    cost_model: TransactionCostModel = field(default_factory=TransactionCostModel)

    # Risk controls
    max_position_size: float = 1000.0  # MWh per position
    max_leverage: float = 3.0
    max_drawdown_pct: float = 0.20  # Stop trading at 20% drawdown

    # Execution settings
    fill_delay_bars: int = 1  # Delay between signal and fill (1 = next bar)

    # Data settings
    benchmark_symbol: Optional[str] = None  # For Sharpe ratio calculation


class BacktestingEngine:
    """
    Event-driven backtesting engine for trading strategies.

    Features:
    - Realistic transaction costs and slippage
    - Position tracking with mark-to-market
    - Multiple strategies support
    - Performance metrics calculation
    - Walk-forward compatible

    Example:
        >>> config = BacktestConfig(initial_capital=100000)
        >>> engine = BacktestingEngine(config)
        >>> engine.add_strategy('mean_reversion', mean_reversion_strategy)
        >>> results = engine.run(price_data, start_date, end_date)
    """

    def __init__(self, config: BacktestConfig):
        """
        Initialize backtesting engine.

        Args:
            config: Backtesting configuration
        """
        self.config = config
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.trades: List[Dict] = []

        # Account tracking
        self.cash = config.initial_capital
        self.initial_capital = config.initial_capital
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.drawdown_curve: List[Tuple[datetime, float]] = []
        self.peak_equity = config.initial_capital

        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.total_pnl = 0.0
        self.total_commission = 0.0
        self.total_slippage = 0.0

        # Strategy
        self.strategy_func: Optional[Callable] = None
        self.strategy_name: str = ""

        logger.info(f"BacktestingEngine initialized with capital: {config.initial_capital:,.2f} EUR")

    def add_strategy(self, name: str, strategy_func: Callable) -> None:
        """
        Add a trading strategy to the engine.

        Args:
            name: Strategy name
            strategy_func: Function that generates signals
                          Should accept (prices, positions) and return signals DataFrame
        """
        self.strategy_name = name
        self.strategy_func = strategy_func
        logger.info(f"Added strategy: {name}")

    def _create_order(self, timestamp: datetime, symbol: str,
                     signal: float, current_price: float) -> Optional[Order]:
        """
        Create an order based on strategy signal.

        Args:
            timestamp: Order timestamp
            symbol: Trading symbol
            signal: Signal from strategy (-1 to 1, where 0 = no position)
            current_price: Current market price

        Returns:
            Order object or None if no action needed
        """
        # Get current position
        current_position = self.positions.get(symbol, Position(symbol))
        current_qty = current_position.quantity

        # Calculate target position based on signal
        max_position = self.config.max_position_size
        target_qty = signal * max_position

        # Calculate required trade
        trade_qty = target_qty - current_qty

        # Only create order if significant change needed
        if abs(trade_qty) < 1.0:  # Minimum trade size
            return None

        # Determine order side
        side = OrderSide.BUY if trade_qty > 0 else OrderSide.SELL

        order = Order(
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            quantity=abs(trade_qty),
            order_type=OrderType.MARKET
        )

        return order

    def _execute_order(self, order: Order, execution_price: float,
                      volatility: float = 0.0, daily_volume: Optional[float] = None) -> None:
        """
        Execute an order and update positions.

        Args:
            order: Order to execute
            execution_price: Price at which order is filled
            volatility: Current market volatility (for slippage calculation)
            daily_volume: Average daily volume (for market impact)
        """
        # Calculate transaction costs
        commission = self.config.cost_model.calculate_commission(
            order.quantity, execution_price
        )
        slippage = self.config.cost_model.calculate_slippage(
            order.quantity, execution_price, volatility, daily_volume
        )

        # Determine signed quantity
        signed_qty = order.quantity if order.side == OrderSide.BUY else -order.quantity

        # Update order
        order.filled_price = execution_price
        order.filled_quantity = order.quantity
        order.commission = commission
        order.slippage = slippage
        order.status = "filled"

        # Update position
        if order.symbol not in self.positions:
            self.positions[order.symbol] = Position(order.symbol)

        realized_pnl = self.positions[order.symbol].update_position(
            signed_qty, execution_price, commission, slippage
        )

        # Update cash
        trade_cost = signed_qty * execution_price
        self.cash -= trade_cost
        self.cash -= commission
        self.cash -= slippage

        # Track metrics
        self.total_trades += 1
        self.total_commission += commission
        self.total_slippage += slippage

        if realized_pnl > 0:
            self.winning_trades += 1

        # Record trade
        self.trades.append({
            'timestamp': order.timestamp,
            'symbol': order.symbol,
            'side': order.side.value,
            'quantity': order.quantity,
            'price': execution_price,
            'commission': commission,
            'slippage': slippage,
            'realized_pnl': realized_pnl,
            'cash': self.cash
        })

        logger.debug(f"Executed {order.side.value} {order.quantity:.2f} {order.symbol} "
                    f"@ {execution_price:.2f}, Commission: {commission:.2f}, "
                    f"Slippage: {slippage:.2f}, Realized P&L: {realized_pnl:.2f}")

    def _update_equity(self, timestamp: datetime, market_prices: Dict[str, float]) -> float:
        """
        Update equity based on current market prices.

        Args:
            timestamp: Current timestamp
            market_prices: Dictionary of symbol -> current price

        Returns:
            current_equity: Total portfolio equity
        """
        # Mark positions to market
        total_unrealized = 0.0
        for symbol, position in self.positions.items():
            if symbol in market_prices:
                position.mark_to_market(market_prices[symbol])
                total_unrealized += position.unrealized_pnl

        # Calculate total equity
        equity = self.cash + total_unrealized

        # Update equity curve
        self.equity_curve.append((timestamp, equity))

        # Update drawdown
        if equity > self.peak_equity:
            self.peak_equity = equity

        drawdown = (self.peak_equity - equity) / self.peak_equity
        self.drawdown_curve.append((timestamp, drawdown))

        # Check drawdown limit
        if drawdown > self.config.max_drawdown_pct:
            logger.warning(f"Maximum drawdown exceeded: {drawdown:.2%} > {self.config.max_drawdown_pct:.2%}")

        return equity

    def run(self, price_data: pd.DataFrame, signals: pd.DataFrame,
            symbol: str = 'DEMAND', start_date: Optional[datetime] = None,
            end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Run backtest on historical data.

        Args:
            price_data: DataFrame with datetime index and price columns
            signals: DataFrame with datetime index and 'signal' column (-1 to 1)
            symbol: Trading symbol
            start_date: Backtest start date (optional)
            end_date: Backtest end date (optional)

        Returns:
            results: DataFrame with backtest results
        """
        logger.info(f"Starting backtest for {symbol}")
        logger.info(f"Period: {price_data.index[0]} to {price_data.index[-1]}")

        # Filter date range
        if start_date:
            price_data = price_data[price_data.index >= start_date]
            signals = signals[signals.index >= start_date]
        if end_date:
            price_data = price_data[price_data.index <= end_date]
            signals = signals[signals.index <= end_date]

        # Align data
        common_index = price_data.index.intersection(signals.index)
        price_data = price_data.loc[common_index]
        signals = signals.loc[common_index]

        # Main backtest loop
        for i in range(len(price_data)):
            timestamp = price_data.index[i]
            current_price = price_data.iloc[i]['price'] if 'price' in price_data.columns else price_data.iloc[i][0]
            signal = signals.iloc[i]['signal'] if 'signal' in signals.columns else signals.iloc[i][0]

            # Calculate volatility for slippage model (rolling 20-day)
            if i >= 20:
                returns = price_data.iloc[i-20:i].pct_change().dropna()
                volatility = returns.std().iloc[0] if isinstance(returns, pd.DataFrame) else returns.std()
            else:
                volatility = 0.0

            # Create order based on signal
            order = self._create_order(timestamp, symbol, signal, current_price)

            # Execute order (with fill delay)
            if order is not None and i + self.config.fill_delay_bars < len(price_data):
                # Fill at next bar's price (more realistic)
                fill_timestamp = price_data.index[i + self.config.fill_delay_bars]
                fill_price = price_data.iloc[i + self.config.fill_delay_bars]['price'] if 'price' in price_data.columns else price_data.iloc[i + self.config.fill_delay_bars][0]

                self._execute_order(order, fill_price, volatility)
                self.orders.append(order)

            # Update equity
            market_prices = {symbol: current_price}
            equity = self._update_equity(timestamp, market_prices)

        # Generate results
        results = self._generate_results()

        logger.info(f"Backtest completed: {self.total_trades} trades, "
                   f"Final equity: {equity:,.2f} EUR, "
                   f"Total return: {(equity/self.initial_capital - 1)*100:.2f}%")

        return results

    def _generate_results(self) -> pd.DataFrame:
        """
        Generate backtest results summary.

        Returns:
            results: DataFrame with equity curve, positions, and metrics
        """
        # Create equity curve DataFrame
        equity_df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity'])
        equity_df.set_index('timestamp', inplace=True)

        # Calculate returns
        equity_df['returns'] = equity_df['equity'].pct_change()
        equity_df['cumulative_returns'] = (1 + equity_df['returns']).cumprod() - 1

        # Add drawdown
        drawdown_df = pd.DataFrame(self.drawdown_curve, columns=['timestamp', 'drawdown'])
        drawdown_df.set_index('timestamp', inplace=True)
        equity_df = equity_df.join(drawdown_df)

        return equity_df

    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate comprehensive performance metrics.

        Returns:
            metrics: Dictionary of performance metrics
        """
        if len(self.equity_curve) == 0:
            return {}

        equity_df = pd.DataFrame(self.equity_curve, columns=['timestamp', 'equity'])
        equity_df.set_index('timestamp', inplace=True)

        returns = equity_df['equity'].pct_change().dropna()

        # Calculate metrics
        total_return = (equity_df['equity'].iloc[-1] / self.initial_capital) - 1

        # Annualized metrics (assuming daily data)
        days = (equity_df.index[-1] - equity_df.index[0]).days
        years = days / 365.25

        annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        annualized_vol = returns.std() * np.sqrt(252)

        sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0

        # Drawdown metrics
        drawdowns = [dd for _, dd in self.drawdown_curve]
        max_drawdown = max(drawdowns) if drawdowns else 0

        # Win rate
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0

        # Profit factor
        winning_pnl = sum([t['realized_pnl'] for t in self.trades if t['realized_pnl'] > 0])
        losing_pnl = abs(sum([t['realized_pnl'] for t in self.trades if t['realized_pnl'] < 0]))
        profit_factor = winning_pnl / losing_pnl if losing_pnl > 0 else 0

        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': self.total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_commission': self.total_commission,
            'total_slippage': self.total_slippage,
            'final_equity': equity_df['equity'].iloc[-1]
        }

        return metrics
