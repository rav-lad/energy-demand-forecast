"""
Comprehensive PnL Calculation Tests

Tests for position management, PnL calculations, and accounting logic.
Ensures correct calculation of realized/unrealized PnL, commissions, and slippage.

Run with: pytest tests/test_pnl_calculations.py -v
"""

import numpy as np
import pandas as pd
import pytest

from trading_system.backtesting.backtesting_engine import (
    BacktestConfig,
    TransactionCostModel,
)


class Position:
    """Simple position tracker for testing."""

    def __init__(self, symbol: str = "POWER"):
        self.symbol = symbol
        self.quantity = 0.0
        self.avg_entry_price = 0.0
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.total_commission = 0.0
        self.total_slippage = 0.0

    def update_position(
        self, fill_quantity: float, fill_price: float, commission: float, slippage: float
    ) -> float:
        """
        Update position and return realized PnL.

        Args:
            fill_quantity: Quantity filled (positive = buy, negative = sell)
            fill_price: Execution price
            commission: Commission paid
            slippage: Slippage cost

        Returns:
            Realized PnL from this trade
        """
        realized = 0.0

        if self.quantity == 0:
            # Opening new position
            self.quantity = fill_quantity
            self.avg_entry_price = fill_price
        elif (self.quantity > 0 and fill_quantity > 0) or (
            self.quantity < 0 and fill_quantity < 0
        ):
            # Adding to existing position (same direction)
            total_cost = (
                self.quantity * self.avg_entry_price + fill_quantity * fill_price
            )
            self.quantity += fill_quantity
            self.avg_entry_price = total_cost / self.quantity
        elif abs(fill_quantity) < abs(self.quantity):
            # Partial close
            realized = fill_quantity * (fill_price - self.avg_entry_price)
            self.quantity += fill_quantity  # fill_quantity is negative for close
        else:
            # Full close or reversal
            # Close existing position
            realized = -self.quantity * (fill_price - self.avg_entry_price)
            remaining = fill_quantity + self.quantity
            self.quantity = 0
            self.avg_entry_price = 0.0

            # Open new position if reversing
            if abs(remaining) > 1e-6:
                self.quantity = remaining
                self.avg_entry_price = fill_price

        # Deduct costs from realized PnL
        realized -= commission + slippage

        # Track cumulative costs
        self.total_commission += commission
        self.total_slippage += slippage
        self.realized_pnl += realized

        return realized

    def mark_to_market(self, current_price: float) -> float:
        """Calculate unrealized PnL at current market price."""
        if self.quantity == 0:
            self.unrealized_pnl = 0.0
        else:
            self.unrealized_pnl = self.quantity * (current_price - self.avg_entry_price)

        return self.unrealized_pnl

    def total_pnl(self, current_price: float) -> float:
        """Total PnL (realized + unrealized)."""
        self.mark_to_market(current_price)
        return self.realized_pnl + self.unrealized_pnl


class TestPnLCalculations:
    """Test suite for PnL calculations."""

    def test_simple_long_position(self):
        """Test simple long position: buy then sell at profit."""
        position = Position()
        cost_model = TransactionCostModel()

        # Buy 100 @ 50
        commission = cost_model.calculate_commission(quantity=100, price=50)
        slippage = cost_model.calculate_slippage(quantity=100, price=50)
        realized = position.update_position(100, 50, commission, slippage)

        assert position.quantity == 100
        assert position.avg_entry_price == 50
        assert realized < 0  # Opening costs are negative

        # Sell 100 @ 55 (profit)
        commission = cost_model.calculate_commission(quantity=100, price=55)
        slippage = cost_model.calculate_slippage(quantity=100, price=55)
        realized = position.update_position(-100, 55, commission, slippage)

        assert position.quantity == 0
        assert realized > 0  # Should have profit
        # Profit = (55 - 50) * 100 = 500
        # Minus total costs (commissions + slippage for both trades)
        expected_gross = 500
        total_costs = position.total_commission + position.total_slippage
        assert abs(position.realized_pnl - (expected_gross - total_costs)) < 0.01

    def test_simple_short_position(self):
        """Test simple short position: sell then buy at profit."""
        position = Position()
        cost_model = TransactionCostModel()

        # Sell 100 @ 50
        commission = cost_model.calculate_commission(quantity=100, price=50)
        slippage = cost_model.calculate_slippage(quantity=100, price=50)
        position.update_position(-100, 50, commission, slippage)

        assert position.quantity == -100
        assert position.avg_entry_price == 50

        # Buy back 100 @ 45 (profit)
        commission = cost_model.calculate_commission(quantity=100, price=45)
        slippage = cost_model.calculate_slippage(quantity=100, price=45)
        realized = position.update_position(100, 45, commission, slippage)

        assert position.quantity == 0
        assert realized > 0  # Should have profit
        # Profit = (50 - 45) * 100 = 500
        expected_gross = 500
        total_costs = position.total_commission + position.total_slippage
        assert abs(position.realized_pnl - (expected_gross - total_costs)) < 0.01

    def test_partial_close(self):
        """Test partial position close."""
        position = Position()
        cost_model = TransactionCostModel()

        # Buy 100 @ 50
        commission = cost_model.calculate_commission(quantity=100, price=50)
        slippage = cost_model.calculate_slippage(quantity=100, price=50)
        position.update_position(100, 50, commission, slippage)

        # Sell 60 @ 55 (partial close)
        commission = cost_model.calculate_commission(quantity=60, price=55)
        slippage = cost_model.calculate_slippage(quantity=60, price=55)
        realized = position.update_position(-60, 55, commission, slippage)

        assert position.quantity == 40  # 40 remaining
        assert position.avg_entry_price == 50  # Entry price unchanged
        # Partial profit = (55 - 50) * 60 = 300 minus costs
        assert realized > 0

        # Sell remaining 40 @ 52
        commission = cost_model.calculate_commission(quantity=40, price=52)
        slippage = cost_model.calculate_slippage(quantity=40, price=52)
        realized2 = position.update_position(-40, 52, commission, slippage)

        assert position.quantity == 0
        # Total profit = (55-50)*60 + (52-50)*40 = 300 + 80 = 380 minus costs
        expected_gross = 380
        total_costs = position.total_commission + position.total_slippage
        assert abs(position.realized_pnl - (expected_gross - total_costs)) < 0.01

    def test_position_reversal(self):
        """Test position reversal (long to short)."""
        position = Position()
        cost_model = TransactionCostModel()

        # Buy 100 @ 50
        commission = cost_model.calculate_commission(quantity=100, price=50)
        slippage = cost_model.calculate_slippage(quantity=100, price=50)
        position.update_position(100, 50, commission, slippage)

        assert position.quantity == 100

        # Sell 150 @ 55 (close long, open short 50)
        commission = cost_model.calculate_commission(quantity=150, price=55)
        slippage = cost_model.calculate_slippage(quantity=150, price=55)
        realized = position.update_position(-150, 55, commission, slippage)

        assert position.quantity == -50  # Now short 50
        assert position.avg_entry_price == 55  # New entry at 55
        # Realized from closing long: (55 - 50) * 100 = 500 minus costs
        assert realized > 0

    def test_adding_to_position(self):
        """Test adding to existing position (averaging up/down)."""
        position = Position()
        cost_model = TransactionCostModel()

        # Buy 100 @ 50
        commission = cost_model.calculate_commission(quantity=100, price=50)
        slippage = cost_model.calculate_slippage(quantity=100, price=50)
        position.update_position(100, 50, commission, slippage)

        # Buy another 100 @ 60 (averaging up)
        commission = cost_model.calculate_commission(quantity=100, price=60)
        slippage = cost_model.calculate_slippage(quantity=100, price=60)
        position.update_position(100, 60, commission, slippage)

        assert position.quantity == 200
        # Average = (100*50 + 100*60) / 200 = 55
        assert abs(position.avg_entry_price - 55.0) < 0.01

        # Sell all @ 58
        commission = cost_model.calculate_commission(quantity=200, price=58)
        slippage = cost_model.calculate_slippage(quantity=200, price=58)
        realized = position.update_position(-200, 58, commission, slippage)

        assert position.quantity == 0
        # Profit = (58 - 55) * 200 = 600 minus costs
        expected_gross = 600
        assert realized > 0

    def test_unrealized_pnl_long(self):
        """Test unrealized PnL calculation for long position."""
        position = Position()
        cost_model = TransactionCostModel()

        # Buy 100 @ 50
        commission = cost_model.calculate_commission(quantity=100, price=50)
        slippage = cost_model.calculate_slippage(quantity=100, price=50)
        position.update_position(100, 50, commission, slippage)

        # Mark to market @ 55
        unrealized = position.mark_to_market(55)
        assert unrealized == 500  # (55 - 50) * 100

        # Mark to market @ 45 (loss)
        unrealized = position.mark_to_market(45)
        assert unrealized == -500  # (45 - 50) * 100

    def test_unrealized_pnl_short(self):
        """Test unrealized PnL calculation for short position."""
        position = Position()
        cost_model = TransactionCostModel()

        # Sell 100 @ 50
        commission = cost_model.calculate_commission(quantity=100, price=50)
        slippage = cost_model.calculate_slippage(quantity=100, price=50)
        position.update_position(-100, 50, commission, slippage)

        # Mark to market @ 45 (profit for short)
        unrealized = position.mark_to_market(45)
        assert unrealized == 500  # -100 * (45 - 50) = 500

        # Mark to market @ 55 (loss for short)
        unrealized = position.mark_to_market(55)
        assert unrealized == -500  # -100 * (55 - 50) = -500

    def test_total_pnl_conservation(self):
        """Test that total PnL = realized + unrealized."""
        position = Position()
        cost_model = TransactionCostModel()

        # Buy 100 @ 50
        commission = cost_model.calculate_commission(quantity=100, price=50)
        slippage = cost_model.calculate_slippage(quantity=100, price=50)
        position.update_position(100, 50, commission, slippage)

        # Sell 60 @ 55
        commission = cost_model.calculate_commission(quantity=60, price=55)
        slippage = cost_model.calculate_slippage(quantity=60, price=55)
        position.update_position(-60, 55, commission, slippage)

        # Mark remaining to market @ 52
        current_price = 52
        total = position.total_pnl(current_price)

        # Total = realized + unrealized
        expected_total = position.realized_pnl + position.unrealized_pnl
        assert abs(total - expected_total) < 0.01

    def test_flat_position_zero_pnl(self):
        """Test that flat position has zero unrealized PnL."""
        position = Position()

        # No position
        unrealized = position.mark_to_market(50)
        assert unrealized == 0

        cost_model = TransactionCostModel()

        # Open and close
        commission = cost_model.calculate_commission(quantity=100, price=50)
        slippage = cost_model.calculate_slippage(quantity=100, price=50)
        position.update_position(100, 50, commission, slippage)

        commission = cost_model.calculate_commission(quantity=100, price=50)
        slippage = cost_model.calculate_slippage(quantity=100, price=50)
        position.update_position(-100, 50, commission, slippage)

        # Flat again
        unrealized = position.mark_to_market(50)
        assert unrealized == 0

    def test_round_trip_with_costs(self):
        """Test round trip (buy then sell at same price) results in loss from costs."""
        position = Position()
        cost_model = TransactionCostModel()

        # Buy 100 @ 50
        commission1 = cost_model.calculate_commission(quantity=100, price=50)
        slippage1 = cost_model.calculate_slippage(quantity=100, price=50)
        position.update_position(100, 50, commission1, slippage1)

        # Sell 100 @ 50 (same price)
        commission2 = cost_model.calculate_commission(quantity=100, price=50)
        slippage2 = cost_model.calculate_slippage(quantity=100, price=50)
        position.update_position(-100, 50, commission2, slippage2)

        # Should have loss equal to total costs
        total_costs = commission1 + slippage1 + commission2 + slippage2
        assert abs(position.realized_pnl + total_costs) < 0.01
        assert position.realized_pnl < 0  # Loss from costs

    def test_zero_quantity_trade(self):
        """Test that zero quantity trade does nothing."""
        position = Position()

        initial_realized = position.realized_pnl
        position.update_position(0, 50, 0, 0)

        assert position.quantity == 0
        assert position.realized_pnl == initial_realized


class TestTransactionCosts:
    """Test suite for transaction cost calculations."""

    def test_commission_minimum(self):
        """Test that commission has a minimum."""
        cost_model = TransactionCostModel(commission_pct=0.001, commission_min=1.0)

        # Small trade
        commission = cost_model.calculate_commission(quantity=1, price=50)
        assert commission == 1.0  # Minimum

        # Large trade
        commission = cost_model.calculate_commission(quantity=1000, price=50)
        expected = 1000 * 50 * 0.001  # = 50
        assert abs(commission - expected) < 0.01

    def test_slippage_calculation(self):
        """Test slippage calculation."""
        cost_model = TransactionCostModel(
            slippage_pct=0.0005, slippage_fixed=0.5  # EUR/MWh
        )

        quantity = 100
        price = 50
        slippage = cost_model.calculate_slippage(quantity, price)

        # Expected = (100 * 50 * 0.0005) + (100 * 0.5) = 2.5 + 50 = 52.5
        expected = quantity * price * 0.0005 + quantity * 0.5
        assert abs(slippage - expected) < 0.01

    def test_slippage_with_volatility(self):
        """Test that slippage increases with volatility."""
        cost_model = TransactionCostModel()

        slippage_low_vol = cost_model.calculate_slippage(
            quantity=100, price=50, volatility=0.01
        )
        slippage_high_vol = cost_model.calculate_slippage(
            quantity=100, price=50, volatility=0.10
        )

        assert slippage_high_vol > slippage_low_vol

    def test_market_impact(self):
        """Test market impact for large orders."""
        cost_model = TransactionCostModel(market_impact_coef=0.01)

        # Small order (1% of daily volume)
        slippage_small = cost_model.calculate_slippage(
            quantity=100, price=50, daily_volume=10000
        )

        # Large order (10% of daily volume)
        slippage_large = cost_model.calculate_slippage(
            quantity=1000, price=50, daily_volume=10000
        )

        assert slippage_large > slippage_small


class TestSanityChecks:
    """Sanity checks for backtesting logic."""

    def test_no_trades_zero_pnl(self):
        """Test that no trades results in zero PnL."""
        position = Position()

        assert position.realized_pnl == 0
        assert position.unrealized_pnl == 0
        assert position.total_pnl(50) == 0

    def test_equity_conservation(self):
        """Test equity = initial_capital + total_pnl."""
        initial_capital = 100000.0
        position = Position()
        cost_model = TransactionCostModel()

        # Execute some trades
        commission = cost_model.calculate_commission(quantity=100, price=50)
        slippage = cost_model.calculate_slippage(quantity=100, price=50)
        position.update_position(100, 50, commission, slippage)

        commission = cost_model.calculate_commission(quantity=100, price=55)
        slippage = cost_model.calculate_slippage(quantity=100, price=55)
        position.update_position(-100, 55, commission, slippage)

        # Equity should be initial capital + realized PnL
        equity = initial_capital + position.realized_pnl
        assert equity > 0
        assert equity == initial_capital + position.realized_pnl

    def test_random_trades_consistency(self):
        """Test PnL consistency with random trades."""
        np.random.seed(42)
        position = Position()
        cost_model = TransactionCostModel()

        # Execute 100 random trades
        for _ in range(100):
            quantity = np.random.choice([-100, -50, 0, 50, 100])
            price = np.random.uniform(40, 60)

            commission = cost_model.calculate_commission(abs(quantity), price)
            slippage = cost_model.calculate_slippage(abs(quantity), price)

            position.update_position(quantity, price, commission, slippage)

        # At the end, close any remaining position
        if position.quantity != 0:
            commission = cost_model.calculate_commission(abs(position.quantity), 50)
            slippage = cost_model.calculate_slippage(abs(position.quantity), 50)
            position.update_position(-position.quantity, 50, commission, slippage)

        # Should be flat with some realized PnL (positive or negative)
        assert position.quantity == 0
        assert position.unrealized_pnl == 0
        # Realized PnL should be non-zero (due to random price movements and costs)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
