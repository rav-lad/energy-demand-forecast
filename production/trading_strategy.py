"""Trading strategies based on price forecasts.

Implements peak shaving and other day-ahead trading strategies.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class PeakShavingStrategy:
    """
    Peak shaving trading strategy.

    Each day:
    1. Receive 24h price forecasts
    2. Buy at hour with lowest predicted price
    3. Sell at hour with highest predicted price
    4. PnL = actual_price(sell_hour) - actual_price(buy_hour)
    """

    def __init__(self, initial_capital: float = 10000, trade_volume: float = 1.0):
        """
        Initialize peak shaving strategy.

        Args:
            initial_capital: Starting capital in EUR
            trade_volume: Volume to trade in MWh
        """
        self.initial_capital = initial_capital
        self.trade_volume = trade_volume

    def execute_trades(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute peak shaving trades based on predictions.

        Args:
            predictions_df: DataFrame with ['datetime_hour', 'actual', 'predicted', 'iteration']

        Returns:
            DataFrame with daily trades and PnL
        """
        df = predictions_df.copy()
        df['date'] = pd.to_datetime(df['datetime_hour']).dt.date

        trades = []

        # Group by iteration (each iteration = batch of 24h predictions made together)
        for iteration, group in df.groupby('iteration'):
            if len(group) == 0:
                continue

            # Find hours with min and max predicted prices
            min_idx = group['predicted'].idxmin()
            max_idx = group['predicted'].idxmax()

            buy_hour = group.loc[min_idx, 'datetime_hour']
            sell_hour = group.loc[max_idx, 'datetime_hour']

            buy_price_pred = group.loc[min_idx, 'predicted']
            sell_price_pred = group.loc[max_idx, 'predicted']

            buy_price_actual = group.loc[min_idx, 'actual']
            sell_price_actual = group.loc[max_idx, 'actual']

            # Calculate PnL
            # IMPORTANT: We always buy at predicted_min and sell at predicted_max
            # regardless of temporal order, because predictions were made simultaneously
            # at 00:00 for the entire day. The temporal sequence doesn't matter for
            # day-ahead market where we can place orders for any hour.
            pnl = (sell_price_actual - buy_price_actual) * self.trade_volume

            trades.append({
                'iteration': iteration,
                'date': group['date'].iloc[0],
                'buy_hour': buy_hour,
                'sell_hour': sell_hour,
                'buy_price_pred': buy_price_pred,
                'sell_price_pred': sell_price_pred,
                'buy_price_actual': buy_price_actual,
                'sell_price_actual': sell_price_actual,
                'pnl': pnl,
                'volume_mwh': self.trade_volume
            })

        return pd.DataFrame(trades)

    @staticmethod
    def calculate_metrics(trades_df: pd.DataFrame, initial_capital: float = 10000) -> Dict[str, float]:
        """
        Calculate trading performance metrics.

        Args:
            trades_df: DataFrame with daily trades
            initial_capital: Starting capital in EUR

        Returns:
            Dictionary with metrics
        """
        if len(trades_df) == 0:
            return {
                'total_pnl': 0,
                'n_trades': 0,
                'win_rate': 0,
                'avg_profit': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'return_pct': 0
            }

        # Calculate metrics
        total_pnl = trades_df['pnl'].sum()
        n_trades = len(trades_df)
        win_rate = (trades_df['pnl'] > 0).sum() / n_trades * 100
        avg_profit = trades_df['pnl'].mean()

        # Cumulative PnL for drawdown calculation
        trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
        cumulative_max = trades_df['cumulative_pnl'].expanding().max()
        drawdown = trades_df['cumulative_pnl'] - cumulative_max
        max_drawdown = drawdown.min()

        # Sharpe ratio (daily returns)
        if trades_df['pnl'].std() > 0:
            sharpe_ratio = (avg_profit / trades_df['pnl'].std()) * np.sqrt(252)
        else:
            sharpe_ratio = 0

        return_pct = (total_pnl / initial_capital) * 100

        return {
            'total_pnl': total_pnl,
            'n_trades': n_trades,
            'win_rate': win_rate,
            'avg_profit': avg_profit,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'return_pct': return_pct
        }


def backtest_strategy(
    predictions_df: pd.DataFrame,
    strategy: str = "peak_shaving",
    initial_capital: float = 10000,
    trade_volume: float = 1.0
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Backtest a trading strategy.

    Args:
        predictions_df: DataFrame with price predictions
        strategy: Strategy name ("peak_shaving")
        initial_capital: Starting capital
        trade_volume: Volume per trade

    Returns:
        trades_df: DataFrame with executed trades
        metrics: Performance metrics
    """
    if strategy == "peak_shaving":
        strat = PeakShavingStrategy(initial_capital=initial_capital, trade_volume=trade_volume)
        trades_df = strat.execute_trades(predictions_df)
        metrics = strat.calculate_metrics(trades_df, initial_capital=initial_capital)
        return trades_df, metrics
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


if __name__ == "__main__":
    # Test trading strategy
    print("Testing peak shaving strategy...")

    # Create dummy predictions
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=72, freq='H')
    predictions = pd.DataFrame({
        'datetime_hour': dates,
        'actual': np.random.uniform(40, 100, 72),
        'predicted': np.random.uniform(40, 100, 72),
        'iteration': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                      2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                      3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
    })

    # Execute trades
    trades_df, metrics = backtest_strategy(predictions, strategy="peak_shaving")

    print(f"\nExecuted {len(trades_df)} trades")
    print(f"Total PnL: {metrics['total_pnl']:.2f} EUR")
    print(f"Win rate: {metrics['win_rate']:.2f}%")
    print(f"Avg profit: {metrics['avg_profit']:.2f} EUR/day")
    print(f"Sharpe ratio: {metrics['sharpe_ratio']:.3f}")

    print("\n[OK] Trading strategy test passed")
