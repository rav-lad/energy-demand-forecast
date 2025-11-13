"""
Backtesting Results Analyzer and Visualization

This module provides comprehensive analysis and visualization of backtest results:
- Equity curve analysis
- Drawdown analysis
- Returns distribution
- Trade analysis
- Performance tear sheets
- Export to PDF/HTML

Author: Energy Trading Quant Team
Date: 2025-11-12
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TradeAnalysis:
    """Analysis of individual trades."""

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # P&L statistics
    total_pnl: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    profit_factor: float

    # Trade duration
    avg_trade_duration_days: float
    max_trade_duration_days: float

    # Consecutive statistics
    max_consecutive_wins: int
    max_consecutive_losses: int

    def __str__(self) -> str:
        """String representation."""
        return f"""
Trade Analysis
{'='*60}
Total Trades:           {self.total_trades:>10}
Winning Trades:         {self.winning_trades:>10}
Losing Trades:          {self.losing_trades:>10}
Win Rate:               {self.win_rate:>10.2%}

P&L Statistics:
Total P&L:              {self.total_pnl:>10,.2f} EUR
Average Win:            {self.avg_win:>10,.2f} EUR
Average Loss:           {self.avg_loss:>10,.2f} EUR
Largest Win:            {self.largest_win:>10,.2f} EUR
Largest Loss:           {self.largest_loss:>10,.2f} EUR
Profit Factor:          {self.profit_factor:>10.3f}

Trade Duration:
Average Duration:       {self.avg_trade_duration_days:>10.1f} days
Maximum Duration:       {self.max_trade_duration_days:>10.1f} days

Consecutive:
Max Consecutive Wins:   {self.max_consecutive_wins:>10}
Max Consecutive Losses: {self.max_consecutive_losses:>10}
{'='*60}
"""


class ResultsAnalyzer:
    """
    Comprehensive analyzer for backtesting results.

    Provides:
    - Statistical analysis of trades
    - Performance metrics calculation
    - Regime analysis (bull/bear market performance)
    - Visualization and reporting

    Example:
        >>> analyzer = ResultsAnalyzer()
        >>> trade_analysis = analyzer.analyze_trades(trades)
        >>> print(trade_analysis)
        >>> analyzer.plot_equity_curve(equity_curve)
    """

    def __init__(self):
        """Initialize results analyzer."""
        logger.info("ResultsAnalyzer initialized")

    def analyze_trades(self, trades: List[Dict]) -> TradeAnalysis:
        """
        Analyze trading activity.

        Args:
            trades: List of trade dictionaries from backtesting engine

        Returns:
            analysis: TradeAnalysis object
        """
        if not trades:
            logger.warning("No trades to analyze")
            return TradeAnalysis(
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0,
                total_pnl=0,
                avg_win=0,
                avg_loss=0,
                largest_win=0,
                largest_loss=0,
                profit_factor=0,
                avg_trade_duration_days=0,
                max_trade_duration_days=0,
                max_consecutive_wins=0,
                max_consecutive_losses=0,
            )

        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(trades)

        # Basic statistics
        total_trades = len(df)
        winning_trades = len(df[df["realized_pnl"] > 0])
        losing_trades = len(df[df["realized_pnl"] < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        # P&L statistics
        total_pnl = df["realized_pnl"].sum()

        wins = df[df["realized_pnl"] > 0]["realized_pnl"]
        losses = df[df["realized_pnl"] < 0]["realized_pnl"]

        avg_win = wins.mean() if len(wins) > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        largest_win = wins.max() if len(wins) > 0 else 0
        largest_loss = losses.min() if len(losses) > 0 else 0

        # Profit factor
        gross_profit = wins.sum() if len(wins) > 0 else 0
        gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Trade duration (need to group by position open/close)
        # For simplicity, assuming each trade represents full round-trip
        if "timestamp" in df.columns and len(df) > 1:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            avg_duration = (df["timestamp"].diff().mean().days) if len(df) > 1 else 0
            max_duration = (df["timestamp"].diff().max().days) if len(df) > 1 else 0
        else:
            avg_duration = 0
            max_duration = 0

        # Consecutive wins/losses
        win_loss = (df["realized_pnl"] > 0).astype(int)
        consecutive_wins = self._max_consecutive(win_loss)
        consecutive_losses = self._max_consecutive(1 - win_loss)

        analysis = TradeAnalysis(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            profit_factor=profit_factor,
            avg_trade_duration_days=avg_duration,
            max_trade_duration_days=max_duration,
            max_consecutive_wins=consecutive_wins,
            max_consecutive_losses=consecutive_losses,
        )

        return analysis

    def _max_consecutive(self, series: pd.Series) -> int:
        """Calculate maximum consecutive 1s in binary series."""
        if len(series) == 0:
            return 0

        max_consec = 0
        current_consec = 0

        for val in series:
            if val == 1:
                current_consec += 1
                max_consec = max(max_consec, current_consec)
            else:
                current_consec = 0

        return max_consec

    def analyze_returns_distribution(self, returns: pd.Series) -> Dict:
        """
        Analyze statistical properties of returns distribution.

        Args:
            returns: Series of returns

        Returns:
            stats: Dictionary of distribution statistics
        """
        stats = {
            "mean": returns.mean(),
            "median": returns.median(),
            "std": returns.std(),
            "skewness": returns.skew(),
            "kurtosis": returns.kurtosis(),
            "min": returns.min(),
            "max": returns.max(),
            "percentile_5": returns.quantile(0.05),
            "percentile_25": returns.quantile(0.25),
            "percentile_75": returns.quantile(0.75),
            "percentile_95": returns.quantile(0.95),
        }

        # Jarque-Bera test for normality
        from scipy import stats as scipy_stats

        jb_stat, jb_pvalue = scipy_stats.jarque_bera(returns.dropna())
        stats["jarque_bera_stat"] = jb_stat
        stats["jarque_bera_pvalue"] = jb_pvalue
        stats["is_normal"] = jb_pvalue > 0.05

        logger.info("Returns Distribution Analysis:")
        logger.info(f"  Mean: {stats['mean']:.4f}")
        logger.info(f"  Std Dev: {stats['std']:.4f}")
        logger.info(f"  Skewness: {stats['skewness']:.3f}")
        logger.info(f"  Kurtosis: {stats['kurtosis']:.3f}")
        logger.info(f"  Normality: {'Yes' if stats['is_normal'] else 'No'} (p={jb_pvalue:.4f})")

        return stats

    def regime_analysis(self, equity_curve: pd.DataFrame, regime_threshold: float = 0.0) -> Dict:
        """
        Analyze performance in different market regimes.

        Args:
            equity_curve: DataFrame with equity and returns
            regime_threshold: Threshold for defining bull/bear (default: 0 = median)

        Returns:
            regime_stats: Dictionary with regime-specific statistics
        """
        if "returns" not in equity_curve.columns:
            equity_curve["returns"] = equity_curve["equity"].pct_change()

        returns = equity_curve["returns"].dropna()

        # Define regimes based on rolling mean
        rolling_mean = returns.rolling(window=20).mean()

        bull_returns = returns[rolling_mean > regime_threshold]
        bear_returns = returns[rolling_mean <= regime_threshold]

        regime_stats = {
            "bull_market": {
                "periods": len(bull_returns),
                "mean_return": bull_returns.mean() if len(bull_returns) > 0 else 0,
                "std_return": bull_returns.std() if len(bull_returns) > 0 else 0,
                "sharpe": (
                    (bull_returns.mean() / bull_returns.std() * np.sqrt(252))
                    if len(bull_returns) > 0 and bull_returns.std() > 0
                    else 0
                ),
            },
            "bear_market": {
                "periods": len(bear_returns),
                "mean_return": bear_returns.mean() if len(bear_returns) > 0 else 0,
                "std_return": bear_returns.std() if len(bear_returns) > 0 else 0,
                "sharpe": (
                    (bear_returns.mean() / bear_returns.std() * np.sqrt(252))
                    if len(bear_returns) > 0 and bear_returns.std() > 0
                    else 0
                ),
            },
        }

        logger.info("Regime Analysis:")
        logger.info(
            f"  Bull Market: {regime_stats['bull_market']['periods']} periods, "
            f"Sharpe: {regime_stats['bull_market']['sharpe']:.3f}"
        )
        logger.info(
            f"  Bear Market: {regime_stats['bear_market']['periods']} periods, "
            f"Sharpe: {regime_stats['bear_market']['sharpe']:.3f}"
        )

        return regime_stats

    def plot_equity_curve(
        self,
        equity_curve: pd.DataFrame,
        title: str = "Equity Curve",
        save_path: Optional[str] = None,
    ) -> None:
        """
        Plot equity curve with drawdown.

        Args:
            equity_curve: DataFrame with equity and drawdown
            title: Plot title
            save_path: Path to save figure (optional)
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            sns.set_style("whitegrid")
        except ImportError:
            logger.warning("matplotlib/seaborn not available")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # Equity curve
        ax1.plot(equity_curve.index, equity_curve["equity"], linewidth=2, color="blue")
        ax1.set_ylabel("Equity (EUR)", fontsize=12)
        ax1.set_title(title, fontsize=14, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:,.0f}"))

        # Drawdown
        if "drawdown" in equity_curve.columns:
            ax2.fill_between(
                equity_curve.index, equity_curve["drawdown"] * 100, 0, color="red", alpha=0.3
            )
            ax2.plot(
                equity_curve.index, equity_curve["drawdown"] * 100, linewidth=1, color="darkred"
            )
            ax2.set_ylabel("Drawdown (%)", fontsize=12)
            ax2.set_xlabel("Date", fontsize=12)
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Equity curve plot saved to {save_path}")

        plt.show()

    def plot_returns_distribution(
        self, returns: pd.Series, save_path: Optional[str] = None
    ) -> None:
        """
        Plot returns distribution with normal overlay.

        Args:
            returns: Series of returns
            save_path: Path to save figure (optional)
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            from scipy import stats as scipy_stats

            sns.set_style("whitegrid")
        except ImportError:
            logger.warning("matplotlib/seaborn not available")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Histogram with normal overlay
        ax1.hist(
            returns * 100, bins=50, density=True, alpha=0.7, edgecolor="black", label="Empirical"
        )

        # Fit normal distribution
        mu, sigma = returns.mean() * 100, returns.std() * 100
        x = np.linspace(returns.min() * 100, returns.max() * 100, 100)
        ax1.plot(x, scipy_stats.norm.pdf(x, mu, sigma), "r-", linewidth=2, label="Normal")

        ax1.set_xlabel("Returns (%)", fontsize=12)
        ax1.set_ylabel("Density", fontsize=12)
        ax1.set_title("Returns Distribution", fontsize=14, fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Q-Q plot
        scipy_stats.probplot(returns, dist="norm", plot=ax2)
        ax2.set_title("Q-Q Plot (Normal)", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Returns distribution plot saved to {save_path}")

        plt.show()

    def plot_rolling_metrics(
        self, equity_curve: pd.DataFrame, window: int = 60, save_path: Optional[str] = None
    ) -> None:
        """
        Plot rolling performance metrics.

        Args:
            equity_curve: DataFrame with equity
            window: Rolling window size (days)
            save_path: Path to save figure (optional)
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            sns.set_style("whitegrid")
        except ImportError:
            logger.warning("matplotlib/seaborn not available")
            return

        if "returns" not in equity_curve.columns:
            equity_curve["returns"] = equity_curve["equity"].pct_change()

        returns = equity_curve["returns"]

        # Calculate rolling metrics
        rolling_sharpe = (
            returns.rolling(window).mean() / returns.rolling(window).std() * np.sqrt(252)
        )
        rolling_vol = returns.rolling(window).std() * np.sqrt(252)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # Rolling Sharpe
        ax1.plot(equity_curve.index, rolling_sharpe, linewidth=2, color="green")
        ax1.axhline(y=1.0, color="r", linestyle="--", alpha=0.5, label="Sharpe = 1.0")
        ax1.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
        ax1.set_ylabel(f"{window}-Day Rolling Sharpe", fontsize=12)
        ax1.set_title("Rolling Performance Metrics", fontsize=14, fontweight="bold")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Rolling Volatility
        ax2.plot(equity_curve.index, rolling_vol * 100, linewidth=2, color="orange")
        ax2.set_ylabel(f"{window}-Day Rolling Vol (%)", fontsize=12)
        ax2.set_xlabel("Date", fontsize=12)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Rolling metrics plot saved to {save_path}")

        plt.show()

    def generate_tearsheet(self, backtest_results: Dict, save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive performance tear sheet.

        Args:
            backtest_results: Dictionary with all backtest results
            save_path: Path to save tear sheet (optional)

        Returns:
            tearsheet: Formatted tear sheet string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("PERFORMANCE TEAR SHEET")
        lines.append("=" * 80)
        lines.append("")

        # Extract components
        perf = backtest_results.get("performance_metrics", {})
        attr = backtest_results.get("attribution", {})
        risk = backtest_results.get("risk_metrics", {})
        trades = backtest_results.get("trades", [])

        # Performance section
        lines.append("RETURNS & RISK")
        lines.append("-" * 80)
        lines.append(f"Total Return:              {perf.get('total_return', 0):>12.2%}")
        lines.append(f"Annualized Return:         {perf.get('annualized_return', 0):>12.2%}")
        lines.append(f"Annualized Volatility:     {perf.get('annualized_volatility', 0):>12.2%}")
        lines.append(f"Sharpe Ratio:              {perf.get('sharpe_ratio', 0):>12.3f}")
        lines.append(f"Sortino Ratio:             {risk.get('sortino_ratio', 0):>12.3f}")
        lines.append(f"Maximum Drawdown:          {perf.get('max_drawdown', 0):>12.2%}")
        lines.append("")

        # Attribution section
        if attr:
            lines.append("PERFORMANCE ATTRIBUTION")
            lines.append("-" * 80)
            if "alpha" in attr:
                lines.append(f"Alpha (annualized):        {attr['alpha']:>12.2%}")
                lines.append(f"Beta:                      {attr['beta']:>12.3f}")
                lines.append(f"Information Ratio:         {attr['information_ratio']:>12.3f}")
            lines.append("")

        # Trading activity
        if trades:
            trade_analysis = self.analyze_trades(trades)
            lines.append("TRADING ACTIVITY")
            lines.append("-" * 80)
            lines.append(f"Total Trades:              {trade_analysis.total_trades:>12}")
            lines.append(f"Win Rate:                  {trade_analysis.win_rate:>12.2%}")
            lines.append(f"Profit Factor:             {trade_analysis.profit_factor:>12.3f}")
            lines.append(f"Avg Win:                   {trade_analysis.avg_win:>12,.2f} EUR")
            lines.append(f"Avg Loss:                  {trade_analysis.avg_loss:>12,.2f} EUR")
            lines.append("")

        lines.append("=" * 80)

        tearsheet = "\n".join(lines)

        if save_path:
            with open(save_path, "w") as f:
                f.write(tearsheet)
            logger.info(f"Tear sheet saved to {save_path}")

        return tearsheet
