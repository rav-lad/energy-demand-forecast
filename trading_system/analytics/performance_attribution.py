"""
Performance Attribution Analysis for Trading Strategies

Decompose portfolio returns into:
- Market return (beta)
- Alpha (excess return)
- Timing vs Selection effects
- Factor contributions

Author: Quant Research Team
Date: 2024-11-12
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
from scipy import stats
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


@dataclass
class AttributionMetrics:
    """Performance attribution metrics."""

    total_return: float
    market_return: float
    alpha: float
    beta: float
    information_ratio: float
    timing_contribution: float
    selection_contribution: float
    interaction_contribution: float


class PerformanceAttributor:
    """
    Performance Attribution Analysis.

    Breaks down strategy returns into components:
    1. Market return (systematic risk)
    2. Alpha (skill-based excess return)
    3. Timing (when to trade)
    4. Selection (what to trade)

    Example:
        >>> attributor = PerformanceAttributor()
        >>> metrics = attributor.analyze(
        ...     strategy_returns=strategy_returns,
        ...     benchmark_returns=benchmark_returns
        ... )
    """

    def __init__(self):
        """Initialize Performance Attributor."""
        logger.info("Initialized Performance Attributor")

    def calculate_alpha_beta(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> Tuple[float, float, float]:
        """
        Calculate alpha and beta using CAPM regression.

        Model: R_strategy = α + β*R_benchmark + ε

        Args:
            strategy_returns: Strategy return series
            benchmark_returns: Benchmark return series

        Returns:
            (alpha, beta, r_squared)
        """
        # Align series
        common_idx = strategy_returns.index.intersection(benchmark_returns.index)
        strat_ret = strategy_returns.loc[common_idx].values.reshape(-1, 1)
        bench_ret = benchmark_returns.loc[common_idx].values

        # OLS regression
        model = LinearRegression()
        model.fit(bench_ret.reshape(-1, 1), strat_ret)

        alpha = model.intercept_[0]
        beta = model.coef_[0][0]
        r_squared = model.score(bench_ret.reshape(-1, 1), strat_ret)

        # Annualize alpha (assuming daily returns)
        alpha_annualized = alpha * 252

        logger.info(
            f"Alpha (annualized): {alpha_annualized:.4f}, "
            f"Beta: {beta:.4f}, "
            f"R²: {r_squared:.4f}"
        )

        return alpha_annualized, beta, r_squared

    def calculate_information_ratio(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series
    ) -> float:
        """
        Calculate Information Ratio (IR).

        IR = (Mean excess return) / (Tracking error)
        IR = Alpha / (Std of excess returns)

        Measures risk-adjusted alpha generation.

        Args:
            strategy_returns: Strategy return series
            benchmark_returns: Benchmark return series

        Returns:
            Information Ratio (annualized)
        """
        # Excess returns
        common_idx = strategy_returns.index.intersection(benchmark_returns.index)
        excess_returns = strategy_returns.loc[common_idx] - benchmark_returns.loc[common_idx]

        # IR
        mean_excess = excess_returns.mean()
        tracking_error = excess_returns.std()

        if tracking_error > 0:
            ir = mean_excess / tracking_error * np.sqrt(252)  # Annualize
        else:
            ir = 0.0

        return ir

    def timing_selection_attribution(
        self,
        portfolio_returns: pd.Series,
        asset_returns: pd.DataFrame,
        weights: pd.DataFrame,
        benchmark_weights: Optional[pd.DataFrame] = None
    ) -> Dict[str, float]:
        """
        Brinson-Fachler attribution: decompose returns into Timing and Selection.

        Total excess return = Timing + Selection + Interaction

        Timing: Return from being overweight/underweight sectors at right time
        Selection: Return from picking right assets within sectors
        Interaction: Combined effect

        Args:
            portfolio_returns: Portfolio return series
            asset_returns: DataFrame of asset returns
            weights: DataFrame of portfolio weights
            benchmark_weights: Benchmark weights (if None, use equal weight)

        Returns:
            Dict with timing, selection, and interaction contributions
        """
        if benchmark_weights is None:
            # Equal weight benchmark
            n_assets = asset_returns.shape[1]
            benchmark_weights = pd.DataFrame(
                1.0 / n_assets,
                index=asset_returns.index,
                columns=asset_returns.columns
            )

        # Calculate component returns
        timing = 0.0
        selection = 0.0
        interaction = 0.0

        for date in portfolio_returns.index:
            if date not in weights.index or date not in benchmark_weights.index:
                continue

            w_p = weights.loc[date]  # Portfolio weights
            w_b = benchmark_weights.loc[date]  # Benchmark weights

            if date not in asset_returns.index:
                continue

            r = asset_returns.loc[date]  # Asset returns

            # Benchmark return
            r_b = (w_b * r).sum()

            # Timing effect: (w_p - w_b) * r_b
            timing += ((w_p - w_b) * r_b).sum()

            # Selection effect: w_b * (r - r_b)
            selection += (w_b * (r - r_b)).sum()

            # Interaction: (w_p - w_b) * (r - r_b)
            interaction += ((w_p - w_b) * (r - r_b)).sum()

        results = {
            'timing': timing,
            'selection': selection,
            'interaction': interaction,
            'total': timing + selection + interaction
        }

        logger.info(
            f"Attribution - Timing: {timing:.4f}, "
            f"Selection: {selection:.4f}, "
            f"Interaction: {interaction:.4f}"
        )

        return results

    def factor_attribution(
        self,
        strategy_returns: pd.Series,
        factor_returns: pd.DataFrame
    ) -> pd.Series:
        """
        Factor-based attribution: decompose returns by factor exposures.

        Model: R_strategy = Σ(β_i * F_i) + α + ε

        Args:
            strategy_returns: Strategy return series
            factor_returns: DataFrame with factor return series

        Returns:
            Series with factor contributions
        """
        # Align data
        common_idx = strategy_returns.index.intersection(factor_returns.index)
        y = strategy_returns.loc[common_idx].values
        X = factor_returns.loc[common_idx].values

        # Multiple regression
        model = LinearRegression()
        model.fit(X, y)

        # Factor exposures (betas)
        betas = model.coef_

        # Factor contributions
        factor_contributions = {}
        for i, factor_name in enumerate(factor_returns.columns):
            # Contribution = beta * mean_factor_return
            contribution = betas[i] * factor_returns[factor_name].mean() * 252  # Annualize
            factor_contributions[factor_name] = contribution

        # Alpha (intercept)
        alpha = model.intercept_ * 252
        factor_contributions['alpha'] = alpha

        # Unexplained (residual)
        y_pred = model.predict(X)
        residuals = y - y_pred
        unexplained = residuals.std() * np.sqrt(252)
        factor_contributions['unexplained'] = unexplained

        return pd.Series(factor_contributions)

    def analyze_strategy(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        factor_returns: Optional[pd.DataFrame] = None
    ) -> AttributionMetrics:
        """
        Comprehensive performance attribution analysis.

        Args:
            strategy_returns: Strategy return series
            benchmark_returns: Benchmark return series
            factor_returns: Optional factor returns for factor attribution

        Returns:
            AttributionMetrics with detailed breakdown
        """
        logger.info("Performing comprehensive performance attribution...")

        # Total returns
        total_strategy_return = (1 + strategy_returns).prod() - 1
        total_benchmark_return = (1 + benchmark_returns).prod() - 1

        # Alpha and beta
        alpha, beta, r_squared = self.calculate_alpha_beta(
            strategy_returns,
            benchmark_returns
        )

        # Information ratio
        ir = self.calculate_information_ratio(strategy_returns, benchmark_returns)

        # Market return component
        market_return = beta * total_benchmark_return

        # For timing/selection, need more granular data
        # Simplified attribution here
        timing_contribution = 0.0
        selection_contribution = alpha
        interaction_contribution = 0.0

        metrics = AttributionMetrics(
            total_return=total_strategy_return,
            market_return=market_return,
            alpha=alpha,
            beta=beta,
            information_ratio=ir,
            timing_contribution=timing_contribution,
            selection_contribution=selection_contribution,
            interaction_contribution=interaction_contribution
        )

        logger.info("Attribution analysis complete")

        return metrics

    def calculate_drawdown_attribution(
        self,
        strategy_equity: pd.Series,
        positions: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Attribute drawdowns to specific positions/strategies.

        Args:
            strategy_equity: Strategy equity curve
            positions: DataFrame with position data

        Returns:
            DataFrame with drawdown attribution
        """
        # Calculate drawdowns
        running_max = strategy_equity.expanding().max()
        drawdown = (strategy_equity - running_max) / running_max

        # Find largest drawdown period
        max_dd_idx = drawdown.idxmin()
        dd_start_idx = strategy_equity[:max_dd_idx].idxmax()

        # Get positions during drawdown
        dd_period = positions.loc[dd_start_idx:max_dd_idx]

        # Calculate contribution of each position to drawdown
        if len(dd_period) > 0 and 'pnl' in dd_period.columns:
            position_contributions = dd_period.groupby('position_id')['pnl'].sum()
            position_contributions = position_contributions.sort_values()

            return position_contributions
        else:
            return pd.Series()

    def rolling_attribution(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        window: int = 60
    ) -> pd.DataFrame:
        """
        Calculate rolling alpha and beta over time.

        Useful for detecting regime changes and strategy drift.

        Args:
            strategy_returns: Strategy return series
            benchmark_returns: Benchmark return series
            window: Rolling window size (days)

        Returns:
            DataFrame with rolling alpha, beta, and IR
        """
        results = []

        common_idx = strategy_returns.index.intersection(benchmark_returns.index)
        aligned_strat = strategy_returns.loc[common_idx]
        aligned_bench = benchmark_returns.loc[common_idx]

        for i in range(window, len(common_idx)):
            window_strat = aligned_strat.iloc[i - window:i]
            window_bench = aligned_bench.iloc[i - window:i]

            # Calculate alpha, beta for window
            try:
                alpha, beta, _ = self.calculate_alpha_beta(window_strat, window_bench)
                ir = self.calculate_information_ratio(window_strat, window_bench)

                results.append({
                    'date': common_idx[i],
                    'alpha': alpha,
                    'beta': beta,
                    'information_ratio': ir
                })
            except Exception as e:
                logger.warning(f"Error calculating rolling attribution: {e}")
                continue

        return pd.DataFrame(results)


def create_performance_report(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    strategy_name: str = "Strategy"
) -> str:
    """
    Generate comprehensive performance report.

    Args:
        strategy_returns: Strategy return series
        benchmark_returns: Benchmark return series
        strategy_name: Name of strategy

    Returns:
        Formatted report string
    """
    attributor = PerformanceAttributor()

    # Calculate metrics
    metrics = attributor.analyze_strategy(strategy_returns, benchmark_returns)

    # Build report
    report = []
    report.append("=" * 60)
    report.append(f"PERFORMANCE ATTRIBUTION REPORT: {strategy_name}")
    report.append("=" * 60)
    report.append("")

    report.append("RETURN DECOMPOSITION:")
    report.append(f"  Total Return:            {metrics.total_return:10.2%}")
    report.append(f"  Market Return (β × Rm):  {metrics.market_return:10.2%}")
    report.append(f"  Alpha:                   {metrics.alpha:10.2%}")
    report.append("")

    report.append("RISK METRICS:")
    report.append(f"  Beta:                    {metrics.beta:10.4f}")
    report.append(f"  Information Ratio:       {metrics.information_ratio:10.4f}")
    report.append("")

    report.append("SKILL ATTRIBUTION:")
    report.append(f"  Timing:                  {metrics.timing_contribution:10.2%}")
    report.append(f"  Selection:               {metrics.selection_contribution:10.2%}")
    report.append(f"  Interaction:             {metrics.interaction_contribution:10.2%}")
    report.append("")

    # Interpretation
    report.append("INTERPRETATION:")

    if metrics.alpha > 0:
        report.append(f"  ✅ Positive alpha ({metrics.alpha:.2%}): Strategy adds value")
    else:
        report.append(f"  ❌ Negative alpha ({metrics.alpha:.2%}): Strategy destroys value")

    if metrics.information_ratio > 0.5:
        report.append(f"  ✅ Strong IR ({metrics.information_ratio:.2f}): Consistent alpha")
    elif metrics.information_ratio > 0:
        report.append(f"  ⚠️  Moderate IR ({metrics.information_ratio:.2f}): Some consistency")
    else:
        report.append(f"  ❌ Poor IR ({metrics.information_ratio:.2f}): Inconsistent alpha")

    if abs(metrics.beta - 1.0) < 0.2:
        report.append(f"  📊 Beta ≈ 1.0: Market-neutral exposure")
    elif metrics.beta > 1.2:
        report.append(f"  📈 Beta > 1.0: Amplified market exposure")
    elif metrics.beta < 0.8:
        report.append(f"  📉 Beta < 1.0: Defensive positioning")

    report.append("")
    report.append("=" * 60)

    return "\n".join(report)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Generate synthetic data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
    n = len(dates)

    # Benchmark returns (market)
    benchmark_returns = pd.Series(
        np.random.normal(0.0005, 0.01, n),  # 12.5% annual return, 16% vol
        index=dates
    )

    # Strategy returns (with alpha)
    strategy_returns = pd.Series(
        1.5 * benchmark_returns + 0.0002 + np.random.normal(0, 0.005, n),  # Beta=1.5, Alpha=5%
        index=dates
    )

    # Create attributor
    attributor = PerformanceAttributor()

    # Analyze
    metrics = attributor.analyze_strategy(strategy_returns, benchmark_returns)

    # Print report
    report = create_performance_report(strategy_returns, benchmark_returns, "Test Strategy")
    print("\n" + report)

    # Rolling attribution
    print("\n" + "="*60)
    print("ROLLING ATTRIBUTION (Last 5 Periods)")
    print("="*60)

    rolling = attributor.rolling_attribution(strategy_returns, benchmark_returns, window=60)
    print(rolling.tail().to_string(index=False))
