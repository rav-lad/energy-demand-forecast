"""
Integrated Backtesting Framework with Performance Attribution

This module combines:
1. Core backtesting engine with realistic costs
2. Performance attribution analysis (alpha/beta decomposition)
3. Risk-adjusted metrics calculation
4. Comprehensive reporting

The integrated approach allows for:
- Understanding sources of returns (market vs skill)
- Risk-adjusted performance evaluation
- Factor-based attribution
- Professional reporting for stakeholders

Author: Energy Trading Quant Team
Date: 2025-11-12
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..analytics.performance_attribution import AttributionMetrics, PerformanceAttributor
from ..risk_management.risk_manager import RiskLimits, RiskManager
from .backtesting_engine import BacktestConfig, BacktestingEngine

logger = logging.getLogger(__name__)


@dataclass
class IntegratedBacktestConfig:
    """Configuration for integrated backtesting."""

    # Core backtest config
    backtest_config: BacktestConfig

    # Risk management
    enable_risk_management: bool = True
    risk_limits: Optional[RiskLimits] = None

    # Performance attribution
    benchmark_returns: Optional[pd.Series] = None
    factor_returns: Optional[pd.DataFrame] = None  # Multiple factors for attribution

    # Reporting
    generate_report: bool = True
    report_path: Optional[str] = None


class IntegratedBacktester:
    """
    Integrated Backtesting Framework.

    Combines backtesting, risk management, and performance attribution
    to provide comprehensive strategy evaluation.

    Features:
    - Realistic transaction cost modeling
    - Real-time risk management enforcement
    - Alpha/beta decomposition
    - Factor-based attribution
    - Professional reporting

    Example:
        >>> config = IntegratedBacktestConfig(
        ...     backtest_config=BacktestConfig(),
        ...     enable_risk_management=True
        ... )
        >>> backtester = IntegratedBacktester(config)
        >>> results = backtester.run(
        ...     price_data=prices,
        ...     signals=signals,
        ...     benchmark_returns=benchmark
        ... )
        >>> print(f"Alpha: {results['attribution']['alpha']:.2%}")
    """

    def __init__(self, config: IntegratedBacktestConfig):
        """
        Initialize integrated backtester.

        Args:
            config: Integrated backtest configuration
        """
        self.config = config

        # Initialize components
        self.engine = BacktestingEngine(config.backtest_config)

        if config.enable_risk_management:
            risk_limits = config.risk_limits or RiskLimits()
            self.risk_manager = RiskManager(risk_limits)
        else:
            self.risk_manager = None

        self.attributor = PerformanceAttributor()

        logger.info("IntegratedBacktester initialized")
        logger.info(f"  Risk management: {config.enable_risk_management}")
        logger.info(f"  Attribution: {config.benchmark_returns is not None}")

    def run(
        self,
        price_data: pd.DataFrame,
        signals: pd.DataFrame,
        symbol: str = "ENERGY",
        benchmark_returns: Optional[pd.Series] = None,
        factor_returns: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        Run integrated backtest with attribution.

        Args:
            price_data: Historical price data
            signals: Trading signals from strategy
            symbol: Trading symbol
            benchmark_returns: Benchmark returns for attribution (optional)
            factor_returns: Factor returns for multi-factor attribution (optional)

        Returns:
            results: Dictionary containing:
                - backtest_results: Core backtest results
                - performance_metrics: Performance metrics
                - attribution: Performance attribution analysis
                - risk_metrics: Risk management metrics
                - report: Summary report (if enabled)
        """
        logger.info("=" * 80)
        logger.info("Integrated Backtesting with Performance Attribution")
        logger.info("=" * 80)

        # Phase 1: Run core backtest
        logger.info("\n[1/4] Running backtest...")
        backtest_results = self.engine.run(price_data, signals, symbol)
        performance_metrics = self.engine.get_performance_metrics()

        logger.info(f"  Total return: {performance_metrics['total_return']:.2%}")
        logger.info(f"  Sharpe ratio: {performance_metrics['sharpe_ratio']:.3f}")
        logger.info(f"  Max drawdown: {performance_metrics['max_drawdown']:.2%}")

        # Phase 2: Risk analysis
        logger.info("\n[2/4] Analyzing risk metrics...")
        risk_metrics = self._analyze_risk(backtest_results)

        # Phase 3: Performance attribution
        logger.info("\n[3/4] Computing performance attribution...")
        attribution = self._compute_attribution(
            backtest_results,
            benchmark_returns or self.config.benchmark_returns,
            factor_returns or self.config.factor_returns,
        )

        # Phase 4: Generate report
        logger.info("\n[4/4] Generating report...")
        report = self._generate_report(performance_metrics, attribution, risk_metrics)

        results = {
            "backtest_results": backtest_results,
            "performance_metrics": performance_metrics,
            "attribution": attribution,
            "risk_metrics": risk_metrics,
            "report": report,
            "equity_curve": backtest_results,
            "trades": self.engine.trades,
        }

        logger.info("\n" + "=" * 80)
        logger.info("Integrated Backtesting Complete")
        logger.info("=" * 80)

        # Print summary
        if report:
            print("\n" + report)

        return results

    def _analyze_risk(self, backtest_results: pd.DataFrame) -> Dict:
        """
        Analyze risk metrics from backtest results.

        Args:
            backtest_results: DataFrame with equity curve

        Returns:
            risk_metrics: Dictionary of risk metrics
        """
        if "returns" not in backtest_results.columns:
            backtest_results["returns"] = backtest_results["equity"].pct_change()

        returns = backtest_results["returns"].dropna()

        risk_metrics = {}

        # VaR/CVaR
        if self.risk_manager:
            var_95 = self.risk_manager.calculate_var(returns, 0.95, method="historical")
            cvar_95 = self.risk_manager.calculate_cvar(returns, 0.95)

            risk_metrics["var_95"] = var_95
            risk_metrics["cvar_95"] = cvar_95

        # Volatility metrics
        risk_metrics["daily_volatility"] = returns.std()
        risk_metrics["annualized_volatility"] = returns.std() * np.sqrt(252)

        # Downside risk
        downside_returns = returns[returns < 0]
        risk_metrics["downside_deviation"] = downside_returns.std() * np.sqrt(252)

        # Sortino ratio
        excess_return = returns.mean() * 252
        sortino = (
            excess_return / risk_metrics["downside_deviation"]
            if risk_metrics["downside_deviation"] > 0
            else 0
        )
        risk_metrics["sortino_ratio"] = sortino

        # Skewness and kurtosis
        risk_metrics["skewness"] = returns.skew()
        risk_metrics["kurtosis"] = returns.kurtosis()

        logger.info(f"  VaR (95%): {risk_metrics.get('var_95', 0):.2%}")
        logger.info(f"  CVaR (95%): {risk_metrics.get('cvar_95', 0):.2%}")
        logger.info(f"  Sortino Ratio: {risk_metrics['sortino_ratio']:.3f}")

        return risk_metrics

    def _compute_attribution(
        self,
        backtest_results: pd.DataFrame,
        benchmark_returns: Optional[pd.Series],
        factor_returns: Optional[pd.DataFrame],
    ) -> Dict:
        """
        Compute performance attribution.

        Args:
            backtest_results: Backtest results
            benchmark_returns: Benchmark returns
            factor_returns: Factor returns

        Returns:
            attribution: Dictionary with attribution metrics
        """
        attribution = {}

        if "returns" not in backtest_results.columns:
            backtest_results["returns"] = backtest_results["equity"].pct_change()

        strategy_returns = backtest_results["returns"].dropna()

        # Alpha/Beta decomposition
        if benchmark_returns is not None:
            logger.info("  Computing alpha/beta decomposition...")

            # Align data
            common_index = strategy_returns.index.intersection(benchmark_returns.index)
            strategy_aligned = strategy_returns.loc[common_index]
            benchmark_aligned = benchmark_returns.loc[common_index]

            # Calculate alpha/beta
            alpha, beta, r_squared = self.attributor.calculate_alpha_beta(
                strategy_aligned, benchmark_aligned
            )

            # Information Ratio
            info_ratio = self.attributor.calculate_information_ratio(
                strategy_aligned, benchmark_aligned
            )

            attribution["alpha"] = alpha
            attribution["beta"] = beta
            attribution["r_squared"] = r_squared
            attribution["information_ratio"] = info_ratio

            logger.info(f"    Alpha: {alpha:.2%} (annualized)")
            logger.info(f"    Beta: {beta:.3f}")
            logger.info(f"    R²: {r_squared:.3f}")
            logger.info(f"    Information Ratio: {info_ratio:.3f}")

        else:
            logger.info("  No benchmark provided, skipping alpha/beta decomposition")

        # Factor attribution
        if factor_returns is not None and len(factor_returns.columns) > 0:
            logger.info("  Computing factor attribution...")

            # Align data
            common_index = strategy_returns.index.intersection(factor_returns.index)
            strategy_aligned = strategy_returns.loc[common_index]
            factors_aligned = factor_returns.loc[common_index]

            # Factor attribution
            factor_attr = self.attributor.factor_attribution(strategy_aligned, factors_aligned)

            attribution["factor_loadings"] = factor_attr["factor_loadings"]
            attribution["factor_contributions"] = factor_attr["factor_contributions"]
            attribution["factor_r_squared"] = factor_attr["r_squared"]

            logger.info(f"    Factor R²: {factor_attr['r_squared']:.3f}")
            for factor, loading in factor_attr["factor_loadings"].items():
                contrib = factor_attr["factor_contributions"][factor]
                logger.info(f"    {factor}: loading={loading:.3f}, contribution={contrib:.2%}")

        return attribution

    def _generate_report(
        self, performance_metrics: Dict, attribution: Dict, risk_metrics: Dict
    ) -> str:
        """
        Generate comprehensive backtest report.

        Args:
            performance_metrics: Performance metrics
            attribution: Attribution metrics
            risk_metrics: Risk metrics

        Returns:
            report: Formatted report string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("INTEGRATED BACKTEST REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Performance Summary
        lines.append("PERFORMANCE SUMMARY")
        lines.append("-" * 80)
        lines.append(f"Total Return:           {performance_metrics['total_return']:>10.2%}")
        lines.append(f"Annualized Return:      {performance_metrics['annualized_return']:>10.2%}")
        lines.append(
            f"Annualized Volatility:  {performance_metrics['annualized_volatility']:>10.2%}"
        )
        lines.append(f"Sharpe Ratio:           {performance_metrics['sharpe_ratio']:>10.3f}")
        lines.append(f"Sortino Ratio:          {risk_metrics.get('sortino_ratio', 0):>10.3f}")
        lines.append(f"Maximum Drawdown:       {performance_metrics['max_drawdown']:>10.2%}")
        lines.append(f"Win Rate:               {performance_metrics['win_rate']:>10.2%}")
        lines.append(f"Profit Factor:          {performance_metrics['profit_factor']:>10.3f}")
        lines.append(f"Total Trades:           {performance_metrics['total_trades']:>10.0f}")
        lines.append("")

        # Attribution
        if attribution:
            lines.append("PERFORMANCE ATTRIBUTION")
            lines.append("-" * 80)

            if "alpha" in attribution:
                lines.append(f"Alpha (annualized):     {attribution['alpha']:>10.2%}")
                lines.append(f"Beta:                   {attribution['beta']:>10.3f}")
                lines.append(f"R-squared:              {attribution['r_squared']:>10.3f}")
                lines.append(f"Information Ratio:      {attribution['information_ratio']:>10.3f}")
                lines.append("")

            if "factor_loadings" in attribution:
                lines.append("Factor Loadings:")
                for factor, loading in attribution["factor_loadings"].items():
                    contrib = attribution["factor_contributions"][factor]
                    lines.append(f"  {factor:20s} {loading:>8.3f}  (contrib: {contrib:>7.2%})")
                lines.append(f"Factor R-squared:       {attribution['factor_r_squared']:>10.3f}")
                lines.append("")

        # Risk Metrics
        lines.append("RISK METRICS")
        lines.append("-" * 80)
        lines.append(f"Daily Volatility:       {risk_metrics['daily_volatility']:>10.4f}")
        lines.append(f"Downside Deviation:     {risk_metrics['downside_deviation']:>10.2%}")

        if "var_95" in risk_metrics:
            lines.append(f"VaR (95%):              {risk_metrics['var_95']:>10.2%}")
            lines.append(f"CVaR (95%):             {risk_metrics['cvar_95']:>10.2%}")

        lines.append(f"Skewness:               {risk_metrics['skewness']:>10.3f}")
        lines.append(f"Kurtosis:               {risk_metrics['kurtosis']:>10.3f}")
        lines.append("")

        # Transaction Costs
        lines.append("TRANSACTION COSTS")
        lines.append("-" * 80)
        lines.append(
            f"Total Commission:       {performance_metrics['total_commission']:>10,.2f} EUR"
        )
        lines.append(f"Total Slippage:         {performance_metrics['total_slippage']:>10,.2f} EUR")
        total_costs = (
            performance_metrics["total_commission"] + performance_metrics["total_slippage"]
        )
        lines.append(f"Total Costs:            {total_costs:>10,.2f} EUR")
        cost_pct = total_costs / self.config.backtest_config.initial_capital
        lines.append(f"Cost as % of Capital:   {cost_pct:>10.2%}")
        lines.append("")

        lines.append("=" * 80)

        report = "\n".join(lines)

        # Save report if path specified
        if self.config.report_path:
            try:
                with open(self.config.report_path, "w") as f:
                    f.write(report)
                logger.info(f"Report saved to {self.config.report_path}")
            except Exception as e:
                logger.warning(f"Failed to save report: {str(e)}")

        return report

    def compare_strategies(
        self,
        strategies: Dict[str, Tuple],
        price_data: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Compare multiple strategies side-by-side.

        Args:
            strategies: Dict of {name: (signals, params)} tuples
            price_data: Historical price data
            benchmark_returns: Benchmark returns (optional)

        Returns:
            comparison: DataFrame with strategy comparison
        """
        logger.info("=" * 80)
        logger.info("Strategy Comparison")
        logger.info("=" * 80)

        results = []

        for name, (signals, params) in strategies.items():
            logger.info(f"\nBacktesting strategy: {name}")

            # Run backtest
            strategy_results = self.run(
                price_data=price_data, signals=signals, benchmark_returns=benchmark_returns
            )

            # Extract key metrics
            perf = strategy_results["performance_metrics"]
            attr = strategy_results["attribution"]
            risk = strategy_results["risk_metrics"]

            row = {
                "strategy": name,
                "total_return": perf["total_return"],
                "sharpe_ratio": perf["sharpe_ratio"],
                "sortino_ratio": risk.get("sortino_ratio", 0),
                "max_drawdown": perf["max_drawdown"],
                "win_rate": perf["win_rate"],
                "alpha": attr.get("alpha", np.nan),
                "beta": attr.get("beta", np.nan),
                "information_ratio": attr.get("information_ratio", np.nan),
                "var_95": risk.get("var_95", np.nan),
                "total_trades": perf["total_trades"],
            }

            results.append(row)

        comparison_df = pd.DataFrame(results)
        comparison_df.set_index("strategy", inplace=True)

        logger.info("\n" + "=" * 80)
        logger.info("STRATEGY COMPARISON SUMMARY")
        logger.info("=" * 80)
        print("\n" + comparison_df.to_string())

        return comparison_df
