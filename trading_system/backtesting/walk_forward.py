"""
Walk-Forward Analysis Framework

Walk-forward analysis is a robust backtesting methodology that prevents overfitting
by simulating how a strategy would perform in real-time trading with periodic
re-optimization.

Key concepts:
- In-sample (IS) period: Training/optimization period
- Out-of-sample (OOS) period: Testing period (walk-forward)
- Anchored vs. Rolling windows
- Re-optimization frequency

Mathematical Framework:
=======================
For T total observations, divide into:
- IS window: n_train observations
- OOS window: n_test observations

Process:
1. Optimize strategy parameters on IS[t:t+n_train]
2. Test optimized parameters on OOS[t+n_train:t+n_train+n_test]
3. Slide window forward by n_step
4. Repeat until end of data

Efficiency Ratio (ER):
ER = OOS_Sharpe / IS_Sharpe

ER < 0.5 indicates severe overfitting
ER > 0.7 indicates robust strategy

Author: Energy Trading Quant Team
Date: 2025-11-12
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

from .backtesting_engine import BacktestingEngine, BacktestConfig

logger = logging.getLogger(__name__)


class WindowType:
    """Window types for walk-forward analysis."""
    ANCHORED = "anchored"  # Fixed start, expanding window
    ROLLING = "rolling"    # Fixed size, rolling window


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward analysis."""

    # Window configuration
    train_period_days: int = 180  # 6 months training
    test_period_days: int = 60    # 2 months testing
    step_days: int = 30            # Move forward 1 month each step

    window_type: str = WindowType.ROLLING  # "anchored" or "rolling"

    # Optimization
    optimization_metric: str = "sharpe_ratio"  # Metric to optimize
    n_trials: int = 50                          # Number of parameter combinations to try

    # Parallel processing
    n_jobs: int = 4  # Number of parallel processes

    # Minimum data requirements
    min_train_samples: int = 100
    min_test_samples: int = 20

    # Backtesting config
    backtest_config: BacktestConfig = field(default_factory=BacktestConfig)


@dataclass
class WalkForwardPeriod:
    """Represents a single walk-forward period."""
    period_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    optimal_params: Dict[str, Any]
    train_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    optimization_trials: List[Dict] = field(default_factory=list)


class WalkForwardAnalyzer:
    """
    Walk-Forward Analysis Framework.

    This class implements a robust walk-forward testing methodology that prevents
    overfitting by:
    1. Optimizing strategy parameters on in-sample data
    2. Testing on out-of-sample data
    3. Repeating the process across multiple time periods

    Example:
        >>> config = WalkForwardConfig(
        ...     train_period_days=180,
        ...     test_period_days=60,
        ...     step_days=30
        ... )
        >>> wfa = WalkForwardAnalyzer(config)
        >>> results = wfa.run(price_data, strategy_func, param_grid)
        >>> print(f"Efficiency Ratio: {results['efficiency_ratio']:.2f}")
    """

    def __init__(self, config: WalkForwardConfig):
        """
        Initialize Walk-Forward Analyzer.

        Args:
            config: Walk-forward configuration
        """
        self.config = config
        self.periods: List[WalkForwardPeriod] = []

        logger.info(f"WalkForwardAnalyzer initialized:")
        logger.info(f"  Window type: {config.window_type}")
        logger.info(f"  Train period: {config.train_period_days} days")
        logger.info(f"  Test period: {config.test_period_days} days")
        logger.info(f"  Step size: {config.step_days} days")

    def _generate_periods(self, data: pd.DataFrame) -> List[Tuple[datetime, datetime, datetime, datetime]]:
        """
        Generate walk-forward periods.

        Args:
            data: Input data with datetime index

        Returns:
            periods: List of (train_start, train_end, test_start, test_end) tuples
        """
        periods = []

        start_date = data.index[0]
        end_date = data.index[-1]

        train_delta = timedelta(days=self.config.train_period_days)
        test_delta = timedelta(days=self.config.test_period_days)
        step_delta = timedelta(days=self.config.step_days)

        current_start = start_date
        period_id = 0

        while True:
            # Training period
            if self.config.window_type == WindowType.ANCHORED:
                train_start = start_date  # Fixed start for anchored
            else:
                train_start = current_start  # Moving start for rolling

            train_end = current_start + train_delta

            # Test period
            test_start = train_end
            test_end = test_start + test_delta

            # Check if we have enough data
            if test_end > end_date:
                break

            # Verify minimum sample requirements
            train_data = data[(data.index >= train_start) & (data.index < train_end)]
            test_data = data[(data.index >= test_start) & (data.index < test_end)]

            if len(train_data) >= self.config.min_train_samples and \
               len(test_data) >= self.config.min_test_samples:
                periods.append((train_start, train_end, test_start, test_end))
                period_id += 1

            # Move forward
            current_start += step_delta

        logger.info(f"Generated {len(periods)} walk-forward periods")
        return periods

    def _optimize_parameters(self, data: pd.DataFrame, strategy_func: Callable,
                            param_grid: Dict[str, List[Any]],
                            train_start: datetime, train_end: datetime) -> Tuple[Dict, List[Dict]]:
        """
        Optimize strategy parameters on training data.

        Args:
            data: Price data
            strategy_func: Strategy function that takes (data, **params) and returns signals
            param_grid: Dictionary of parameter names and their possible values
            train_start: Training period start
            train_end: Training period end

        Returns:
            best_params: Optimal parameters
            all_trials: List of all optimization trials
        """
        # Extract training data
        train_data = data[(data.index >= train_start) & (data.index < train_end)]

        # Generate parameter combinations
        from itertools import product

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(product(*param_values))

        # Sample if too many combinations
        if len(param_combinations) > self.config.n_trials:
            import random
            param_combinations = random.sample(param_combinations, self.config.n_trials)

        logger.info(f"Testing {len(param_combinations)} parameter combinations")

        # Test each combination
        trials = []
        best_metric = -np.inf
        best_params = {}

        for i, param_values in enumerate(param_combinations):
            params = dict(zip(param_names, param_values))

            try:
                # Generate signals with these parameters
                signals = strategy_func(train_data, **params)

                # Run backtest
                engine = BacktestingEngine(self.config.backtest_config)
                results = engine.run(train_data, signals, symbol='ENERGY')
                metrics = engine.get_performance_metrics()

                # Evaluate based on optimization metric
                metric_value = metrics.get(self.config.optimization_metric, 0)

                trial = {
                    'params': params,
                    'metrics': metrics,
                    self.config.optimization_metric: metric_value
                }
                trials.append(trial)

                # Update best
                if metric_value > best_metric:
                    best_metric = metric_value
                    best_params = params.copy()

                if (i + 1) % 10 == 0:
                    logger.debug(f"  Completed {i+1}/{len(param_combinations)} trials")

            except Exception as e:
                logger.warning(f"Failed to test params {params}: {str(e)}")
                continue

        logger.info(f"Optimization complete. Best {self.config.optimization_metric}: {best_metric:.4f}")
        logger.info(f"Best parameters: {best_params}")

        return best_params, trials

    def _test_parameters(self, data: pd.DataFrame, strategy_func: Callable,
                        params: Dict[str, Any],
                        test_start: datetime, test_end: datetime) -> Dict[str, float]:
        """
        Test strategy parameters on out-of-sample data.

        Args:
            data: Price data
            strategy_func: Strategy function
            params: Parameters to test
            test_start: Test period start
            test_end: Test period end

        Returns:
            metrics: Performance metrics on test data
        """
        # Extract test data
        test_data = data[(data.index >= test_start) & (data.index < test_end)]

        # Generate signals with optimal parameters
        signals = strategy_func(test_data, **params)

        # Run backtest
        engine = BacktestingEngine(self.config.backtest_config)
        results = engine.run(test_data, signals, symbol='ENERGY')
        metrics = engine.get_performance_metrics()

        return metrics

    def run(self, data: pd.DataFrame, strategy_func: Callable,
            param_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
        """
        Run complete walk-forward analysis.

        Args:
            data: Price data with datetime index
            strategy_func: Strategy function that takes (data, **params) and returns signals DataFrame
            param_grid: Parameter grid for optimization
                       Example: {'lookback': [20, 30, 50], 'threshold': [1.5, 2.0, 2.5]}

        Returns:
            results: Dictionary containing:
                - periods: List of WalkForwardPeriod objects
                - aggregate_metrics: Combined metrics across all periods
                - efficiency_ratio: OOS Sharpe / IS Sharpe
                - period_summary: DataFrame with per-period results
        """
        logger.info("="*80)
        logger.info("Starting Walk-Forward Analysis")
        logger.info("="*80)

        # Generate periods
        period_tuples = self._generate_periods(data)

        if len(period_tuples) == 0:
            raise ValueError("No valid walk-forward periods generated. Check data length and config.")

        # Run walk-forward for each period
        for i, (train_start, train_end, test_start, test_end) in enumerate(period_tuples):
            logger.info(f"\n{'='*80}")
            logger.info(f"Period {i+1}/{len(period_tuples)}")
            logger.info(f"{'='*80}")
            logger.info(f"Train: {train_start.date()} to {train_end.date()} ({(train_end-train_start).days} days)")
            logger.info(f"Test:  {test_start.date()} to {test_end.date()} ({(test_end-test_start).days} days)")

            # Optimize on training data
            logger.info("\nOptimizing parameters...")
            best_params, trials = self._optimize_parameters(
                data, strategy_func, param_grid, train_start, train_end
            )

            # Get training metrics
            train_metrics = next(
                (t['metrics'] for t in trials if t['params'] == best_params),
                {}
            )

            # Test on out-of-sample data
            logger.info("\nTesting on out-of-sample data...")
            test_metrics = self._test_parameters(
                data, strategy_func, best_params, test_start, test_end
            )

            # Create period object
            period = WalkForwardPeriod(
                period_id=i,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                optimal_params=best_params,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                optimization_trials=trials
            )

            self.periods.append(period)

            # Log results
            train_sharpe = train_metrics.get('sharpe_ratio', 0)
            test_sharpe = test_metrics.get('sharpe_ratio', 0)
            logger.info(f"\nResults:")
            logger.info(f"  Train Sharpe: {train_sharpe:.3f}")
            logger.info(f"  Test Sharpe:  {test_sharpe:.3f}")
            logger.info(f"  Efficiency Ratio: {test_sharpe/train_sharpe:.3f}" if train_sharpe != 0 else "  Efficiency Ratio: N/A")

        # Aggregate results
        results = self._aggregate_results()

        logger.info(f"\n{'='*80}")
        logger.info("Walk-Forward Analysis Complete")
        logger.info(f"{'='*80}")
        logger.info(f"Efficiency Ratio: {results['efficiency_ratio']:.3f}")
        logger.info(f"Average OOS Sharpe: {results['aggregate_metrics']['avg_test_sharpe']:.3f}")
        logger.info(f"Average OOS Return: {results['aggregate_metrics']['avg_test_return']:.2%}")

        return results

    def _aggregate_results(self) -> Dict[str, Any]:
        """
        Aggregate results across all walk-forward periods.

        Returns:
            results: Dictionary with aggregate metrics and analysis
        """
        if not self.periods:
            return {}

        # Extract metrics
        train_sharpes = [p.train_metrics.get('sharpe_ratio', 0) for p in self.periods]
        test_sharpes = [p.test_metrics.get('sharpe_ratio', 0) for p in self.periods]
        train_returns = [p.train_metrics.get('total_return', 0) for p in self.periods]
        test_returns = [p.test_metrics.get('total_return', 0) for p in self.periods]

        # Calculate efficiency ratio
        avg_train_sharpe = np.mean([s for s in train_sharpes if s != 0])
        avg_test_sharpe = np.mean([s for s in test_sharpes if s != 0])

        efficiency_ratio = avg_test_sharpe / avg_train_sharpe if avg_train_sharpe != 0 else 0

        # Build period summary DataFrame
        summary_data = []
        for p in self.periods:
            summary_data.append({
                'period_id': p.period_id,
                'train_start': p.train_start,
                'train_end': p.train_end,
                'test_start': p.test_start,
                'test_end': p.test_end,
                'train_sharpe': p.train_metrics.get('sharpe_ratio', 0),
                'test_sharpe': p.test_metrics.get('sharpe_ratio', 0),
                'train_return': p.train_metrics.get('total_return', 0),
                'test_return': p.test_metrics.get('total_return', 0),
                'train_max_dd': p.train_metrics.get('max_drawdown', 0),
                'test_max_dd': p.test_metrics.get('max_drawdown', 0),
                'optimal_params': str(p.optimal_params)
            })

        period_summary = pd.DataFrame(summary_data)

        # Aggregate metrics
        aggregate_metrics = {
            'avg_train_sharpe': avg_train_sharpe,
            'avg_test_sharpe': avg_test_sharpe,
            'avg_train_return': np.mean(train_returns),
            'avg_test_return': np.mean(test_returns),
            'median_test_sharpe': np.median(test_sharpes),
            'std_test_sharpe': np.std(test_sharpes),
            'min_test_sharpe': np.min(test_sharpes),
            'max_test_sharpe': np.max(test_sharpes),
            'positive_periods_pct': sum(1 for r in test_returns if r > 0) / len(test_returns),
            'n_periods': len(self.periods)
        }

        results = {
            'periods': self.periods,
            'period_summary': period_summary,
            'aggregate_metrics': aggregate_metrics,
            'efficiency_ratio': efficiency_ratio
        }

        return results

    def plot_results(self, results: Dict[str, Any], save_path: Optional[str] = None) -> None:
        """
        Plot walk-forward analysis results.

        Args:
            results: Results from run() method
            save_path: Path to save figure (optional)
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.set_style("whitegrid")
        except ImportError:
            logger.warning("matplotlib/seaborn not available, skipping plots")
            return

        period_summary = results['period_summary']

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Sharpe Ratio comparison
        ax = axes[0, 0]
        x = range(len(period_summary))
        ax.plot(x, period_summary['train_sharpe'], 'o-', label='In-Sample', linewidth=2)
        ax.plot(x, period_summary['test_sharpe'], 's-', label='Out-of-Sample', linewidth=2)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        ax.set_xlabel('Period')
        ax.set_ylabel('Sharpe Ratio')
        ax.set_title('Sharpe Ratio: In-Sample vs Out-of-Sample')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Returns comparison
        ax = axes[0, 1]
        ax.plot(x, period_summary['train_return'] * 100, 'o-', label='In-Sample', linewidth=2)
        ax.plot(x, period_summary['test_return'] * 100, 's-', label='Out-of-Sample', linewidth=2)
        ax.axhline(y=0, color='r', linestyle='--', alpha=0.3)
        ax.set_xlabel('Period')
        ax.set_ylabel('Return (%)')
        ax.set_title('Returns: In-Sample vs Out-of-Sample')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Drawdown comparison
        ax = axes[1, 0]
        ax.plot(x, period_summary['train_max_dd'] * 100, 'o-', label='In-Sample', linewidth=2)
        ax.plot(x, period_summary['test_max_dd'] * 100, 's-', label='Out-of-Sample', linewidth=2)
        ax.set_xlabel('Period')
        ax.set_ylabel('Max Drawdown (%)')
        ax.set_title('Maximum Drawdown: In-Sample vs Out-of-Sample')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Efficiency summary
        ax = axes[1, 1]
        metrics = results['aggregate_metrics']
        data_text = [
            f"Efficiency Ratio: {results['efficiency_ratio']:.3f}",
            f"",
            f"Average OOS Sharpe: {metrics['avg_test_sharpe']:.3f}",
            f"Average OOS Return: {metrics['avg_test_return']:.2%}",
            f"Median OOS Sharpe: {metrics['median_test_sharpe']:.3f}",
            f"",
            f"Positive Periods: {metrics['positive_periods_pct']:.1%}",
            f"Number of Periods: {metrics['n_periods']}",
            f"",
            f"Interpretation:",
            f"ER > 0.7: Robust ✓" if results['efficiency_ratio'] > 0.7 else f"ER < 0.5: Overfitting ⚠" if results['efficiency_ratio'] < 0.5 else f"ER 0.5-0.7: Acceptable"
        ]

        ax.text(0.1, 0.9, '\n'.join(data_text), transform=ax.transAxes,
                fontsize=11, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.axis('off')
        ax.set_title('Walk-Forward Summary')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")

        plt.show()


def create_simple_strategy(lookback: int = 20, threshold: float = 2.0):
    """
    Example strategy function for walk-forward testing.

    This is a simple mean reversion strategy based on z-scores.

    Args:
        lookback: Lookback period for mean/std calculation
        threshold: Z-score threshold for entry

    Returns:
        strategy_func: Function that takes data and returns signals
    """
    def strategy_func(data: pd.DataFrame, **params) -> pd.DataFrame:
        """Generate signals based on z-score."""
        # Extract parameters (allow override)
        lb = params.get('lookback', lookback)
        th = params.get('threshold', threshold)

        # Calculate z-score
        prices = data['price'] if 'price' in data.columns else data.iloc[:, 0]
        rolling_mean = prices.rolling(window=lb).mean()
        rolling_std = prices.rolling(window=lb).std()
        z_score = (prices - rolling_mean) / rolling_std

        # Generate signals
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0.0

        signals.loc[z_score < -th, 'signal'] = 1.0   # Buy when oversold
        signals.loc[z_score > th, 'signal'] = -1.0   # Sell when overbought
        signals.loc[abs(z_score) < 0.5, 'signal'] = 0.0  # Exit when near mean

        # Forward fill to maintain positions
        signals['signal'] = signals['signal'].replace(0, np.nan).fillna(method='ffill').fillna(0)

        return signals

    return strategy_func
