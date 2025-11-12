"""
Monte Carlo Simulation for Trading Strategy Robustness Testing

Monte Carlo methods provide statistical confidence in strategy performance by:
1. Simulating thousands of alternative return paths
2. Assessing sensitivity to parameter variations
3. Estimating probability distributions of outcomes
4. Stress testing under extreme scenarios

Key Methods:
============

1. Bootstrap Resampling
   - Randomly resample historical returns with replacement
   - Preserves empirical distribution
   - Tests robustness to different market sequences

2. Parametric Simulation
   - Fit returns to statistical distribution (e.g., t-distribution)
   - Generate synthetic returns from fitted model
   - Tests sensitivity to distributional assumptions

3. Block Bootstrap
   - Resample blocks of consecutive returns
   - Preserves serial correlation structure
   - More realistic for time series with autocorrelation

4. Parameter Perturbation
   - Add random noise to strategy parameters
   - Tests parameter stability
   - Identifies overfitting

Confidence Intervals:
====================
For n simulations, (1-α)% confidence interval:
[Percentile(α/2), Percentile(1-α/2)]

Example: 95% CI = [2.5th percentile, 97.5th percentile]

Author: Energy Trading Quant Team
Date: 2025-11-12
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import stats
import warnings

from .backtesting_engine import BacktestingEngine, BacktestConfig

logger = logging.getLogger(__name__)


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo simulation."""

    # Simulation settings
    n_simulations: int = 1000      # Number of Monte Carlo runs
    confidence_level: float = 0.95  # For confidence intervals

    # Resampling method
    method: str = "bootstrap"  # "bootstrap", "parametric", "block_bootstrap", "parameter_perturbation"
    block_size: int = 20       # For block bootstrap (days)

    # Parameter perturbation
    param_noise_pct: float = 0.1  # ±10% parameter variation

    # Random seed for reproducibility
    random_seed: Optional[int] = 42

    # Backtesting config
    backtest_config: BacktestConfig = field(default_factory=BacktestConfig)


@dataclass
class MonteCarloResults:
    """Results from Monte Carlo simulation."""

    # Simulation data
    n_simulations: int
    method: str

    # Performance distributions
    returns: np.ndarray
    sharpe_ratios: np.ndarray
    max_drawdowns: np.ndarray
    win_rates: np.ndarray

    # Statistical summary
    mean_return: float
    median_return: float
    std_return: float
    ci_lower: float
    ci_upper: float

    mean_sharpe: float
    median_sharpe: float
    ci_sharpe_lower: float
    ci_sharpe_upper: float

    mean_drawdown: float
    median_drawdown: float
    worst_drawdown: float

    # Risk metrics
    prob_positive_return: float
    prob_sharpe_above_1: float
    prob_drawdown_below_20pct: float

    # Original backtest comparison
    original_return: float
    original_sharpe: float

    # Percentile rankings
    return_percentile: float  # Where original falls in distribution
    sharpe_percentile: float

    def __str__(self) -> str:
        """String representation of results."""
        ci_pct = 95  # Assuming 95% CI
        return f"""
Monte Carlo Simulation Results
{'='*60}
Method: {self.method}
Simulations: {self.n_simulations}

Returns:
  Mean:     {self.mean_return:8.2%}
  Median:   {self.median_return:8.2%}
  Std Dev:  {self.std_return:8.2%}
  {ci_pct}% CI: [{self.ci_lower:7.2%}, {self.ci_upper:7.2%}]

Sharpe Ratio:
  Mean:     {self.mean_sharpe:8.3f}
  Median:   {self.median_sharpe:8.3f}
  {ci_pct}% CI: [{self.ci_sharpe_lower:7.3f}, {self.ci_sharpe_upper:7.3f}]

Max Drawdown:
  Mean:     {self.mean_drawdown:8.2%}
  Median:   {self.median_drawdown:8.2%}
  Worst:    {self.worst_drawdown:8.2%}

Risk Metrics:
  P(Return > 0):     {self.prob_positive_return:6.1%}
  P(Sharpe > 1.0):   {self.prob_sharpe_above_1:6.1%}
  P(DD < 20%):       {self.prob_drawdown_below_20pct:6.1%}

Original Backtest:
  Return:   {self.original_return:8.2%}  (Percentile: {self.return_percentile:5.1f})
  Sharpe:   {self.original_sharpe:8.3f}  (Percentile: {self.sharpe_percentile:5.1f})
{'='*60}
"""


class MonteCarloSimulator:
    """
    Monte Carlo Simulator for Trading Strategy Robustness Testing.

    This class implements various Monte Carlo methods to:
    - Assess statistical confidence in strategy performance
    - Test sensitivity to market conditions
    - Identify overfitting through parameter perturbation
    - Generate confidence intervals for key metrics

    Example:
        >>> config = MonteCarloConfig(n_simulations=1000, method='bootstrap')
        >>> simulator = MonteCarloSimulator(config)
        >>> results = simulator.run(returns, strategy_func, price_data, params)
        >>> print(f"95% CI Sharpe: [{results.ci_sharpe_lower:.2f}, {results.ci_sharpe_upper:.2f}]")
    """

    def __init__(self, config: MonteCarloConfig):
        """
        Initialize Monte Carlo simulator.

        Args:
            config: Monte Carlo configuration
        """
        self.config = config

        if config.random_seed is not None:
            np.random.seed(config.random_seed)

        logger.info(f"MonteCarloSimulator initialized:")
        logger.info(f"  Method: {config.method}")
        logger.info(f"  Simulations: {config.n_simulations}")
        logger.info(f"  Confidence: {config.confidence_level*100:.0f}%")

    def run(self, original_returns: pd.Series, strategy_func: Callable,
            price_data: pd.DataFrame, params: Dict,
            original_backtest_results: Optional[Dict] = None) -> MonteCarloResults:
        """
        Run Monte Carlo simulation.

        Args:
            original_returns: Original strategy returns (from backtest)
            strategy_func: Strategy function that generates signals
            price_data: Historical price data
            params: Strategy parameters
            original_backtest_results: Results from original backtest (optional)

        Returns:
            results: MonteCarloResults object with simulation results
        """
        logger.info("="*80)
        logger.info("Starting Monte Carlo Simulation")
        logger.info("="*80)

        method_func = self._get_method_func()

        # Run simulations
        sim_returns = []
        sim_sharpe = []
        sim_drawdowns = []
        sim_win_rates = []

        for i in range(self.config.n_simulations):
            if (i + 1) % 100 == 0:
                logger.info(f"Completed {i+1}/{self.config.n_simulations} simulations")

            try:
                # Generate synthetic scenario
                synthetic_data = method_func(original_returns, price_data, params)

                # Run backtest on synthetic data
                metrics = self._run_synthetic_backtest(
                    synthetic_data, strategy_func, params
                )

                sim_returns.append(metrics['total_return'])
                sim_sharpe.append(metrics['sharpe_ratio'])
                sim_drawdowns.append(metrics['max_drawdown'])
                sim_win_rates.append(metrics.get('win_rate', 0))

            except Exception as e:
                logger.warning(f"Simulation {i} failed: {str(e)}")
                continue

        # Convert to arrays
        sim_returns = np.array(sim_returns)
        sim_sharpe = np.array(sim_sharpe)
        sim_drawdowns = np.array(sim_drawdowns)
        sim_win_rates = np.array(sim_win_rates)

        # Calculate statistics
        results = self._calculate_statistics(
            sim_returns, sim_sharpe, sim_drawdowns, sim_win_rates,
            original_backtest_results
        )

        logger.info("\n" + str(results))

        return results

    def _get_method_func(self) -> Callable:
        """Get simulation method function."""
        methods = {
            'bootstrap': self._bootstrap_resample,
            'parametric': self._parametric_simulation,
            'block_bootstrap': self._block_bootstrap,
            'parameter_perturbation': self._parameter_perturbation
        }

        if self.config.method not in methods:
            raise ValueError(f"Unknown method: {self.config.method}")

        return methods[self.config.method]

    def _bootstrap_resample(self, returns: pd.Series, price_data: pd.DataFrame,
                           params: Dict) -> pd.DataFrame:
        """
        Bootstrap resampling: randomly resample returns with replacement.

        Args:
            returns: Original returns
            price_data: Original price data
            params: Strategy parameters

        Returns:
            synthetic_data: Synthetic price data
        """
        n = len(returns)

        # Resample returns with replacement
        resampled_indices = np.random.choice(n, size=n, replace=True)
        resampled_returns = returns.iloc[resampled_indices].values

        # Reconstruct prices from returns
        initial_price = price_data.iloc[0]['price'] if 'price' in price_data.columns else price_data.iloc[0][0]
        synthetic_prices = initial_price * np.cumprod(1 + resampled_returns)

        # Create synthetic DataFrame
        synthetic_data = pd.DataFrame({
            'price': synthetic_prices
        }, index=price_data.index[:n])

        return synthetic_data

    def _block_bootstrap(self, returns: pd.Series, price_data: pd.DataFrame,
                        params: Dict) -> pd.DataFrame:
        """
        Block bootstrap: resample blocks of consecutive returns.

        Preserves serial correlation structure.

        Args:
            returns: Original returns
            price_data: Original price data
            params: Strategy parameters

        Returns:
            synthetic_data: Synthetic price data
        """
        n = len(returns)
        block_size = self.config.block_size

        # Calculate number of blocks needed
        n_blocks = int(np.ceil(n / block_size))

        # Resample blocks
        resampled_returns = []
        for _ in range(n_blocks):
            # Random starting point
            start_idx = np.random.randint(0, n - block_size + 1)
            block = returns.iloc[start_idx:start_idx + block_size].values
            resampled_returns.extend(block)

        # Trim to original length
        resampled_returns = np.array(resampled_returns[:n])

        # Reconstruct prices
        initial_price = price_data.iloc[0]['price'] if 'price' in price_data.columns else price_data.iloc[0][0]
        synthetic_prices = initial_price * np.cumprod(1 + resampled_returns)

        synthetic_data = pd.DataFrame({
            'price': synthetic_prices
        }, index=price_data.index[:n])

        return synthetic_data

    def _parametric_simulation(self, returns: pd.Series, price_data: pd.DataFrame,
                               params: Dict) -> pd.DataFrame:
        """
        Parametric simulation: fit returns to distribution and simulate.

        Uses Student's t-distribution (better captures fat tails than normal).

        Args:
            returns: Original returns
            price_data: Original price data
            params: Strategy parameters

        Returns:
            synthetic_data: Synthetic price data
        """
        # Fit t-distribution to returns
        df_fit, loc_fit, scale_fit = stats.t.fit(returns)

        # Generate synthetic returns
        n = len(returns)
        synthetic_returns = stats.t.rvs(df_fit, loc=loc_fit, scale=scale_fit, size=n)

        # Reconstruct prices
        initial_price = price_data.iloc[0]['price'] if 'price' in price_data.columns else price_data.iloc[0][0]
        synthetic_prices = initial_price * np.cumprod(1 + synthetic_returns)

        synthetic_data = pd.DataFrame({
            'price': synthetic_prices
        }, index=price_data.index[:n])

        return synthetic_data

    def _parameter_perturbation(self, returns: pd.Series, price_data: pd.DataFrame,
                                params: Dict) -> Tuple[pd.DataFrame, Dict]:
        """
        Parameter perturbation: use original data but perturb strategy parameters.

        Args:
            returns: Original returns
            price_data: Original price data
            params: Strategy parameters

        Returns:
            price_data: Original price data (unchanged)
            perturbed_params: Parameters with random noise
        """
        # Add noise to each parameter
        perturbed_params = {}
        for key, value in params.items():
            if isinstance(value, (int, float)):
                noise = np.random.uniform(-self.config.param_noise_pct,
                                         self.config.param_noise_pct)
                perturbed_value = value * (1 + noise)

                # Keep integers as integers
                if isinstance(value, int):
                    perturbed_value = int(round(perturbed_value))

                perturbed_params[key] = perturbed_value
            else:
                perturbed_params[key] = value

        # Return original data with perturbed params
        # Note: Need to modify _run_synthetic_backtest to handle this
        return price_data, perturbed_params

    def _run_synthetic_backtest(self, synthetic_data: pd.DataFrame,
                                strategy_func: Callable, params: Dict) -> Dict:
        """
        Run backtest on synthetic data.

        Args:
            synthetic_data: Synthetic price data (or tuple for parameter perturbation)
            strategy_func: Strategy function
            params: Strategy parameters

        Returns:
            metrics: Performance metrics
        """
        # Handle parameter perturbation case
        if isinstance(synthetic_data, tuple):
            data, perturbed_params = synthetic_data
        else:
            data = synthetic_data
            perturbed_params = params

        # Generate signals
        signals = strategy_func(data, **perturbed_params)

        # Run backtest
        engine = BacktestingEngine(self.config.backtest_config)
        engine.run(data, signals, symbol='ENERGY')
        metrics = engine.get_performance_metrics()

        return metrics

    def _calculate_statistics(self, returns: np.ndarray, sharpe: np.ndarray,
                             drawdowns: np.ndarray, win_rates: np.ndarray,
                             original_results: Optional[Dict]) -> MonteCarloResults:
        """
        Calculate statistical summary of simulation results.

        Args:
            returns: Array of simulated returns
            sharpe: Array of simulated Sharpe ratios
            drawdowns: Array of simulated drawdowns
            win_rates: Array of simulated win rates
            original_results: Original backtest results

        Returns:
            results: MonteCarloResults object
        """
        # Calculate confidence intervals
        alpha = 1 - self.config.confidence_level
        ci_lower_pct = alpha / 2
        ci_upper_pct = 1 - alpha / 2

        # Return statistics
        mean_return = np.mean(returns)
        median_return = np.median(returns)
        std_return = np.std(returns)
        ci_lower = np.percentile(returns, ci_lower_pct * 100)
        ci_upper = np.percentile(returns, ci_upper_pct * 100)

        # Sharpe statistics
        mean_sharpe = np.mean(sharpe)
        median_sharpe = np.median(sharpe)
        ci_sharpe_lower = np.percentile(sharpe, ci_lower_pct * 100)
        ci_sharpe_upper = np.percentile(sharpe, ci_upper_pct * 100)

        # Drawdown statistics
        mean_drawdown = np.mean(drawdowns)
        median_drawdown = np.median(drawdowns)
        worst_drawdown = np.max(drawdowns)

        # Risk metrics
        prob_positive = np.mean(returns > 0)
        prob_sharpe_above_1 = np.mean(sharpe > 1.0)
        prob_dd_below_20 = np.mean(drawdowns < 0.20)

        # Original results comparison
        if original_results:
            original_return = original_results.get('total_return', 0)
            original_sharpe = original_results.get('sharpe_ratio', 0)

            # Percentile rankings
            return_percentile = stats.percentileofscore(returns, original_return)
            sharpe_percentile = stats.percentileofscore(sharpe, original_sharpe)
        else:
            original_return = 0
            original_sharpe = 0
            return_percentile = 50
            sharpe_percentile = 50

        results = MonteCarloResults(
            n_simulations=len(returns),
            method=self.config.method,
            returns=returns,
            sharpe_ratios=sharpe,
            max_drawdowns=drawdowns,
            win_rates=win_rates,
            mean_return=mean_return,
            median_return=median_return,
            std_return=std_return,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            mean_sharpe=mean_sharpe,
            median_sharpe=median_sharpe,
            ci_sharpe_lower=ci_sharpe_lower,
            ci_sharpe_upper=ci_sharpe_upper,
            mean_drawdown=mean_drawdown,
            median_drawdown=median_drawdown,
            worst_drawdown=worst_drawdown,
            prob_positive_return=prob_positive,
            prob_sharpe_above_1=prob_sharpe_above_1,
            prob_drawdown_below_20pct=prob_dd_below_20,
            original_return=original_return,
            original_sharpe=original_sharpe,
            return_percentile=return_percentile,
            sharpe_percentile=sharpe_percentile
        )

        return results

    def plot_results(self, results: MonteCarloResults, save_path: Optional[str] = None) -> None:
        """
        Plot Monte Carlo simulation results.

        Args:
            results: MonteCarloResults object
            save_path: Path to save figure (optional)
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.set_style("whitegrid")
        except ImportError:
            logger.warning("matplotlib/seaborn not available, skipping plots")
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Plot 1: Return distribution
        ax = axes[0, 0]
        ax.hist(results.returns * 100, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(results.mean_return * 100, color='r', linestyle='--',
                   linewidth=2, label=f'Mean: {results.mean_return:.2%}')
        ax.axvline(results.original_return * 100, color='g', linestyle='-',
                   linewidth=2, label=f'Original: {results.original_return:.2%}')
        ax.axvline(results.ci_lower * 100, color='orange', linestyle=':',
                   label=f'{int(self.config.confidence_level*100)}% CI')
        ax.axvline(results.ci_upper * 100, color='orange', linestyle=':')
        ax.set_xlabel('Total Return (%)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'Return Distribution ({results.n_simulations} simulations)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Sharpe ratio distribution
        ax = axes[0, 1]
        ax.hist(results.sharpe_ratios, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(results.mean_sharpe, color='r', linestyle='--',
                   linewidth=2, label=f'Mean: {results.mean_sharpe:.3f}')
        ax.axvline(results.original_sharpe, color='g', linestyle='-',
                   linewidth=2, label=f'Original: {results.original_sharpe:.3f}')
        ax.axvline(results.ci_sharpe_lower, color='orange', linestyle=':',
                   label=f'{int(self.config.confidence_level*100)}% CI')
        ax.axvline(results.ci_sharpe_upper, color='orange', linestyle=':')
        ax.axvline(1.0, color='purple', linestyle='--', alpha=0.5,
                   label='Sharpe = 1.0')
        ax.set_xlabel('Sharpe Ratio')
        ax.set_ylabel('Frequency')
        ax.set_title('Sharpe Ratio Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Drawdown distribution
        ax = axes[1, 0]
        ax.hist(results.max_drawdowns * 100, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(results.mean_drawdown * 100, color='r', linestyle='--',
                   linewidth=2, label=f'Mean: {results.mean_drawdown:.2%}')
        ax.axvline(20, color='purple', linestyle='--', alpha=0.5,
                   label='20% threshold')
        ax.set_xlabel('Maximum Drawdown (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Maximum Drawdown Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Risk metrics summary
        ax = axes[1, 1]
        risk_data = [
            f"Method: {results.method}",
            f"Simulations: {results.n_simulations}",
            f"",
            f"Return:",
            f"  Mean:   {results.mean_return:7.2%}",
            f"  Median: {results.median_return:7.2%}",
            f"  95% CI: [{results.ci_lower:.2%}, {results.ci_upper:.2%}]",
            f"",
            f"Sharpe Ratio:",
            f"  Mean:   {results.mean_sharpe:7.3f}",
            f"  95% CI: [{results.ci_sharpe_lower:.3f}, {results.ci_sharpe_upper:.3f}]",
            f"",
            f"Probabilities:",
            f"  P(Return > 0):   {results.prob_positive_return:5.1%}",
            f"  P(Sharpe > 1):   {results.prob_sharpe_above_1:5.1%}",
            f"  P(DD < 20%):     {results.prob_drawdown_below_20pct:5.1%}",
        ]

        ax.text(0.1, 0.9, '\n'.join(risk_data), transform=ax.transAxes,
                fontsize=10, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        ax.axis('off')
        ax.set_title('Monte Carlo Summary')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")

        plt.show()
