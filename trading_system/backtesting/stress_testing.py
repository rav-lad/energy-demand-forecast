"""Stress Testing Framework for Trading Strategies

This module provides stress testing scenarios to evaluate strategy robustness
under extreme market conditions.

Key scenarios:
- 2022 Energy Crisis (high prices, extreme volatility)
- Negative Price Events (renewable flush)
- Market Regime Shifts (sudden structural changes)
- Liquidity Crises (wide spreads, high slippage)

Usage:
    stress_tester = StressTester()
    results = stress_tester.run_crisis_2022_scenario(strategy, prices)
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from trading_system.constants import (
    CRISIS_2022_GAS_MULTIPLIER,
    CRISIS_2022_CARBON_MULTIPLIER,
    CRISIS_2022_VOLATILITY_MULTIPLIER,
    CRISIS_2022_SPIKE_PROBABILITY,
    TYPICAL_TTF_GAS_EUR_MWH,
    TYPICAL_EUA_CARBON_EUR_TCO2,
    MIN_ELECTRICITY_PRICE_EUR_MWH,
    MAX_ELECTRICITY_PRICE_EUR_MWH,
)

logger = logging.getLogger(__name__)


@dataclass
class StressScenario:
    """Defines a stress test scenario."""

    name: str
    description: str
    price_multiplier: float = 1.0
    volatility_multiplier: float = 1.0
    spike_probability: float = 0.01
    min_price: float = MIN_ELECTRICITY_PRICE_EUR_MWH
    max_price: float = MAX_ELECTRICITY_PRICE_EUR_MWH
    transaction_cost_multiplier: float = 1.0


class StressTester:
    """Stress testing framework for trading strategies.

    Applies various market stress scenarios to evaluate strategy resilience.
    """

    def __init__(self):
        """Initialize stress tester with predefined scenarios."""
        self.scenarios = self._create_scenarios()

    def _create_scenarios(self) -> Dict[str, StressScenario]:
        """Create library of stress scenarios.

        Returns:
            Dictionary of scenario_name -> StressScenario
        """
        return {
            "crisis_2022": StressScenario(
                name="2022 Energy Crisis",
                description="Russia-Ukraine war impact: 3x gas prices, extreme volatility",
                price_multiplier=CRISIS_2022_GAS_MULTIPLIER,
                volatility_multiplier=CRISIS_2022_VOLATILITY_MULTIPLIER,
                spike_probability=CRISIS_2022_SPIKE_PROBABILITY,
                max_price=900.0,  # Historical max in 2022
                transaction_cost_multiplier=1.5,  # Higher spreads during crisis
            ),
            "negative_prices": StressScenario(
                name="Renewable Flush",
                description="High renewable penetration, negative prices frequent",
                price_multiplier=0.5,
                volatility_multiplier=2.0,
                spike_probability=0.03,
                min_price=-200.0,  # Extreme negative prices
                max_price=200.0,
            ),
            "regime_shift": StressScenario(
                name="Regime Shift",
                description="Sudden structural market change (policy, technology)",
                price_multiplier=1.5,
                volatility_multiplier=3.0,
                spike_probability=0.04,
                transaction_cost_multiplier=2.0,  # Uncertainty → wide spreads
            ),
            "liquidity_crisis": StressScenario(
                name="Liquidity Crisis",
                description="Market participants withdraw, spreads widen dramatically",
                price_multiplier=1.2,
                volatility_multiplier=2.5,
                spike_probability=0.03,
                transaction_cost_multiplier=5.0,  # 5x transaction costs
            ),
            "baseline": StressScenario(
                name="Baseline (Normal Market)",
                description="Normal market conditions for comparison",
                price_multiplier=1.0,
                volatility_multiplier=1.0,
                spike_probability=0.01,
            ),
        }

    def apply_stress_to_prices(
        self,
        prices: pd.Series,
        scenario: StressScenario,
        random_seed: Optional[int] = None,
    ) -> pd.Series:
        """Apply stress scenario to price series.

        Args:
            prices: Original price series
            scenario: Stress scenario to apply
            random_seed: Random seed for reproducibility

        Returns:
            Stressed price series
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        n = len(prices)
        stressed_prices = prices.copy()

        # Apply price multiplier
        stressed_prices = stressed_prices * scenario.price_multiplier

        # Apply volatility multiplier
        price_returns = stressed_prices.pct_change()
        mean_return = price_returns.mean()
        centered_returns = price_returns - mean_return

        # Scale volatility
        scaled_returns = centered_returns * scenario.volatility_multiplier + mean_return

        # Reconstruct prices
        stressed_prices = prices.iloc[0] * (1 + scaled_returns).cumprod()

        # Add price spikes
        spike_mask = np.random.rand(n) < scenario.spike_probability
        spike_direction = np.random.choice([-1, 1], size=n)  # Up or down spikes
        spike_magnitude = np.random.exponential(50, n) * spike_mask * spike_direction

        stressed_prices = stressed_prices + spike_magnitude

        # Enforce bounds
        stressed_prices = stressed_prices.clip(
            lower=scenario.min_price,
            upper=scenario.max_price,
        )

        return stressed_prices

    def run_stress_test(
        self,
        strategy,
        prices: pd.Series,
        scenario_name: str,
        initial_capital: float = 100000.0,
        random_seed: Optional[int] = None,
        **strategy_kwargs,
    ) -> Dict:
        """Run stress test for a strategy under specific scenario.

        Args:
            strategy: Strategy instance with backtest() method
            prices: Original price series
            scenario_name: Name of scenario from self.scenarios
            initial_capital: Starting capital
            random_seed: Random seed
            **strategy_kwargs: Additional strategy parameters

        Returns:
            Dictionary with:
            - scenario: Scenario details
            - stressed_prices: Price series under stress
            - backtest_results: Full backtest results
            - performance_degradation: % change in metrics vs baseline
        """
        if scenario_name not in self.scenarios:
            raise ValueError(
                f"Unknown scenario: {scenario_name}. "
                f"Available: {list(self.scenarios.keys())}"
            )

        scenario = self.scenarios[scenario_name]

        logger.info(f"Running stress test: {scenario.name}")
        logger.info(f"Description: {scenario.description}")

        # Apply stress to prices
        stressed_prices = self.apply_stress_to_prices(
            prices=prices,
            scenario=scenario,
            random_seed=random_seed,
        )

        # Adjust transaction costs
        base_txn_cost = strategy_kwargs.get("transaction_cost", 0.001)
        stressed_txn_cost = base_txn_cost * scenario.transaction_cost_multiplier

        # Run backtest with stressed conditions
        backtest_results = strategy.backtest(
            prices=stressed_prices,
            initial_capital=initial_capital,
            transaction_cost=stressed_txn_cost,
            **strategy_kwargs,
        )

        return {
            "scenario": scenario,
            "stressed_prices": stressed_prices,
            "backtest_results": backtest_results,
            "scenario_name": scenario_name,
        }

    def run_all_scenarios(
        self,
        strategy,
        prices: pd.Series,
        initial_capital: float = 100000.0,
        random_seed: Optional[int] = None,
        **strategy_kwargs,
    ) -> pd.DataFrame:
        """Run strategy against all stress scenarios.

        Args:
            strategy: Strategy instance
            prices: Original price series
            initial_capital: Starting capital
            random_seed: Random seed
            **strategy_kwargs: Strategy parameters

        Returns:
            DataFrame with performance metrics across all scenarios
        """
        results = []

        for scenario_name in self.scenarios.keys():
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Scenario: {scenario_name}")
            logger.info(f"{'=' * 60}")

            stress_result = self.run_stress_test(
                strategy=strategy,
                prices=prices,
                scenario_name=scenario_name,
                initial_capital=initial_capital,
                random_seed=random_seed,
                **strategy_kwargs,
            )

            metrics = stress_result["backtest_results"]["metrics"]
            risk_metrics = stress_result["backtest_results"]["risk_metrics"]

            results.append({
                "scenario": scenario_name,
                "description": stress_result["scenario"].description,
                "sharpe_ratio": metrics["sharpe_ratio"],
                "sortino_ratio": metrics["sortino_ratio"],
                "max_drawdown": metrics["max_drawdown"],
                "total_return": metrics["total_return"],
                "win_rate": metrics["win_rate"],
                "profit_factor": metrics["profit_factor"],
                "var_95": risk_metrics["var_95"],
                "cvar_95": risk_metrics["cvar_95"],
                "volatility": risk_metrics["volatility_annual"],
            })

        results_df = pd.DataFrame(results)

        # Calculate degradation relative to baseline
        baseline_metrics = results_df[results_df["scenario"] == "baseline"].iloc[0]

        for idx, row in results_df.iterrows():
            if row["scenario"] != "baseline":
                results_df.loc[idx, "sharpe_degradation_%"] = (
                    (row["sharpe_ratio"] - baseline_metrics["sharpe_ratio"])
                    / baseline_metrics["sharpe_ratio"]
                    * 100
                )
                results_df.loc[idx, "return_degradation_%"] = (
                    (row["total_return"] - baseline_metrics["total_return"])
                    / baseline_metrics["total_return"]
                    * 100
                )

        logger.info("\n" + "=" * 80)
        logger.info("STRESS TEST SUMMARY")
        logger.info("=" * 80)
        logger.info("\n" + results_df.to_string(index=False))

        return results_df

    def run_crisis_2022_scenario(
        self,
        strategy,
        prices: pd.Series,
        initial_capital: float = 100000.0,
        **strategy_kwargs,
    ) -> Dict:
        """Convenience method for 2022 energy crisis stress test.

        This is the most important stress test for energy trading strategies.

        Args:
            strategy: Strategy instance
            prices: Original price series
            initial_capital: Starting capital
            **strategy_kwargs: Strategy parameters

        Returns:
            Stress test results dictionary

        Example:
            >>> from trading_system.strategies.mean_reversion import MeanReversionStrategy
            >>> from trading_system.backtesting.stress_testing import StressTester
            >>>
            >>> strategy = MeanReversionStrategy()
            >>> stress_tester = StressTester()
            >>> results = stress_tester.run_crisis_2022_scenario(strategy, prices)
            >>> print(f"Sharpe under crisis: {results['backtest_results']['metrics']['sharpe_ratio']:.2f}")
        """
        return self.run_stress_test(
            strategy=strategy,
            prices=prices,
            scenario_name="crisis_2022",
            initial_capital=initial_capital,
            random_seed=42,  # Reproducible crisis
            **strategy_kwargs,
        )


def main():
    """Example usage of stress testing framework."""
    # Create synthetic price series
    np.random.seed(42)
    n = 1000
    prices = pd.Series(
        50 + np.cumsum(np.random.randn(n) * 2),
        index=pd.date_range("2022-01-01", periods=n, freq="H"),
    )

    # Initialize stress tester
    stress_tester = StressTester()

    # Apply 2022 crisis scenario
    scenario = stress_tester.scenarios["crisis_2022"]
    stressed_prices = stress_tester.apply_stress_to_prices(
        prices=prices,
        scenario=scenario,
        random_seed=42,
    )

    # Plot comparison
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # Original prices
        axes[0].plot(prices.index, prices.values, label="Normal Market", alpha=0.7)
        axes[0].set_ylabel("Price (EUR/MWh)")
        axes[0].set_title("Normal Market Conditions")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Stressed prices
        axes[1].plot(
            stressed_prices.index,
            stressed_prices.values,
            label="2022 Energy Crisis",
            color="red",
            alpha=0.7,
        )
        axes[1].set_ylabel("Price (EUR/MWh)")
        axes[1].set_title("2022 Energy Crisis Scenario (3x gas prices, 2.5x volatility)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("stress_test_comparison.png", dpi=150)
        print("✅ Saved visualization: stress_test_comparison.png")

    except ImportError:
        print("⚠️ matplotlib not installed, skipping visualization")

    # Print statistics
    print("\n" + "=" * 60)
    print("STRESS TEST: 2022 Energy Crisis")
    print("=" * 60)
    print(f"\nNormal Market:")
    print(f"  Mean price:  {prices.mean():.2f} EUR/MWh")
    print(f"  Std price:   {prices.std():.2f} EUR/MWh")
    print(f"  Min price:   {prices.min():.2f} EUR/MWh")
    print(f"  Max price:   {prices.max():.2f} EUR/MWh")

    print(f"\n2022 Crisis:")
    print(f"  Mean price:  {stressed_prices.mean():.2f} EUR/MWh")
    print(f"  Std price:   {stressed_prices.std():.2f} EUR/MWh")
    print(f"  Min price:   {stressed_prices.min():.2f} EUR/MWh")
    print(f"  Max price:   {stressed_prices.max():.2f} EUR/MWh")

    print(f"\nMultipliers:")
    print(f"  Mean increase: {stressed_prices.mean() / prices.mean():.2f}x")
    print(f"  Volatility increase: {stressed_prices.std() / prices.std():.2f}x")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main()
