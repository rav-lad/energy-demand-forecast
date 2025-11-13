"""
Cross-Regional Arbitrage Strategy for Electricity Markets

This strategy exploits price differentials between interconnected electricity
markets (e.g., France-Germany, France-Spain) while accounting for transmission
capacity constraints and congestion.

Core Logic:
If Price_FR > Price_DE + Transmission_Cost → SHORT FR, LONG DE
If Price_FR < Price_DE - Transmission_Cost → LONG FR, SHORT DE

Author: Quant Research Team
Date: 2024-11-12
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


@dataclass
class CrossRegionalArbitrageConfig:
    """Configuration for Cross-Regional Arbitrage Strategy."""

    # Regions to trade
    region_pairs: List[Tuple[str, str]] = None  # e.g., [('FR', 'DE'), ('FR', 'ES')]

    # Thresholds
    spread_entry_threshold: float = 5.0  # EUR/MWh minimum spread to enter
    spread_exit_threshold: float = 1.0  # EUR/MWh spread to exit position

    # Transmission
    transmission_cost: float = 2.0  # EUR/MWh (includes losses, fees)
    transmission_capacity: float = 3000.0  # MW (max flow capacity)
    congestion_penalty: float = 0.1  # Extra cost when near capacity

    # Position sizing
    max_position_size: float = 1000.0  # MW per trade
    position_sizing_method: str = "proportional"  # 'fixed', 'proportional', 'optimal'

    # Risk management
    correlation_threshold: float = 0.7  # Minimum correlation to trade
    volatility_scaling: bool = True  # Scale positions by volatility
    max_leverage: float = 2.0  # Maximum leverage ratio

    # Costs
    transaction_cost_bps: float = 5.0  # Basis points per leg
    holding_cost_daily: float = 0.5  # EUR/MW/day


class CrossRegionalArbitrageStrategy:
    """
    Cross-Regional Arbitrage Strategy.

    Exploits temporary price dislocations between interconnected electricity
    markets while managing transmission constraints and correlation breakdown risks.

    Key Features:
    - Cointegration testing (pairs must be cointegrated)
    - Transmission capacity modeling
    - Correlation breakdown detection
    - Optimal position sizing
    - Congestion risk management

    Example:
        >>> config = CrossRegionalArbitrageConfig(region_pairs=[('FR', 'DE')])
        >>> strategy = CrossRegionalArbitrageStrategy(config)
        >>> signals = strategy.generate_signals(
        ...     prices_df=df,  # Must contain 'price_FR', 'price_DE'
        ...     transmission_flows=flows_df
        ... )
    """

    def __init__(self, config: Optional[CrossRegionalArbitrageConfig] = None):
        """
        Initialize Cross-Regional Arbitrage Strategy.

        Args:
            config: Strategy configuration
        """
        if config is None:
            config = CrossRegionalArbitrageConfig()

        # Default region pairs if not specified
        if config.region_pairs is None:
            config.region_pairs = [("FR", "DE"), ("FR", "ES")]

        self.config = config
        self.is_calibrated = False
        self.cointegration_results = {}
        self.correlation_matrix = None
        self.hedge_ratios = {}

        logger.info(
            f"Initialized Cross-Regional Arbitrage Strategy. "
            f"Trading pairs: {self.config.region_pairs}"
        )

    def test_cointegration(
        self, price_series_1: pd.Series, price_series_2: pd.Series, significance_level: float = 0.05
    ) -> Tuple[bool, float, float]:
        """
        Test for cointegration between two price series using Engle-Granger test.

        Cointegration implies a long-term equilibrium relationship, making
        mean reversion strategies viable.

        Args:
            price_series_1: First price series
            price_series_2: Second price series
            significance_level: Significance level for test

        Returns:
            (is_cointegrated, p_value, hedge_ratio)
        """
        from statsmodels.tsa.stattools import coint

        # Align series
        common_idx = price_series_1.index.intersection(price_series_2.index)
        p1 = price_series_1.loc[common_idx]
        p2 = price_series_2.loc[common_idx]

        # Engle-Granger cointegration test
        score, p_value, _ = coint(p1, p2)

        is_cointegrated = p_value < significance_level

        # Calculate hedge ratio (OLS regression)
        # p1 = β * p2 + ε
        X = p2.values.reshape(-1, 1)
        y = p1.values
        hedge_ratio = np.linalg.lstsq(X, y, rcond=None)[0][0]

        logger.info(
            f"Cointegration test: p-value = {p_value:.4f}, "
            f"hedge_ratio = {hedge_ratio:.4f}, "
            f"cointegrated = {is_cointegrated}"
        )

        return is_cointegrated, p_value, hedge_ratio

    def calculate_spread(
        self, price_1: pd.Series, price_2: pd.Series, hedge_ratio: float = 1.0
    ) -> pd.Series:
        """
        Calculate price spread accounting for hedge ratio.

        Spread = Price_1 - hedge_ratio * Price_2

        Args:
            price_1: First price series
            price_2: Second price series
            hedge_ratio: Hedge ratio from cointegration

        Returns:
            Spread series
        """
        spread = price_1 - hedge_ratio * price_2
        return spread

    def calculate_half_life(self, spread: pd.Series) -> float:
        """
        Calculate half-life of spread mean reversion.

        Args:
            spread: Spread series

        Returns:
            Half-life in days
        """
        # AR(1) regression: Δspread_t = α + β*spread_{t-1} + ε
        lag_spread = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()

        # Align
        common_idx = lag_spread.index.intersection(spread_diff.index)
        X = lag_spread.loc[common_idx].values
        y = spread_diff.loc[common_idx].values

        # OLS
        X_with_const = np.column_stack([np.ones_like(X), X])
        beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
        mean_reversion_speed = -beta[1]

        if mean_reversion_speed > 0:
            half_life = np.log(2) / mean_reversion_speed
        else:
            half_life = np.inf

        return half_life

    def calibrate(self, prices_df: pd.DataFrame) -> None:
        """
        Calibrate strategy on historical data.

        Args:
            prices_df: DataFrame with price columns for each region
                      (e.g., 'price_FR', 'price_DE', 'price_ES')
        """
        logger.info("Calibrating cross-regional arbitrage strategy...")

        # Test cointegration for each pair
        for region_1, region_2 in self.config.region_pairs:
            price_col_1 = f"price_{region_1}"
            price_col_2 = f"price_{region_2}"

            if price_col_1 not in prices_df.columns or price_col_2 not in prices_df.columns:
                logger.warning(f"Missing price data for {region_1}-{region_2} pair. Skipping.")
                continue

            p1 = prices_df[price_col_1]
            p2 = prices_df[price_col_2]

            # Test cointegration
            is_coint, p_value, hedge_ratio = self.test_cointegration(p1, p2)

            pair_key = f"{region_1}_{region_2}"
            self.cointegration_results[pair_key] = {
                "is_cointegrated": is_coint,
                "p_value": p_value,
                "hedge_ratio": hedge_ratio,
            }
            self.hedge_ratios[pair_key] = hedge_ratio

            if not is_coint:
                logger.warning(
                    f"{region_1}-{region_2} not cointegrated (p={p_value:.4f}). "
                    f"Strategy may not work well."
                )
            else:
                logger.info(
                    f"{region_1}-{region_2} cointegrated (p={p_value:.4f}). "
                    f"Hedge ratio: {hedge_ratio:.4f}"
                )

                # Calculate spread half-life
                spread = self.calculate_spread(p1, p2, hedge_ratio)
                half_life = self.calculate_half_life(spread)
                logger.info(f"   Spread half-life: {half_life:.2f} days")

        # Calculate correlation matrix
        price_cols = [f"price_{r1}" for r1, _ in self.config.region_pairs] + [
            f"price_{r2}" for _, r2 in self.config.region_pairs
        ]
        price_cols = list(set(price_cols))  # Remove duplicates
        available_cols = [col for col in price_cols if col in prices_df.columns]

        if len(available_cols) > 1:
            self.correlation_matrix = prices_df[available_cols].corr()
            logger.info(f"Correlation matrix:\n{self.correlation_matrix}")

        self.is_calibrated = True
        logger.info("Calibration complete.")

    def detect_correlation_breakdown(
        self, price_1: pd.Series, price_2: pd.Series, rolling_window: int = 30
    ) -> pd.Series:
        """
        Detect periods of correlation breakdown.

        Correlation breakdown = risk that spread won't converge.

        Args:
            price_1: First price series
            price_2: Second price series
            rolling_window: Window for rolling correlation

        Returns:
            Series indicating breakdown (True/False)
        """
        # Calculate rolling correlation
        rolling_corr = price_1.rolling(rolling_window).corr(price_2)

        # Breakdown when correlation drops below threshold
        breakdown = rolling_corr < self.config.correlation_threshold

        return breakdown

    def calculate_optimal_position_size(
        self, spread: float, spread_std: float, correlation: float
    ) -> float:
        """
        Calculate optimal position size using Kelly criterion.

        Kelly fraction = (p*b - q) / b
        Where:
        - p = probability of winning
        - q = probability of losing
        - b = odds (payoff ratio)

        Simplified for mean reversion:
        Optimal size ∝ |spread| / variance * correlation

        Args:
            spread: Current spread value
            spread_std: Spread standard deviation
            correlation: Current correlation between regions

        Returns:
            Optimal position size (MW)
        """
        if spread_std == 0 or correlation < 0.5:
            return 0.0

        # Signal strength (Z-score)
        z_score = abs(spread) / spread_std

        # Base position from signal strength
        base_size = z_score * 100  # Scale factor

        # Scale by correlation (lower correlation = higher risk)
        correlation_adjustment = correlation / 0.9  # Normalize to 0.9 target

        # Kelly-inspired sizing
        kelly_size = base_size * correlation_adjustment

        # Apply max position limit
        position_size = min(kelly_size, self.config.max_position_size)

        return position_size

    def generate_signals(
        self, prices_df: pd.DataFrame, transmission_flows: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Generate trading signals for cross-regional arbitrage.

        Args:
            prices_df: DataFrame with price columns for regions
            transmission_flows: Optional DataFrame with actual transmission flows

        Returns:
            DataFrame with signals for each region pair
        """
        if not self.is_calibrated:
            logger.warning("Strategy not calibrated. Running calibration...")
            self.calibrate(prices_df)

        all_signals = []

        for region_1, region_2 in self.config.region_pairs:
            price_col_1 = f"price_{region_1}"
            price_col_2 = f"price_{region_2}"

            if price_col_1 not in prices_df.columns or price_col_2 not in prices_df.columns:
                continue

            pair_key = f"{region_1}_{region_2}"
            hedge_ratio = self.hedge_ratios.get(pair_key, 1.0)

            # Calculate spread
            p1 = prices_df[price_col_1]
            p2 = prices_df[price_col_2]
            spread = self.calculate_spread(p1, p2, hedge_ratio)

            # Rolling statistics
            spread_mean = spread.rolling(window=30, min_periods=1).mean()
            spread_std = spread.rolling(window=30, min_periods=1).std()
            spread_z = (spread - spread_mean) / spread_std

            # Correlation breakdown detection
            breakdown = self.detect_correlation_breakdown(p1, p2)

            # Generate signals
            df = pd.DataFrame(
                {
                    "date": (
                        prices_df.index
                        if isinstance(prices_df.index, pd.DatetimeIndex)
                        else pd.RangeIndex(len(prices_df))
                    ),
                    "region_1": region_1,
                    "region_2": region_2,
                    "price_1": p1.values,
                    "price_2": p2.values,
                    "spread": spread.values,
                    "spread_z": spread_z.values,
                    "correlation_breakdown": breakdown.values,
                }
            )

            # Signal logic
            df["signal"] = 0

            # Entry conditions (accounting for transmission cost)
            adjusted_threshold = self.config.spread_entry_threshold + self.config.transmission_cost

            # LONG spread (buy region_1, sell region_2) when spread is low
            long_condition = (df["spread"] < -adjusted_threshold) & (~df["correlation_breakdown"])
            df.loc[long_condition, "signal"] = 1

            # SHORT spread (sell region_1, buy region_2) when spread is high
            short_condition = (df["spread"] > adjusted_threshold) & (~df["correlation_breakdown"])
            df.loc[short_condition, "signal"] = -1

            # Exit when spread normalizes
            df["should_exit"] = df["spread"].abs() < self.config.spread_exit_threshold

            # Position sizing
            if self.config.position_sizing_method == "fixed":
                df["position_size"] = self.config.max_position_size
            elif self.config.position_sizing_method == "proportional":
                df["position_size"] = df["spread"].abs() * 50  # Scale factor
                df["position_size"] = df["position_size"].clip(upper=self.config.max_position_size)
            elif self.config.position_sizing_method == "optimal":
                # Calculate rolling correlation
                rolling_corr = p1.rolling(30).corr(p2)
                df["position_size"] = df.apply(
                    lambda row: self.calculate_optimal_position_size(
                        row["spread"],
                        spread_std.loc[row.name] if row.name in spread_std.index else 1.0,
                        rolling_corr.loc[row.name] if row.name in rolling_corr.index else 0.8,
                    ),
                    axis=1,
                )

            # Apply signal to position size
            df["position_size"] = df["position_size"] * df["signal"].abs()

            # Transaction costs (both legs)
            df["transaction_cost"] = (
                df["position_size"]
                * (df["price_1"] + df["price_2"])
                / 2
                * (self.config.transaction_cost_bps / 10000)
                * 2  # Two legs
            )

            # Transmission cost
            df["transmission_cost"] = (
                df["position_size"] * self.config.transmission_cost * df["signal"].abs()
            )

            all_signals.append(df)

        # Combine all pairs
        if len(all_signals) > 0:
            result_df = pd.concat(all_signals, ignore_index=True)
            logger.info(
                f"Generated signals for {len(self.config.region_pairs)} pairs. "
                f"Total signals: {(result_df['signal'] != 0).sum()}"
            )
            return result_df
        else:
            logger.warning("No signals generated (missing price data)")
            return pd.DataFrame()

    def calculate_performance_metrics(self, signals_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate strategy performance metrics.

        Args:
            signals_df: DataFrame from generate_signals()

        Returns:
            Dictionary with performance metrics
        """
        if len(signals_df) == 0:
            return {}

        # Group by region pair
        results_by_pair = {}

        for pair in signals_df.groupby(["region_1", "region_2"]):
            (region_1, region_2), df = pair
            pair_key = f"{region_1}_{region_2}"

            # Calculate PnL
            # When long spread: profit if spread increases
            # When short spread: profit if spread decreases
            df["spread_change"] = df["spread"].diff()
            df["gross_pnl"] = df["signal"].shift(1) * df["spread_change"] * df["position_size"]

            # Net PnL after costs
            df["net_pnl"] = df["gross_pnl"] - df["transaction_cost"] - df["transmission_cost"]

            # Cumulative PnL
            df["cumulative_pnl"] = df["net_pnl"].cumsum()

            # Metrics
            total_pnl = df["net_pnl"].sum()
            num_trades = (df["signal"].diff() != 0).sum()

            # Returns
            capital_at_risk = df["position_size"] * (df["price_1"] + df["price_2"]) / 2
            returns = df["net_pnl"] / (capital_at_risk + 1)
            returns = returns.replace([np.inf, -np.inf], 0)

            sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

            # Win rate
            profitable_trades = (df["net_pnl"] > 0).sum()
            win_rate = profitable_trades / num_trades if num_trades > 0 else 0

            # Maximum drawdown
            cumulative = df["cumulative_pnl"]
            running_max = cumulative.expanding().max()
            drawdown = cumulative - running_max
            max_drawdown = drawdown.min()

            results_by_pair[pair_key] = {
                "total_pnl": total_pnl,
                "num_trades": num_trades,
                "sharpe_ratio": sharpe,
                "win_rate": win_rate,
                "max_drawdown": max_drawdown,
                "avg_pnl_per_trade": total_pnl / num_trades if num_trades > 0 else 0,
            }

        # Aggregate across all pairs
        aggregate_metrics = {
            "total_pnl": sum(m["total_pnl"] for m in results_by_pair.values()),
            "num_trades": sum(m["num_trades"] for m in results_by_pair.values()),
            "avg_sharpe": np.mean([m["sharpe_ratio"] for m in results_by_pair.values()]),
            "avg_win_rate": np.mean([m["win_rate"] for m in results_by_pair.values()]),
            "worst_drawdown": min(m["max_drawdown"] for m in results_by_pair.values()),
        }

        return {"aggregate": aggregate_metrics, "by_pair": results_by_pair}


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Generate synthetic data
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2024-01-01", freq="D")
    n = len(dates)

    # Cointegrated prices with spread
    common_component = np.cumsum(np.random.randn(n) * 0.5)
    price_FR = 50 + common_component + np.random.randn(n) * 2
    price_DE = 48 + common_component + np.random.randn(n) * 2  # Slightly lower
    price_ES = 52 + common_component + np.random.randn(n) * 3  # Slightly higher

    prices_df = pd.DataFrame(
        {"price_FR": price_FR, "price_DE": price_DE, "price_ES": price_ES}, index=dates
    )

    # Initialize and calibrate strategy
    config = CrossRegionalArbitrageConfig(
        region_pairs=[("FR", "DE"), ("FR", "ES")], spread_entry_threshold=3.0
    )
    strategy = CrossRegionalArbitrageStrategy(config)
    strategy.calibrate(prices_df)

    # Generate signals
    signals = strategy.generate_signals(prices_df)

    # Calculate performance
    metrics = strategy.calculate_performance_metrics(signals)

    print("\n" + "=" * 60)
    print("CROSS-REGIONAL ARBITRAGE - PERFORMANCE METRICS")
    print("=" * 60)

    if "aggregate" in metrics:
        print("\nAGGREGATE METRICS:")
        for key, value in metrics["aggregate"].items():
            if isinstance(value, float):
                print(f"{key:25s}: {value:12.4f}")
            else:
                print(f"{key:25s}: {value}")

        print("\nBY REGION PAIR:")
        for pair, pair_metrics in metrics["by_pair"].items():
            print(f"\n{pair}:")
            for key, value in pair_metrics.items():
                if isinstance(value, float):
                    print(f"  {key:23s}: {value:12.4f}")
                else:
                    print(f"  {key:23s}: {value}")
