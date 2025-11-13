"""
Advanced Risk Management Module for Trading System

Comprehensive risk management with:
- VaR (Value at Risk) and CVaR (Conditional VaR)
- Stress testing and scenario analysis
- Position limits and exposure monitoring
- Drawdown control
- Correlation monitoring

Author: Quant Research Team
Date: 2024-11-12
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Risk limits configuration."""

    # VaR limits
    var_confidence: float = 0.95  # 95% VaR
    max_var_daily: float = 10000.0  # EUR
    max_var_pct: float = 0.02  # 2% of portfolio

    # Position limits
    max_position_size: float = 5000.0  # MW
    max_portfolio_notional: float = 100000.0  # EUR
    max_concentration: float = 0.3  # Max 30% in single position

    # Drawdown limits
    max_drawdown_pct: float = 0.10  # 10%
    max_drawdown_duration_days: int = 30  # Days
    stop_trading_drawdown: float = 0.15  # Stop at 15% drawdown

    # Leverage
    max_leverage: float = 3.0  # Maximum leverage ratio

    # Correlation
    min_diversification_ratio: float = 0.7  # Portfolio diversification


class RiskManager:
    """
    Advanced Risk Management System.

    Features:
    - Real-time VaR/CVaR monitoring
    - Position limit enforcement
    - Drawdown tracking
    - Stress testing
    - Correlation breakdown detection
    - Risk-adjusted position sizing

    Example:
        >>> limits = RiskLimits(max_var_daily=10000)
        >>> risk_mgr = RiskManager(limits)
        >>> is_allowed = risk_mgr.check_position_limits(position_size=1000)
    """

    def __init__(self, limits: Optional[RiskLimits] = None):
        """
        Initialize Risk Manager.

        Args:
            limits: Risk limits configuration
        """
        self.limits = limits or RiskLimits()
        self.current_positions = {}
        self.pnl_history = []
        self.drawdown_start = None
        self.peak_capital = 0.0

        logger.info(f"Initialized Risk Manager with limits: {self.limits}")

    def calculate_var(
        self, returns: pd.Series, confidence_level: float = 0.95, method: str = "historical"
    ) -> float:
        """
        Calculate Value at Risk (VaR).

        VaR = Maximum expected loss at given confidence level

        Methods:
        - 'historical': Historical simulation
        - 'parametric': Assume normal distribution
        - 'monte_carlo': Monte Carlo simulation

        Args:
            returns: Return series
            confidence_level: Confidence level (e.g., 0.95 for 95% VaR)
            method: Calculation method

        Returns:
            VaR value (positive number represents loss)
        """
        if len(returns) < 30:
            logger.warning("Insufficient data for VaR calculation")
            return 0.0

        returns_clean = returns.dropna()

        if method == "historical":
            # Historical VaR: percentile of actual returns
            var = -np.percentile(returns_clean, (1 - confidence_level) * 100)

        elif method == "parametric":
            # Parametric VaR: assume normal distribution
            mu = returns_clean.mean()
            sigma = returns_clean.std()
            z_score = stats.norm.ppf(1 - confidence_level)
            var = -(mu + z_score * sigma)

        elif method == "monte_carlo":
            # Monte Carlo VaR: simulate future returns
            mu = returns_clean.mean()
            sigma = returns_clean.std()
            n_simulations = 10000

            simulated_returns = np.random.normal(mu, sigma, n_simulations)
            var = -np.percentile(simulated_returns, (1 - confidence_level) * 100)

        else:
            raise ValueError(f"Unknown VaR method: {method}")

        return var

    def calculate_cvar(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calculate Conditional VaR (CVaR / Expected Shortfall).

        CVaR = Expected loss given that loss exceeds VaR

        CVaR is more conservative than VaR and accounts for tail risk.

        Args:
            returns: Return series
            confidence_level: Confidence level

        Returns:
            CVaR value (positive number represents loss)
        """
        if len(returns) < 30:
            logger.warning("Insufficient data for CVaR calculation")
            return 0.0

        returns_clean = returns.dropna()

        # Calculate VaR threshold
        var_threshold = -np.percentile(returns_clean, (1 - confidence_level) * 100)

        # CVaR = mean of returns worse than VaR
        tail_returns = returns_clean[returns_clean <= -var_threshold]

        if len(tail_returns) > 0:
            cvar = -tail_returns.mean()
        else:
            cvar = var_threshold

        return cvar

    def check_var_limit(self, current_var: float, portfolio_value: float) -> Tuple[bool, str]:
        """
        Check if VaR is within limits.

        Args:
            current_var: Current VaR value
            portfolio_value: Current portfolio value

        Returns:
            (is_within_limits, message)
        """
        # Check absolute limit
        if current_var > self.limits.max_var_daily:
            return False, f"VaR {current_var:.2f} exceeds limit {self.limits.max_var_daily:.2f}"

        # Check percentage limit
        var_pct = current_var / portfolio_value if portfolio_value > 0 else 0
        if var_pct > self.limits.max_var_pct:
            return False, f"VaR {var_pct:.2%} exceeds limit {self.limits.max_var_pct:.2%}"

        return True, "VaR within limits"

    def calculate_drawdown(self, equity_curve: pd.Series) -> Tuple[pd.Series, float, int]:
        """
        Calculate drawdown statistics.

        Args:
            equity_curve: Portfolio equity time series

        Returns:
            (drawdown_series, max_drawdown, current_drawdown_duration)
        """
        # Running maximum
        running_max = equity_curve.expanding().max()

        # Drawdown series
        drawdown = (equity_curve - running_max) / running_max

        # Maximum drawdown
        max_drawdown = drawdown.min()

        # Current drawdown duration
        current_drawdown_duration = 0
        if len(drawdown) > 0 and drawdown.iloc[-1] < 0:
            # Find last peak
            last_peak_idx = equity_curve.expanding().apply(lambda x: x.argmax(), raw=False).iloc[-1]
            current_drawdown_duration = len(equity_curve) - int(last_peak_idx)

        return drawdown, max_drawdown, current_drawdown_duration

    def check_drawdown_limit(self, equity_curve: pd.Series) -> Tuple[bool, Dict[str, float]]:
        """
        Check if drawdown is within limits.

        Args:
            equity_curve: Portfolio equity time series

        Returns:
            (is_within_limits, metrics)
        """
        drawdown_series, max_dd, dd_duration = self.calculate_drawdown(equity_curve)

        current_dd = drawdown_series.iloc[-1] if len(drawdown_series) > 0 else 0

        metrics = {
            "current_drawdown": current_dd,
            "max_drawdown": max_dd,
            "drawdown_duration": dd_duration,
        }

        # Check if should stop trading
        if abs(current_dd) > self.limits.stop_trading_drawdown:
            return False, metrics

        # Check max drawdown limit
        if abs(max_dd) > self.limits.max_drawdown_pct:
            logger.warning(
                f"Max drawdown {max_dd:.2%} exceeds limit {self.limits.max_drawdown_pct:.2%}"
            )

        # Check duration limit
        if dd_duration > self.limits.max_drawdown_duration_days:
            logger.warning(
                f"Drawdown duration {dd_duration} days exceeds limit "
                f"{self.limits.max_drawdown_duration_days} days"
            )

        return True, metrics

    def check_position_limits(
        self, position_size: float, current_positions: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, str]:
        """
        Check if position size is within limits.

        Args:
            position_size: Proposed position size
            current_positions: Dict of current positions {asset: size}

        Returns:
            (is_allowed, message)
        """
        # Check individual position limit
        if abs(position_size) > self.limits.max_position_size:
            return False, f"Position {position_size} exceeds limit {self.limits.max_position_size}"

        # Check portfolio limits if current positions provided
        if current_positions:
            total_exposure = sum(abs(pos) for pos in current_positions.values())
            total_exposure += abs(position_size)

            if total_exposure > self.limits.max_portfolio_notional:
                return False, f"Total exposure {total_exposure} exceeds limit"

            # Check concentration
            if total_exposure > 0:
                concentration = abs(position_size) / total_exposure
                if concentration > self.limits.max_concentration:
                    return False, f"Concentration {concentration:.2%} exceeds limit"

        return True, "Position within limits"

    def calculate_leverage(
        self, positions: Dict[str, float], prices: Dict[str, float], capital: float
    ) -> float:
        """
        Calculate current leverage ratio.

        Leverage = Total notional exposure / Capital

        Args:
            positions: Dict of positions {asset: quantity}
            prices: Dict of prices {asset: price}
            capital: Available capital

        Returns:
            Leverage ratio
        """
        total_notional = sum(
            abs(positions.get(asset, 0)) * prices.get(asset, 0) for asset in positions
        )

        leverage = total_notional / capital if capital > 0 else 0

        return leverage

    def stress_test(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
        scenarios: Dict[str, Dict[str, float]],
    ) -> pd.DataFrame:
        """
        Perform stress testing under various scenarios.

        Args:
            positions: Current positions {asset: quantity}
            prices: Current prices {asset: price}
            scenarios: Dict of scenarios {name: {asset: price_shock_pct}}

        Returns:
            DataFrame with scenario results
        """
        results = []

        for scenario_name, shocks in scenarios.items():
            scenario_pnl = 0.0

            for asset, position in positions.items():
                if asset in prices and asset in shocks:
                    current_price = prices[asset]
                    shock_pct = shocks[asset]
                    shocked_price = current_price * (1 + shock_pct)
                    pnl = position * (shocked_price - current_price)
                    scenario_pnl += pnl

            results.append(
                {
                    "scenario": scenario_name,
                    "pnl": scenario_pnl,
                    "pnl_pct": (
                        scenario_pnl
                        / sum(abs(pos) * prices.get(asset, 0) for asset, pos in positions.items())
                        if positions
                        else 0
                    ),
                }
            )

        return pd.DataFrame(results)

    def calculate_portfolio_var(
        self,
        positions: Dict[str, float],
        returns: pd.DataFrame,  # Returns for each asset
        confidence_level: float = 0.95,
    ) -> float:
        """
        Calculate portfolio VaR accounting for correlations.

        Args:
            positions: Dict of positions {asset: quantity}
            returns: DataFrame with return series for each asset
            confidence_level: Confidence level

        Returns:
            Portfolio VaR
        """
        # Get returns for assets in portfolio
        assets = list(positions.keys())
        asset_returns = returns[assets]

        # Position weights
        total_value = sum(abs(positions[asset]) for asset in assets)
        weights = np.array([positions[asset] / total_value for asset in assets])

        # Calculate portfolio returns
        portfolio_returns = (asset_returns * weights).sum(axis=1)

        # Calculate VaR
        var = self.calculate_var(portfolio_returns, confidence_level)

        return var

    def risk_adjusted_position_size(
        self,
        signal_strength: float,
        volatility: float,
        correlation_to_portfolio: float = 0.0,
        risk_budget: float = 0.01,
    ) -> float:
        """
        Calculate risk-adjusted position size.

        Size based on:
        - Signal strength
        - Asset volatility
        - Correlation to existing portfolio
        - Risk budget

        Args:
            signal_strength: Signal strength (0-1)
            volatility: Asset volatility (annualized)
            correlation_to_portfolio: Correlation to existing portfolio
            risk_budget: Fraction of capital to risk (e.g., 0.01 = 1%)

        Returns:
            Position size
        """
        if volatility == 0:
            return 0.0

        # Base size from signal
        base_size = signal_strength * 1000  # Scale factor

        # Adjust for volatility (inverse relationship)
        volatility_adjustment = 0.2 / volatility  # Target 20% vol

        # Adjust for correlation (reduce size if highly correlated)
        correlation_adjustment = 1.0 - abs(correlation_to_portfolio) * 0.5

        # Apply risk budget
        risk_adjusted_size = base_size * volatility_adjustment * correlation_adjustment

        # Cap at limits
        risk_adjusted_size = min(risk_adjusted_size, self.limits.max_position_size)

        return risk_adjusted_size


def create_stress_scenarios() -> Dict[str, Dict[str, float]]:
    """
    Create predefined stress testing scenarios.

    Returns:
        Dict of scenarios with price shocks
    """
    scenarios = {
        "market_crash": {"price_FR": -0.30, "price_DE": -0.28, "price_ES": -0.32},  # -30%
        "energy_crisis": {"price_FR": 0.50, "price_DE": 0.55, "price_ES": 0.48},  # +50%
        "renewable_surge": {"price_FR": -0.15, "price_DE": -0.20, "price_ES": -0.10},  # -15%
        "cold_snap": {"price_FR": 0.40, "price_DE": 0.35, "price_ES": 0.30},  # +40%
        "correlation_breakdown": {
            "price_FR": 0.20,
            "price_DE": -0.15,  # Opposite direction
            "price_ES": 0.25,
        },
    }

    return scenarios


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create risk manager
    limits = RiskLimits(max_var_daily=5000, max_drawdown_pct=0.10)
    risk_mgr = RiskManager(limits)

    # Generate synthetic returns
    np.random.seed(42)
    returns = pd.Series(np.random.randn(252) * 0.02)  # Daily returns, 2% vol

    # Calculate VaR
    var_95 = risk_mgr.calculate_var(returns, 0.95, method="historical")
    cvar_95 = risk_mgr.calculate_cvar(returns, 0.95)

    print("\n" + "=" * 60)
    print("RISK METRICS")
    print("=" * 60)
    print(f"VaR (95%):  {var_95:.4f}")
    print(f"CVaR (95%): {cvar_95:.4f}")

    # Check limits
    portfolio_value = 100000
    is_ok, msg = risk_mgr.check_var_limit(var_95 * portfolio_value, portfolio_value)
    print(f"\nVaR check: {msg}")

    # Drawdown analysis
    equity_curve = (1 + returns).cumprod() * 100000
    is_ok, dd_metrics = risk_mgr.check_drawdown_limit(equity_curve)

    print("\nDRAWDOWN METRICS:")
    for key, value in dd_metrics.items():
        print(f"{key:25s}: {value:.4f}")

    # Stress testing
    positions = {"price_FR": 1000, "price_DE": -800, "price_ES": 500}
    prices = {"price_FR": 50, "price_DE": 48, "price_ES": 52}

    scenarios = create_stress_scenarios()
    stress_results = risk_mgr.stress_test(positions, prices, scenarios)

    print("\nSTRESS TEST RESULTS:")
    print(stress_results.to_string(index=False))
