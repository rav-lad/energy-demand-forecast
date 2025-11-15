"""Trading System Constants

Centralized constants to replace magic numbers throughout the codebase.

Benefits:
- Single source of truth for parameters
- Easier parameter tuning and calibration
- Self-documenting code (names explain purpose)
- Consistency across modules

Usage:
    from trading_system.constants import (
        HOURS_PER_YEAR,
        DEFAULT_TRANSACTION_COST_BPS,
        MIN_ELECTRICITY_PRICE_EUR_MWH,
    )
"""

# =============================================================================
# Time Constants
# =============================================================================

HOURS_PER_DAY = 24
DAYS_PER_YEAR = 365
HOURS_PER_YEAR = DAYS_PER_YEAR * HOURS_PER_DAY  # 8760
TRADING_DAYS_PER_YEAR = 252  # For daily data annualization
WEEKS_PER_YEAR = 52

# =============================================================================
# Transaction Cost Constants
# =============================================================================

# Basis points (1 bp = 0.01%)
DEFAULT_TRANSACTION_COST_BPS = 10.0  # 10 bps = 0.1%
AGGRESSIVE_TRANSACTION_COST_BPS = 5.0  # For liquid markets
CONSERVATIVE_TRANSACTION_COST_BPS = 20.0  # For illiquid periods

# Commission structure
MIN_COMMISSION_EUR = 1.0  # Minimum commission per trade
DEFAULT_COMMISSION_PCT = 0.001  # 0.1%

# Slippage model
DEFAULT_SLIPPAGE_PCT = 0.0005  # 0.05%
DEFAULT_SLIPPAGE_FIXED_EUR_MWH = 0.5  # Fixed slippage per MWh
MARKET_IMPACT_COEFFICIENT = 0.01  # For large orders

# =============================================================================
# Price Constants (EUR/MWh)
# =============================================================================

# Electricity prices
MIN_ELECTRICITY_PRICE_EUR_MWH = -50.0  # Negative prices possible (renewable flush)
MAX_ELECTRICITY_PRICE_EUR_MWH = 3000.0  # Scarcity events (2022 crisis max ~900)
TYPICAL_BASE_PRICE_EUR_MWH = 50.0  # Pre-2022 average
TYPICAL_PEAK_PRICE_EUR_MWH = 80.0

# Fuel prices
TYPICAL_TTF_GAS_EUR_MWH = 30.0  # Pre-2022 average
TTF_GAS_FLOOR_EUR_MWH = 5.0  # Minimum gas price
TYPICAL_EUA_CARBON_EUR_TCO2 = 80.0  # 2023-2024 average
TYPICAL_COAL_API2_EUR_TONNE = 100.0  # 2019-2021 average

# =============================================================================
# Market Fundamentals
# =============================================================================

# Power plant efficiencies (thermal)
GAS_CCGT_EFFICIENCY = 0.55  # Combined cycle gas turbine
COAL_PLANT_EFFICIENCY = 0.38  # Subcritical coal plant
OIL_PLANT_EFFICIENCY = 0.35  # Diesel/fuel oil

# Emission factors (tCO2/MWh)
GAS_EMISSION_FACTOR = 0.35  # Natural gas
COAL_EMISSION_FACTOR = 0.95  # Coal
OIL_EMISSION_FACTOR = 0.65  # Oil

# Coal conversion factor
COAL_HEATING_VALUE_MWH_PER_TONNE = 8.14  # Typical heating value

# =============================================================================
# Statistical Thresholds
# =============================================================================

# MAPE calculation
EPSILON_MAPE = 1e-8  # Standard epsilon to avoid division by zero

# Sharpe ratio calculation
MIN_STD_FOR_SHARPE = 1e-10  # Minimum std to avoid division by zero

# Z-score calculation
MIN_STD_FOR_ZSCORE_EUR_MWH = 0.01  # Minimum price std for Z-score

# Correlation thresholds
HIGH_CORRELATION_THRESHOLD = 0.7
MODERATE_CORRELATION_THRESHOLD = 0.4
LOW_CORRELATION_THRESHOLD = 0.2

# =============================================================================
# Strategy Parameters - Mean Reversion
# =============================================================================

# Entry/exit thresholds
MEAN_REVERSION_ENTRY_ZSCORE = 2.0  # Standard threshold
MEAN_REVERSION_EXIT_ZSCORE = 0.5  # Exit near mean
MEAN_REVERSION_STOP_LOSS_ZSCORE = 3.5  # Emergency exit

# Lookback periods (hours)
MEAN_REVERSION_LOOKBACK_SHORT = 24  # 1 day
MEAN_REVERSION_LOOKBACK_MEDIUM = 168  # 1 week
MEAN_REVERSION_LOOKBACK_LONG = 720  # 1 month

# Half-life (typical range: 5-15 days)
MEAN_REVERSION_HALF_LIFE_HOURS = 240  # 10 days

# =============================================================================
# Strategy Parameters - Forecast Arbitrage
# =============================================================================

# Threshold for entry (EUR/MWh forecast error)
FORECAST_ENTRY_THRESHOLD_EUR_MWH = 5.0
FORECAST_EXIT_THRESHOLD_EUR_MWH = 2.0

# Maximum holding period (prevent indefinite positions)
MAX_HOLDING_PERIOD_HOURS = 72  # 3 days

# =============================================================================
# Strategy Parameters - Cross-Regional Arbitrage
# =============================================================================

# Spread thresholds (EUR/MWh)
ARBITRAGE_ENTRY_SPREAD_EUR_MWH = 10.0
ARBITRAGE_EXIT_SPREAD_EUR_MWH = 3.0

# Transmission costs (EUR/MWh)
TYPICAL_TRANSMISSION_COST_EUR_MWH = 2.0

# Interconnector capacity limits (MW) - France-Germany example
TYPICAL_INTERCONNECTOR_CAPACITY_MW = 3000.0

# =============================================================================
# Risk Management
# =============================================================================

# Position sizing
DEFAULT_POSITION_SIZE_FRACTION = 0.02  # 2% of capital per trade
MAX_POSITION_SIZE_FRACTION = 0.10  # 10% maximum
MIN_POSITION_SIZE_FRACTION = 0.001  # 0.1% minimum

# Stop loss
DEFAULT_STOP_LOSS_PCT = 0.02  # 2% stop loss
AGGRESSIVE_STOP_LOSS_PCT = 0.01  # 1% tight stop

# VaR calculation
VAR_CONFIDENCE_LEVEL = 0.95  # 95% confidence
CVAR_CONFIDENCE_LEVEL = 0.95  # Expected shortfall

# Maximum drawdown alerts
WARNING_DRAWDOWN_PCT = 10.0  # Warning threshold
CRITICAL_DRAWDOWN_PCT = 20.0  # Critical threshold (consider stopping)

# =============================================================================
# Walk-Forward Validation
# =============================================================================

# Period lengths (days)
DEFAULT_TRAIN_PERIOD_DAYS = 180  # 6 months
DEFAULT_TEST_PERIOD_DAYS = 60  # 2 months
DEFAULT_STEP_DAYS = 30  # 1 month rolling

# Efficiency ratio thresholds
ROBUST_EFFICIENCY_RATIO = 0.70  # ER > 0.70 indicates robust strategy
WEAK_EFFICIENCY_RATIO = 0.50  # ER < 0.50 indicates severe overfitting

# =============================================================================
# Performance Benchmarks (Realistic Targets)
# =============================================================================

# Sharpe ratio expectations (based on academic literature)
# Source: Bunn & Karakatsani (2016), Weron et al. (2014)

# Mean reversion
MEAN_REVERSION_REALISTIC_SHARPE_LOWER = 0.4
MEAN_REVERSION_REALISTIC_SHARPE_UPPER = 0.8

# Forecast arbitrage
FORECAST_ARBITRAGE_REALISTIC_SHARPE_LOWER = 0.6
FORECAST_ARBITRAGE_REALISTIC_SHARPE_UPPER = 1.0

# Cross-regional arbitrage
CROSS_REGIONAL_REALISTIC_SHARPE_LOWER = 0.3
CROSS_REGIONAL_REALISTIC_SHARPE_UPPER = 0.6

# Industry baseline (buy & hold equivalent)
INDUSTRY_BASELINE_SHARPE = 0.39

# =============================================================================
# Data Quality Thresholds
# =============================================================================

# Missing data tolerance
MAX_MISSING_DATA_PCT = 5.0  # 5% missing values tolerated
MAX_CONSECUTIVE_MISSING_HOURS = 24  # 1 day gap max

# Outlier detection (for price validation)
PRICE_OUTLIER_ZSCORE_THRESHOLD = 5.0  # > 5 sigma likely data error

# Minimum sample sizes
MIN_TRAIN_SAMPLES = 100  # Minimum training samples
MIN_TEST_SAMPLES = 20  # Minimum test samples

# =============================================================================
# Renewable Energy Parameters
# =============================================================================

# Wind turbine parameters
WIND_CUT_IN_SPEED_MS = 3.0  # Minimum wind speed for generation
WIND_RATED_SPEED_MS = 12.0  # Speed at rated power
WIND_CUT_OUT_SPEED_MS = 25.0  # Maximum speed (safety shutdown)
WIND_RATED_POWER_MW = 3.0  # Typical turbine capacity

# Solar parameters
TYPICAL_SOLAR_CAPACITY_FACTOR = 0.12  # ~12% in Northern Europe
TYPICAL_WIND_CAPACITY_FACTOR = 0.30  # ~30% in Northern Europe

# Curtailment risk threshold
RENEWABLE_SHARE_CURTAILMENT_RISK = 0.70  # >70% renewable share → curtailment risk
RENEWABLE_SHARE_NEGATIVE_PRICE_RISK = 0.80  # >80% → likely negative prices

# =============================================================================
# Stress Test Scenarios
# =============================================================================

# 2022 Energy Crisis parameters
CRISIS_2022_GAS_MULTIPLIER = 3.0  # Gas prices 3x higher
CRISIS_2022_CARBON_MULTIPLIER = 1.5  # Carbon 1.5x higher
CRISIS_2022_VOLATILITY_MULTIPLIER = 2.5  # Volatility 2.5x higher
CRISIS_2022_SPIKE_PROBABILITY = 0.05  # 5% vs normal 1-2%

# =============================================================================
# API and Data Sources
# =============================================================================

# ENTSO-E API
ENTSOE_API_RATE_LIMIT_PER_MINUTE = 400
ENTSOE_DEFAULT_TIMEOUT_SECONDS = 30
ENTSOE_MAX_RETRIES = 3
ENTSOE_RETRY_INITIAL_DELAY_SECONDS = 1.0

# =============================================================================
# Validation Functions
# =============================================================================


def validate_price(price: float, price_type: str = "electricity") -> bool:
    """Validate if price is within realistic bounds.

    Args:
        price: Price value
        price_type: "electricity", "gas", "carbon", "coal"

    Returns:
        True if price is realistic
    """
    if price_type == "electricity":
        return MIN_ELECTRICITY_PRICE_EUR_MWH <= price <= MAX_ELECTRICITY_PRICE_EUR_MWH
    elif price_type == "gas":
        return TTF_GAS_FLOOR_EUR_MWH <= price <= 500.0  # Max ~500 EUR/MWh (2022 peak)
    elif price_type == "carbon":
        return 0 <= price <= 200.0  # Max ~200 EUR/tCO2 realistic
    elif price_type == "coal":
        return 0 <= price <= 500.0  # Max ~500 EUR/tonne realistic
    else:
        raise ValueError(f"Unknown price_type: {price_type}")


def validate_sharpe_ratio(sharpe: float, strategy_type: str) -> str:
    """Validate Sharpe ratio against realistic benchmarks.

    Args:
        sharpe: Sharpe ratio
        strategy_type: "mean_reversion", "forecast_arbitrage", "cross_regional"

    Returns:
        "excellent", "good", "acceptable", "questionable", "unrealistic"
    """
    if strategy_type == "mean_reversion":
        if sharpe > 1.5:
            return "unrealistic"  # Suspiciously high
        elif sharpe > MEAN_REVERSION_REALISTIC_SHARPE_UPPER:
            return "questionable"
        elif sharpe >= MEAN_REVERSION_REALISTIC_SHARPE_LOWER:
            return "good"
        else:
            return "acceptable"

    elif strategy_type == "forecast_arbitrage":
        if sharpe > 2.0:
            return "unrealistic"
        elif sharpe > FORECAST_ARBITRAGE_REALISTIC_SHARPE_UPPER:
            return "questionable"
        elif sharpe >= FORECAST_ARBITRAGE_REALISTIC_SHARPE_LOWER:
            return "good"
        else:
            return "acceptable"

    elif strategy_type == "cross_regional":
        if sharpe > 1.0:
            return "unrealistic"
        elif sharpe > CROSS_REGIONAL_REALISTIC_SHARPE_UPPER:
            return "questionable"
        elif sharpe >= CROSS_REGIONAL_REALISTIC_SHARPE_LOWER:
            return "good"
        else:
            return "acceptable"

    else:
        raise ValueError(f"Unknown strategy_type: {strategy_type}")
