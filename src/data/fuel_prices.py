"""Fuel and carbon prices data collection and simulation.

[WARNING]  CRITICAL NOTICE: REAL DATA REQUIRED FOR PRODUCTION
================================================================================

The simulation functions in this file (simulate_ttf_gas_prices, etc.) are
ONLY for research prototyping and unit testing.

For production trading:
1. Subscribe to ICE, Bloomberg, or Refinitiv data feed
2. Use load_fuel_prices() function below
3. NEVER use simulated fuel prices for real trading decisions

Simulated fuel prices lack:
- Real correlation structures with power prices
- Geopolitical events (Russia-Ukraine, pipeline outages)
- Market microstructure (bid-ask spreads, liquidity)
- Actual supply/demand fundamentals

================================================================================

This module provides functions to load and simulate fuel prices (natural gas,
coal, carbon allowances) that are fundamental drivers of electricity prices
in European markets.

In production, this MUST interface with:
- ICE (Intercontinental Exchange) for EUA carbon prices
- TTF (Title Transfer Facility) for Dutch gas hub prices
- globalCOAL for API2 coal prices
- Bloomberg/Refinitiv for real-time data

# Parameter Sources and Calibration

All default parameters are based on historical market data and academic literature.
For production use, recalibrate using recent historical data.

## TTF Natural Gas Prices (EUR/MWh)
Sources:
- Historical data: ICE Endex, PEGAS platform
- Academic: Haldrup & Nielsen (2006), Benth et al. (2008)

Default parameters:
- base_price=30.0 EUR/MWh: Pre-2022 median, post-crisis ~80-100 EUR/MWh
- volatility=0.40: Annualized volatility from 2019-2021 data (~35-45%)
- mean_reversion_speed=0.1: Half-life ~7 days (Lucia & Schwartz, 2002)
- seasonal_amplitude=10.0 EUR/MWh: Winter-summer spread (Cartea & Figueroa, 2005)
- spike_probability=0.01: Cold spells, geopolitical events (~1-2% of days)

Calibration method: Fit Ornstein-Uhlenbeck to log-returns, estimate θ via AR(1)

## EUA Carbon Prices (EUR/tCO2)
Sources:
- Historical data: ICE ECX, EEX
- Policy: EU ETS Phase 4 (2021-2030), Fit for 55 package

Default parameters:
- base_price=80.0 EUR/tCO2: 2023-2024 average, up from ~25 EUR (2019-2020)
- trend=0.00005: ~18% annual growth (reflecting tightening cap)
- volatility=0.25: Lower than gas due to policy support

Calibration method: Fit GBM with drift to historical prices, estimate μ and σ

## Coal Prices (EUR/tonne, API2 Rotterdam)
Sources:
- Historical data: globalCOAL, Argus
- Correlation with gas: ~0.6-0.7 (fuel substitution effect)

Default parameters:
- base_price=100.0 EUR/tonne: 2019-2021 average
- volatility=0.35: Moderate volatility
- gas_correlation=0.65: Fuel switching in power generation

Calibration method: Fit correlated GBM, estimate covariance with gas prices

## Spreads (EUR/MWh)
- Spark Spread = Power - (Gas/0.55) - (Carbon × 0.35)
  → Gas CCGT efficiency: 55%, emission: 0.35 tCO2/MWh
- Dark Spread = Power - (Coal/0.38) - (Carbon × 0.95)
  → Coal plant efficiency: 38%, emission: 0.95 tCO2/MWh
- Clean Spread = Spark Spread - Dark Spread
  → Positive: gas is marginal; Negative: coal is marginal

## Production Calibration Workflow
1. Fetch last 3 years of daily prices from APIs
2. Estimate volatility: σ = std(log_returns) × sqrt(252)
3. Estimate mean reversion: θ = -log(autocorr(1)) × 252
4. Estimate correlation: ρ = corr(gas_returns, coal_returns)
5. Update parameters in production config file
6. Re-run backtests to validate strategy robustness

## References
- Benth, F.E., Kallsen, J., Meyer-Brandis, T. (2008). A non-Gaussian Ornstein-Uhlenbeck process for electricity spot price modeling.
- Lucia, J., Schwartz, E. (2002). Electricity prices and power derivatives: Evidence from the Nordic Power Exchange.
- Cartea, A., Figueroa, M. (2005). Pricing in electricity markets: A mean reverting jump diffusion model with seasonality.
"""

from typing import Optional, Tuple
from pathlib import Path
import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# DEPRECATION WARNING FOR SIMULATED FUEL PRICES
# ============================================================================


def _deprecation_warning_fuel_simulation():
    """Warn when using simulated fuel prices."""
    warnings.warn(
        "\n" + "=" * 80 + "\n"
        "[WARNING]  WARNING: Using SIMULATED fuel prices!\n"
        "\n"
        "Simulated fuel prices are NOT suitable for production trading.\n"
        "They lack:\n"
        "  - Real correlations with power prices\n"
        "  - Geopolitical events (Russia-Ukraine, pipeline outages)\n"
        "  - Market microstructure (bid-ask spreads, liquidity)\n"
        "  - Actual supply/demand fundamentals\n"
        "\n"
        "For production, you MUST use real data from:\n"
        "  - ICE (Intercontinental Exchange)\n"
        "  - Bloomberg Terminal\n"
        "  - Refinitiv Eikon\n"
        "\n"
        "See load_fuel_prices() function for real data loader.\n" + "=" * 80,
        UserWarning,
        stacklevel=3,
    )


# ============================================================================
# REAL DATA LOADER (REQUIRED FOR PRODUCTION)
# ============================================================================


def load_fuel_prices(
    start_date: str,
    end_date: str,
    data_dir: str = "data/external/fuel_prices",
    source: str = "csv",
) -> pd.DataFrame:
    """Load REAL fuel prices from external data source.

    This function loads historical fuel prices from either:
    - CSV files (for backtesting with downloaded data)
    - Live API (for production inference)

    Required Data Sources:
    ----------------------
    1. TTF Natural Gas: ICE Endex, PEGAS
       - Contract: TTF Natural Gas Futures (Month-ahead)
       - Units: EUR/MWh
       - Frequency: Daily settlement prices

    2. EUA Carbon Allowances: ICE ECX
       - Contract: EUA Futures (Dec expiry)
       - Units: EUR/tCO2
       - Frequency: Daily settlement prices

    3. Coal API2: globalCOAL, Argus
       - Contract: API2 CIF ARA (Rotterdam)
       - Units: EUR/tonne
       - Frequency: Daily assessments

    File Format (CSV):
    -----------------
    Expected directory structure:
        data/external/fuel_prices/
        ├── ttf_gas_eur_mwh.csv       # TTF gas prices
        ├── eua_carbon_eur_tco2.csv   # EUA carbon prices
        └── coal_api2_eur_tonne.csv   # Coal prices

    Each CSV must have columns: [date, price]
    Date format: YYYY-MM-DD

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        data_dir: Directory containing fuel price CSV files
        source: Data source ('csv' or 'api')

    Returns:
        pd.DataFrame with columns:
            - datetime: timestamp
            - ttf_gas_price: EUR/MWh
            - eua_carbon_price: EUR/tCO2
            - coal_price: EUR/tonne

    Raises:
        FileNotFoundError: If CSV files not found
        ValueError: If data quality checks fail

    Example:
        >>> fuel_prices = load_fuel_prices("2023-01-01", "2024-12-31")
        >>> print(fuel_prices.head())
    """
    data_path = Path(data_dir)

    # File paths
    ttf_file = data_path / "ttf_gas_eur_mwh.csv"
    eua_file = data_path / "eua_carbon_eur_tco2.csv"
    coal_file = data_path / "coal_api2_eur_tonne.csv"

    # Check if files exist
    missing_files = []
    for file in [ttf_file, eua_file, coal_file]:
        if not file.exists():
            missing_files.append(str(file))

    if missing_files:
        error_msg = (
            f"[ERROR] CRITICAL ERROR: Real fuel price data files not found!\n"
            f"\n"
            f"Missing files:\n"
        )
        for f in missing_files:
            error_msg += f"  - {f}\n"

        error_msg += (
            f"\n"
            f"To fix this:\n"
            f"1. Download historical fuel prices from ICE/Bloomberg\n"
            f"2. Place CSV files in: {data_dir}\n"
            f"3. Required format: columns [date, price]\n"
            f"\n"
            f"Data Sources:\n"
            f"  - TTF Gas: https://www.theice.com/products/27996665/dutch-ttf-gas-futures\n"
            f"  - EUA Carbon: https://www.theice.com/products/197/eua-futures\n"
            f"  - Coal API2: https://www.globalcoal.com/ or Argus Media\n"
            f"\n"
            f"Temporary workaround (TESTING ONLY):\n"
            f"  - Use generate_fuel_price_features() with simulated=True\n"
            f"  - WARNING: Results will not be production-valid!\n"
        )
        raise FileNotFoundError(error_msg)

    # Load data
    logger.info(f"Loading real fuel prices from {data_dir}")

    ttf_df = pd.read_csv(ttf_file, parse_dates=["date"])
    eua_df = pd.read_csv(eua_file, parse_dates=["date"])
    coal_df = pd.read_csv(coal_file, parse_dates=["date"])

    # Rename columns
    ttf_df = ttf_df.rename(columns={"date": "datetime", "price": "ttf_gas_price"})
    eua_df = eua_df.rename(columns={"date": "datetime", "price": "eua_carbon_price"})
    coal_df = coal_df.rename(columns={"date": "datetime", "price": "coal_price"})

    # Merge on datetime
    fuel_df = ttf_df.merge(eua_df, on="datetime", how="outer")
    fuel_df = fuel_df.merge(coal_df, on="datetime", how="outer")

    # Filter date range
    fuel_df["datetime"] = pd.to_datetime(fuel_df["datetime"])
    mask = (fuel_df["datetime"] >= start_date) & (fuel_df["datetime"] <= end_date)
    fuel_df = fuel_df[mask].copy()

    # Forward fill missing values (weekends, holidays)
    fuel_df = fuel_df.sort_values("datetime")
    fuel_df[["ttf_gas_price", "eua_carbon_price", "coal_price"]] = fuel_df[
        ["ttf_gas_price", "eua_carbon_price", "coal_price"]
    ].fillna(method="ffill")

    # Data quality checks
    for col in ["ttf_gas_price", "eua_carbon_price", "coal_price"]:
        missing_pct = fuel_df[col].isna().sum() / len(fuel_df)
        if missing_pct > 0.1:
            raise ValueError(
                f"Too many missing values in {col}: {fuel_df[col].isna().sum()} / {len(fuel_df)} ({missing_pct*100:.1f}%)"
            )

    # Validate price ranges
    if not (5 <= fuel_df["ttf_gas_price"].mean() <= 200):
        logger.warning(
            f"[WARNING]  TTF gas price outside expected range: {fuel_df['ttf_gas_price'].mean():.2f} EUR/MWh"
        )

    if not (20 <= fuel_df["eua_carbon_price"].mean() <= 150):
        logger.warning(
            f"[WARNING]  EUA carbon price outside expected range: {fuel_df['eua_carbon_price'].mean():.2f} EUR/tCO2"
        )

    if not (50 <= fuel_df["coal_price"].mean() <= 300):
        logger.warning(
            f"[WARNING]  Coal price outside expected range: {fuel_df['coal_price'].mean():.2f} EUR/tonne"
        )

    logger.info(
        f"[OK] Loaded {len(fuel_df)} days of real fuel prices\n"
        f"   TTF Gas: {fuel_df['ttf_gas_price'].mean():.2f} EUR/MWh (±{fuel_df['ttf_gas_price'].std():.2f})\n"
        f"   EUA Carbon: {fuel_df['eua_carbon_price'].mean():.2f} EUR/tCO2 (±{fuel_df['eua_carbon_price'].std():.2f})\n"
        f"   Coal API2: {fuel_df['coal_price'].mean():.2f} EUR/tonne (±{fuel_df['coal_price'].std():.2f})"
    )

    return fuel_df


def calculate_spreads(
    power_price: pd.Series,
    ttf_gas_price: pd.Series,
    eua_carbon_price: pd.Series,
    coal_price: pd.Series,
) -> pd.DataFrame:
    """Calculate spark and dark spreads from fuel prices.

    Spark Spread = Power - (Gas / η_gas) - (Carbon × emission_gas)
    Dark Spread = Power - (Coal / η_coal) - (Carbon × emission_coal)

    Where:
        η_gas = 0.55 (CCGT efficiency)
        emission_gas = 0.35 tCO2/MWh
        η_coal = 0.38 (coal plant efficiency)
        emission_coal = 0.95 tCO2/MWh

    Args:
        power_price: Electricity price (EUR/MWh)
        ttf_gas_price: TTF gas price (EUR/MWh)
        eua_carbon_price: EUA carbon price (EUR/tCO2)
        coal_price: Coal price (EUR/tonne)

    Returns:
        DataFrame with spark_spread, dark_spread, clean_spark_spread
    """
    # Efficiencies and emissions
    CCGT_EFFICIENCY = 0.55
    COAL_EFFICIENCY = 0.38
    GAS_EMISSION = 0.35  # tCO2/MWh
    COAL_EMISSION = 0.95  # tCO2/MWh
    COAL_HEAT_RATE = 7.0  # MWh per tonne (API2 coal)

    # Spark spread (gas CCGT)
    gas_cost = ttf_gas_price / CCGT_EFFICIENCY
    gas_carbon_cost = eua_carbon_price * GAS_EMISSION
    spark_spread = power_price - gas_cost - gas_carbon_cost

    # Dark spread (coal plant)
    coal_cost = (coal_price / COAL_HEAT_RATE) / COAL_EFFICIENCY
    coal_carbon_cost = eua_carbon_price * COAL_EMISSION
    dark_spread = power_price - coal_cost - coal_carbon_cost

    # Clean spark spread (spark - dark)
    clean_spark_spread = spark_spread - dark_spread

    return pd.DataFrame(
        {
            "spark_spread": spark_spread,
            "dark_spread": dark_spread,
            "clean_spark_spread": clean_spark_spread,
        }
    )


# ============================================================================
# SIMULATION FUNCTIONS (TESTING ONLY - NOT FOR PRODUCTION)
# ============================================================================


def simulate_ttf_gas_prices(
    dates: pd.DatetimeIndex,
    base_price: float = 30.0,
    volatility: float = 0.40,
    mean_reversion_speed: float = 0.1,
    seasonal_amplitude: float = 10.0,
    spike_probability: float = 0.01,
    random_seed: Optional[int] = None,
) -> pd.Series:
    """Simulate TTF natural gas prices (EUR/MWh).

    TTF (Title Transfer Facility) is the main European gas hub benchmark.
    Prices exhibit:
    - Strong seasonality (winter premium)
    - High volatility (supply shocks, geopolitics)
    - Mean reversion
    - Occasional extreme spikes (cold spells, supply disruptions)

    Args:
        dates: Datetime index for simulation.
        base_price: Long-term equilibrium price (EUR/MWh).
        volatility: Annual volatility (default 40%).
        mean_reversion_speed: Speed of reversion to base price.
        seasonal_amplitude: Magnitude of seasonal variation.
        spike_probability: Probability of supply shock spikes.
        random_seed: Random seed for reproducibility.

    Returns:
        pd.Series: Simulated TTF gas prices indexed by datetime.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    n = len(dates)

    # Seasonal component (winter premium)
    # Handle both DatetimeIndex and Series
    if isinstance(dates, pd.Series):
        dates = pd.DatetimeIndex(dates)

    day_of_year = dates.dayofyear
    seasonal = seasonal_amplitude * np.cos(2 * np.pi * (day_of_year - 15) / 365)

    # Mean-reverting component (Ornstein-Uhlenbeck process)
    dt = 1 / 365  # Daily timesteps
    price = np.zeros(n)
    price[0] = base_price

    for t in range(1, n):
        drift = mean_reversion_speed * (base_price - price[t - 1]) * dt
        diffusion = volatility * np.sqrt(dt) * np.random.randn()
        price[t] = price[t - 1] + drift + diffusion * price[t - 1]

    # Add seasonal component
    price = price + seasonal

    # Add supply shock spikes
    spikes = np.random.rand(n) < spike_probability
    spike_magnitude = np.random.exponential(50, n) * spikes
    price = price + spike_magnitude

    # Ensure non-negative prices
    price = np.maximum(price, 5.0)

    return pd.Series(price, index=dates, name="ttf_gas_price")


def simulate_eua_carbon_prices(
    dates: pd.DatetimeIndex,
    base_price: float = 80.0,
    trend: float = 0.00005,
    volatility: float = 0.25,
    random_seed: Optional[int] = None,
) -> pd.Series:
    """Simulate EUA carbon allowance prices (EUR/tCO2).

    EUA (EU Allowance) prices from EU ETS (Emissions Trading System).
    Characteristics:
    - Upward trend (tightening cap, higher ambition)
    - Moderate volatility
    - Policy-driven dynamics

    Args:
        dates: Datetime index for simulation.
        base_price: Starting price (EUR/tCO2).
        trend: Daily drift rate (upward trend).
        volatility: Annual volatility (default 25%).
        random_seed: Random seed for reproducibility.

    Returns:
        pd.Series: Simulated EUA carbon prices indexed by datetime.
    """
    if random_seed is not None:
        np.random.seed(random_seed + 1)  # Different seed than gas

    # Handle both DatetimeIndex and Series
    if isinstance(dates, pd.Series):
        dates = pd.DatetimeIndex(dates)

    n = len(dates)

    # Geometric Brownian Motion with positive drift
    dt = 1 / 365
    price = np.zeros(n)
    price[0] = base_price

    for t in range(1, n):
        drift = trend * price[t - 1] * dt
        diffusion = volatility * np.sqrt(dt) * np.random.randn() * price[t - 1]
        price[t] = price[t - 1] + drift + diffusion

    # Ensure non-negative
    price = np.maximum(price, 20.0)

    return pd.Series(price, index=dates, name="eua_carbon_price")


def simulate_coal_prices(
    dates: pd.DatetimeIndex,
    base_price: float = 100.0,
    volatility: float = 0.30,
    correlation_with_gas: float = 0.6,
    ttf_prices: Optional[pd.Series] = None,
    random_seed: Optional[int] = None,
) -> pd.Series:
    """Simulate API2 coal prices (USD/tonne, converted to EUR/MWh equivalent).

    API2 is the main European coal benchmark (ARA ports - Amsterdam-Rotterdam-Antwerp).
    Prices are correlated with gas prices (fuel substitution).

    Args:
        dates: Datetime index for simulation.
        base_price: Base coal price (EUR/MWh equivalent).
        volatility: Annual volatility (default 30%).
        correlation_with_gas: Correlation with TTF gas prices.
        ttf_prices: Optional TTF prices for correlation.
        random_seed: Random seed for reproducibility.

    Returns:
        pd.Series: Simulated coal prices indexed by datetime.
    """
    if random_seed is not None:
        np.random.seed(random_seed + 2)

    # Handle both DatetimeIndex and Series
    if isinstance(dates, pd.Series):
        dates = pd.DatetimeIndex(dates)

    n = len(dates)

    # Base price process
    dt = 1 / 365
    price = np.zeros(n)
    price[0] = base_price

    for t in range(1, n):
        # Correlated component with gas (if provided)
        if ttf_prices is not None and t > 0:
            gas_return = (
                ttf_prices.iloc[t] - ttf_prices.iloc[t - 1]
            ) / ttf_prices.iloc[t - 1]
            correlated_shock = correlation_with_gas * gas_return
        else:
            correlated_shock = 0

        # Independent component
        independent_shock = (
            np.sqrt(1 - correlation_with_gas**2)
            * volatility
            * np.sqrt(dt)
            * np.random.randn()
        )

        price[t] = price[t - 1] * (1 + correlated_shock + independent_shock)

    # Ensure non-negative
    price = np.maximum(price, 20.0)

    return pd.Series(price, index=dates, name="coal_price")


def calculate_spark_spread(
    power_price: pd.Series,
    gas_price: pd.Series,
    carbon_price: pd.Series,
    efficiency: float = 0.55,
    emission_factor: float = 0.35,
) -> pd.Series:
    """Calculate spark spread (gas-to-power profitability).

    Spark Spread = Power Price - (Gas Price / Efficiency) - (Carbon Price × Emission Factor)

    This represents the gross margin for gas-fired power generation.

    Args:
        power_price: Electricity prices (EUR/MWh).
        gas_price: Gas prices (EUR/MWh).
        carbon_price: Carbon prices (EUR/tCO2).
        efficiency: Gas plant efficiency (default 55% for CCGT).
        emission_factor: CO2 emissions (tCO2/MWh, default 0.35 for CCGT).

    Returns:
        pd.Series: Spark spread (EUR/MWh).
    """
    variable_cost = (gas_price / efficiency) + (carbon_price * emission_factor)
    spark_spread = power_price - variable_cost

    return spark_spread.rename("spark_spread")


def calculate_dark_spread(
    power_price: pd.Series,
    coal_price: pd.Series,
    carbon_price: pd.Series,
    efficiency: float = 0.38,
    emission_factor: float = 0.95,
) -> pd.Series:
    """Calculate dark spread (coal-to-power profitability).

    Dark Spread = Power Price - (Coal Price / Efficiency) - (Carbon Price × Emission Factor)

    This represents the gross margin for coal-fired power generation.

    Args:
        power_price: Electricity prices (EUR/MWh).
        coal_price: Coal prices (EUR/MWh equivalent).
        carbon_price: Carbon prices (EUR/tCO2).
        efficiency: Coal plant efficiency (default 38%).
        emission_factor: CO2 emissions (tCO2/MWh, default 0.95 for coal).

    Returns:
        pd.Series: Dark spread (EUR/MWh).
    """
    variable_cost = (coal_price / efficiency) + (carbon_price * emission_factor)
    dark_spread = power_price - variable_cost

    return dark_spread.rename("dark_spread")


def calculate_clean_spread(
    spark_spread: pd.Series, dark_spread: pd.Series
) -> pd.Series:
    """Calculate clean spread (difference between spark and dark spread).

    Clean Spread = Spark Spread - Dark Spread

    Indicates the relative profitability of gas vs coal generation.
    Positive → gas more profitable, negative → coal more profitable.

    Args:
        spark_spread: Gas-to-power spread.
        dark_spread: Coal-to-power spread.

    Returns:
        pd.Series: Clean spread (EUR/MWh).
    """
    clean_spread = spark_spread - dark_spread
    return clean_spread.rename("clean_spread")


def generate_fuel_price_features(
    dates: pd.DatetimeIndex,
    power_prices: Optional[pd.Series] = None,
    data_dir: str = "data/external/fuel_prices",
    simulated: bool = False,  # [OK] NEW: Explicit flag (defaults to REAL data)
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate comprehensive fuel price dataset with all features.

    [WARNING] CRITICAL: Uses REAL fuel prices by default.
    Set simulated=True ONLY for unit testing.

    This function loads real fuel prices (or simulates for testing) and calculates
    derived features (spreads, ratios) that are predictive of electricity prices.

    Args:
        dates: Datetime index for features.
        power_prices: Optional electricity prices for spread calculation.
        data_dir: Directory with real fuel price CSVs.
        simulated: If True, use simulation (TESTING ONLY).
        random_seed: Random seed for simulation (if simulated=True).

    Returns:
        pd.DataFrame: Complete fuel price feature set with columns:
            - ttf_gas_price: TTF natural gas price (EUR/MWh)
            - eua_carbon_price: EUA carbon price (EUR/tCO2)
            - coal_price: API2 coal price (EUR/tonne)
            - spark_spread: Gas-to-power margin (EUR/MWh) [if power_prices provided]
            - dark_spread: Coal-to-power margin (EUR/MWh) [if power_prices provided]
            - clean_spark_spread: Gas vs coal profitability [if power_prices provided]
            - gas_carbon_ratio: Gas price / carbon price
            - coal_carbon_ratio: Coal price / carbon price
    """
    if simulated:
        # [OK] EXPLICIT WARNING
        _deprecation_warning_fuel_simulation()

        # Use legacy simulation functions
        ttf = simulate_ttf_gas_prices(dates, random_seed=random_seed)
        eua = simulate_eua_carbon_prices(dates, random_seed=random_seed)
        coal = simulate_coal_prices(dates, ttf_prices=ttf, random_seed=random_seed)

        fuel_df = pd.DataFrame(
            {
                "datetime": dates,
                "ttf_gas_price": ttf.values,
                "eua_carbon_price": eua.values,
                "coal_price": coal.values,
            }
        )
    else:
        # [OK] LOAD REAL DATA
        start_date = dates.min().strftime("%Y-%m-%d")
        end_date = dates.max().strftime("%Y-%m-%d")

        fuel_df = load_fuel_prices(start_date, end_date, data_dir=data_dir)

        # Align to requested dates (interpolate if needed)
        fuel_df = fuel_df.set_index("datetime")
        fuel_df = fuel_df.reindex(dates, method="ffill")
        fuel_df = fuel_df.reset_index()

    # Calculate spreads if power prices provided
    if power_prices is not None:
        # Ensure same index
        power_prices = power_prices.reindex(dates)

        spreads = calculate_spreads(
            power_prices,
            fuel_df["ttf_gas_price"],
            fuel_df["eua_carbon_price"],
            fuel_df["coal_price"],
        )
        fuel_df = pd.concat([fuel_df, spreads], axis=1)

    # Add fuel-carbon ratios (important for merit order)
    fuel_df["gas_carbon_ratio"] = fuel_df["ttf_gas_price"] / fuel_df["eua_carbon_price"]
    fuel_df["coal_carbon_ratio"] = fuel_df["coal_price"] / fuel_df["eua_carbon_price"]

    # Add fuel price changes (momentum features)
    fuel_df["ttf_gas_pct_change"] = fuel_df["ttf_gas_price"].pct_change()
    fuel_df["eua_carbon_pct_change"] = fuel_df["eua_carbon_price"].pct_change()
    fuel_df["coal_pct_change"] = fuel_df["coal_price"].pct_change()

    return fuel_df


def load_or_simulate_fuel_prices(
    dates: pd.DatetimeIndex,
    power_prices: Optional[pd.Series] = None,
    use_real_data: bool = False,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Load real fuel prices or simulate if not available.

    Args:
        dates: Datetime index.
        power_prices: Optional electricity prices for spread calculation.
        use_real_data: If True, attempt to load real data (not yet implemented).
        random_seed: Random seed for simulation.

    Returns:
        pd.DataFrame: Fuel price features.
    """
    if use_real_data:
        # TODO: Implement real data loading from APIs
        # - ICE API for EUA carbon
        # - Platts/ICIS for TTF gas
        # - globalCOAL for API2 coal
        raise NotImplementedError(
            "Real fuel price data loading not yet implemented.\n"
            "Use use_real_data=False for simulated data."
        )
    else:
        return generate_fuel_price_features(dates, power_prices, random_seed)
