"""Fuel and carbon prices data collection and simulation.

This module provides functions to load and simulate fuel prices (natural gas,
coal, carbon allowances) that are fundamental drivers of electricity prices
in European markets.

In production, this would interface with:
- ICE (Intercontinental Exchange) for EUA carbon prices
- TTF (Title Transfer Facility) for Dutch gas hub prices
- globalCOAL for API2 coal prices
- EIA/Bloomberg for Brent oil prices

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

import numpy as np
import pandas as pd


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
            gas_return = (ttf_prices.iloc[t] - ttf_prices.iloc[t - 1]) / ttf_prices.iloc[
                t - 1
            ]
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
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate comprehensive fuel price dataset with all features.

    This function simulates all fuel prices and calculates derived features
    (spreads, ratios) that are predictive of electricity prices.

    Args:
        dates: Datetime index for simulation.
        power_prices: Optional electricity prices for spread calculation.
        random_seed: Random seed for reproducibility.

    Returns:
        pd.DataFrame: Complete fuel price feature set with columns:
            - ttf_gas_price: TTF natural gas price (EUR/MWh)
            - eua_carbon_price: EUA carbon price (EUR/tCO2)
            - coal_price: API2 coal price (EUR/MWh equivalent)
            - spark_spread: Gas-to-power margin (EUR/MWh) [if power_prices provided]
            - dark_spread: Coal-to-power margin (EUR/MWh) [if power_prices provided]
            - clean_spread: Gas vs coal profitability (EUR/MWh) [if power_prices provided]
            - gas_carbon_ratio: Gas price / carbon price
            - coal_carbon_ratio: Coal price / carbon price
    """
    # Simulate fuel prices
    ttf = simulate_ttf_gas_prices(dates, random_seed=random_seed)
    eua = simulate_eua_carbon_prices(dates, random_seed=random_seed)
    coal = simulate_coal_prices(dates, ttf_prices=ttf, random_seed=random_seed)

    # Create DataFrame
    df = pd.DataFrame({"ttf_gas_price": ttf, "eua_carbon_price": eua, "coal_price": coal})

    # Add spreads if power prices provided
    if power_prices is not None:
        # Ensure same index
        power_prices = power_prices.reindex(dates)

        df["spark_spread"] = calculate_spark_spread(power_prices, ttf, eua)
        df["dark_spread"] = calculate_dark_spread(power_prices, coal, eua)
        df["clean_spread"] = calculate_clean_spread(df["spark_spread"], df["dark_spread"])

    # Add fuel-carbon ratios (important for merit order)
    df["gas_carbon_ratio"] = df["ttf_gas_price"] / df["eua_carbon_price"]
    df["coal_carbon_ratio"] = df["coal_price"] / df["eua_carbon_price"]

    # Add fuel price changes (momentum features)
    df["ttf_gas_pct_change"] = df["ttf_gas_price"].pct_change()
    df["eua_carbon_pct_change"] = df["eua_carbon_price"].pct_change()
    df["coal_pct_change"] = df["coal_price"].pct_change()

    return df


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
