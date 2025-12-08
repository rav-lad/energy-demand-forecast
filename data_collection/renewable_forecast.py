"""Renewable energy production forecast for price forecasting.

This module calculates wind and solar production forecasts based on:
- Weather forecasts (wind speed, solar radiation)
- Installed capacity data
- Power curves and efficiency factors
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# France renewable capacity (approximate 2024 values)
# Source: RTE Bilan électrique
WIND_CAPACITY_MW = 22000  # Onshore + offshore
SOLAR_CAPACITY_MW = 17000  # PV installed


def calculate_wind_power_from_speed(
    wind_speed_ms: np.ndarray,
    capacity_mw: float = WIND_CAPACITY_MW,
    hub_height_m: float = 100
) -> np.ndarray:
    """Calculate wind power production from wind speed using power curve.

    Uses a simplified power curve for wind turbines:
    - Cut-in: 3 m/s
    - Rated: 12 m/s
    - Cut-out: 25 m/s

    Args:
        wind_speed_ms: Wind speed in m/s (at hub height)
        capacity_mw: Installed wind capacity in MW
        hub_height_m: Hub height (affects capacity factor)

    Returns:
        Wind production in MW
    """
    power_curve = np.zeros_like(wind_speed_ms, dtype=float)

    # Cut-in wind speed (3 m/s)
    cut_in = 3.0
    # Rated wind speed (12 m/s) - full power
    rated = 12.0
    # Cut-out wind speed (25 m/s) - shutdown
    cut_out = 25.0

    # Below cut-in: no production
    mask_cutout = wind_speed_ms >= cut_out
    power_curve[mask_cutout] = 0

    # Between cut-in and rated: cubic increase
    mask_ramp = (wind_speed_ms >= cut_in) & (wind_speed_ms < rated)
    power_curve[mask_ramp] = ((wind_speed_ms[mask_ramp] - cut_in) / (rated - cut_in)) ** 3

    # Above rated: full power
    mask_rated = (wind_speed_ms >= rated) & (wind_speed_ms < cut_out)
    power_curve[mask_rated] = 1.0

    # Apply capacity and availability factor (wind farms not always at 100%)
    availability = 0.95  # 95% availability
    production_mw = power_curve * capacity_mw * availability

    return production_mw


def calculate_solar_power_from_radiation(
    solar_radiation_wm2: np.ndarray,
    capacity_mw: float = SOLAR_CAPACITY_MW,
    hour: np.ndarray = None
) -> np.ndarray:
    """Calculate solar power production from solar radiation.

    Args:
        solar_radiation_wm2: Solar radiation in W/m²
        capacity_mw: Installed solar capacity in MW
        hour: Hour of day (0-23) - needed to zero out nighttime

    Returns:
        Solar production in MW
    """
    # Standard test condition (STC): 1000 W/m²
    stc_radiation = 1000.0

    # Panel efficiency (typical ~20% for modern panels)
    panel_efficiency = 0.20

    # System losses (wiring, inverter, temperature, soiling)
    system_efficiency = 0.85

    # Performance ratio (real-world vs. ideal)
    performance_ratio = 0.80

    # Calculate production
    production_fraction = (solar_radiation_wm2 / stc_radiation)

    # Apply all efficiency factors
    production_mw = (
        production_fraction
        * capacity_mw
        * performance_ratio
    )

    # Zero out nighttime (if hour provided)
    if hour is not None:
        nighttime_mask = (hour < 6) | (hour > 20)
        production_mw[nighttime_mask] = 0

    # Ensure non-negative
    production_mw = np.maximum(production_mw, 0)

    return production_mw


def calculate_renewable_forecasts(
    df: pd.DataFrame,
    wind_capacity_mw: float = WIND_CAPACITY_MW,
    solar_capacity_mw: float = SOLAR_CAPACITY_MW
) -> pd.DataFrame:
    """Calculate wind and solar production forecasts from weather forecasts.

    Args:
        df: DataFrame with weather forecast columns:
            - wind100m_forecast (or wind10m_forecast)
            - solar_radiation_forecast
            - datetime_hour
        wind_capacity_mw: Installed wind capacity
        solar_capacity_mw: Installed solar capacity

    Returns:
        DataFrame with renewable production forecast columns
    """
    df = df.copy()

    # Wind production forecast
    if "wind100m_forecast" in df.columns:
        wind_speed = df["wind100m_forecast"].values
    elif "wind10m_forecast" in df.columns:
        # Scale 10m wind to 100m (roughness length approximation)
        wind_speed = df["wind10m_forecast"].values * (100/10) ** 0.2
    else:
        logger.warning("No wind speed forecast available, using dummy values")
        wind_speed = np.full(len(df), 8.0)  # Typical average

    df["wind_production_forecast_mw"] = calculate_wind_power_from_speed(
        wind_speed,
        capacity_mw=wind_capacity_mw
    )

    # Solar production forecast
    if "solar_radiation_forecast" in df.columns:
        solar_radiation = df["solar_radiation_forecast"].values
    else:
        logger.warning("No solar radiation forecast available, using dummy values")
        solar_radiation = np.full(len(df), 200.0)

    hour = pd.to_datetime(df["datetime_hour"]).dt.hour.values if "datetime_hour" in df.columns else None

    df["solar_production_forecast_mw"] = calculate_solar_power_from_radiation(
        solar_radiation,
        capacity_mw=solar_capacity_mw,
        hour=hour
    )

    # Total renewable production
    df["renewable_production_forecast_mw"] = (
        df["wind_production_forecast_mw"] +
        df["solar_production_forecast_mw"]
    )

    # Renewable penetration (as % of typical load)
    typical_load_mw = 50000  # France typical load
    df["renewable_penetration_pct"] = (
        df["renewable_production_forecast_mw"] / typical_load_mw * 100
    )

    logger.info(
        f"✅ Calculated renewable production forecasts\n"
        f"   Mean wind production: {df['wind_production_forecast_mw'].mean():.0f} MW "
        f"({df['wind_production_forecast_mw'].mean()/wind_capacity_mw*100:.1f}% capacity factor)\n"
        f"   Mean solar production: {df['solar_production_forecast_mw'].mean():.0f} MW "
        f"({df['solar_production_forecast_mw'].mean()/solar_capacity_mw*100:.1f}% capacity factor)\n"
        f"   Mean renewable penetration: {df['renewable_penetration_pct'].mean():.1f}%"
    )

    return df


def calculate_net_load_forecast(
    df: pd.DataFrame,
    load_col: str = "load_forecast",
    renewable_col: str = "renewable_production_forecast_mw"
) -> pd.DataFrame:
    """Calculate net load (load minus renewable production).

    Net load represents the demand that must be met by dispatchable generation
    (gas, coal, nuclear, hydro). This is a KEY driver of electricity prices.

    Args:
        df: DataFrame with load and renewable forecasts
        load_col: Name of load forecast column
        renewable_col: Name of renewable production column

    Returns:
        DataFrame with net_load_forecast_mw column
    """
    df = df.copy()

    if load_col not in df.columns:
        raise ValueError(f"Load column '{load_col}' not found in DataFrame")

    if renewable_col not in df.columns:
        raise ValueError(f"Renewable column '{renewable_col}' not found in DataFrame")

    # Net load = Total load - Renewable production
    df["net_load_forecast_mw"] = df[load_col] - df[renewable_col]

    # Ensure non-negative (in extreme cases, renewables can exceed load)
    # When this happens, prices often go negative
    df["net_load_forecast_mw"] = np.maximum(df["net_load_forecast_mw"], 0)

    # Calculate renewable curtailment (when renewables > load)
    df["renewable_curtailment_mw"] = np.maximum(
        df[renewable_col] - df[load_col],
        0
    )

    # Renewable coverage (what % of load is covered by renewables)
    df["renewable_coverage_pct"] = np.minimum(
        df[renewable_col] / df[load_col] * 100,
        100
    )

    logger.info(
        f"✅ Calculated net load forecast\n"
        f"   Mean net load: {df['net_load_forecast_mw'].mean():.0f} MW\n"
        f"   Mean renewable coverage: {df['renewable_coverage_pct'].mean():.1f}%\n"
        f"   Hours with curtailment: {(df['renewable_curtailment_mw'] > 0).sum()} "
        f"({(df['renewable_curtailment_mw'] > 0).sum()/len(df)*100:.1f}%)"
    )

    return df


def add_renewable_features(
    df: pd.DataFrame,
    include_net_load: bool = True
) -> pd.DataFrame:
    """Add all renewable-related features for price forecasting.

    This is a convenience function that combines:
    1. Renewable production forecasts
    2. Net load calculation
    3. Derived features (penetration, coverage, etc.)

    Args:
        df: DataFrame with weather forecasts
        include_net_load: Whether to calculate net load (requires load forecast)

    Returns:
        DataFrame with all renewable features
    """
    # Calculate renewable production
    df = calculate_renewable_forecasts(df)

    # Calculate net load if load forecast available
    if include_net_load and "load_forecast" in df.columns:
        df = calculate_net_load_forecast(df)

    # Add derived features useful for price modeling
    # Wind capacity factor (actual / capacity)
    df["wind_capacity_factor"] = (
        df["wind_production_forecast_mw"] / WIND_CAPACITY_MW
    )

    # Solar capacity factor
    df["solar_capacity_factor"] = (
        df["solar_production_forecast_mw"] / SOLAR_CAPACITY_MW
    )

    # Residual load intensity (net load relative to typical)
    typical_load = 50000
    df["residual_load_intensity"] = df.get("net_load_forecast_mw", df.get("load_forecast", typical_load)) / typical_load

    return df


if __name__ == "__main__":
    # Test the module
    logging.basicConfig(level=logging.INFO)

    print("\n" + "="*80)
    print("TESTING RENEWABLE FORECAST MODULE")
    print("="*80)

    # Create test data
    dates = pd.date_range("2024-01-01", "2024-01-07", freq="H")

    df = pd.DataFrame({
        "datetime_hour": dates,
        "wind100m_forecast": np.random.uniform(2, 20, len(dates)),
        "solar_radiation_forecast": np.maximum(0, np.random.normal(300, 200, len(dates)) * (
            (dates.hour >= 6) & (dates.hour <= 20)
        )),
        "load_forecast": np.random.uniform(40000, 60000, len(dates))
    })

    # Calculate renewable forecasts
    df = add_renewable_features(df, include_net_load=True)

    print("\n📊 Sample output:")
    print(df[["datetime_hour", "wind_production_forecast_mw", "solar_production_forecast_mw",
              "renewable_production_forecast_mw", "net_load_forecast_mw"]].head(24))

    print("\n📈 Daily statistics:")
    daily = df.groupby(df["datetime_hour"].dt.date).agg({
        "wind_production_forecast_mw": "mean",
        "solar_production_forecast_mw": "mean",
        "renewable_coverage_pct": "mean",
        "renewable_curtailment_mw": "sum"
    })
    print(daily)

    # Test power curves
    print("\n" + "="*80)
    print("TESTING WIND POWER CURVE")
    print("="*80)

    wind_speeds = np.arange(0, 30, 1)
    wind_power = calculate_wind_power_from_speed(wind_speeds, capacity_mw=1000)

    print("\nWind speed (m/s) -> Power (MW):")
    for ws, wp in zip(wind_speeds[::3], wind_power[::3]):
        print(f"  {ws:5.1f} m/s -> {wp:6.1f} MW ({wp/1000*100:5.1f}% capacity)")
