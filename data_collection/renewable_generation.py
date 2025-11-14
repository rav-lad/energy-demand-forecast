"""Renewable energy generation forecasting (wind and solar).

This module provides functions to simulate and forecast renewable electricity
generation, which is critical for modern power markets with high renewable
penetration (Germany >50%, France >25%).

In production, this would integrate with:
- GraphCast/ECMWF for wind speed forecasts
- GFS/HRRR for solar irradiance forecasts
- ENTSO-E for installed capacity data
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd


def wind_power_curve(
    wind_speed: np.ndarray,
    cut_in_speed: float = 3.0,
    rated_speed: float = 12.0,
    cut_out_speed: float = 25.0,
    rated_power: float = 3.0,
) -> np.ndarray:
    """Calculate wind turbine power output from wind speed.

    Uses a simplified power curve model typical of modern 3MW onshore turbines.
    Real power curves are manufacturer-specific (Vestas, Siemens Gamesa, etc.).

    Power curve regions:
    1. Below cut-in: 0 MW (not enough wind)
    2. Cut-in to rated: Cubic relationship (P ∝ v³)
    3. Rated to cut-out: Constant rated power
    4. Above cut-out: 0 MW (safety shutdown)

    Args:
        wind_speed: Wind speed at hub height (m/s), typically 80-120m.
        cut_in_speed: Minimum wind speed for generation (m/s).
        rated_speed: Wind speed at rated power (m/s).
        cut_out_speed: Maximum safe wind speed (m/s).
        rated_power: Turbine rated capacity (MW).

    Returns:
        np.ndarray: Power output (MW per turbine).
    """
    power = np.zeros_like(wind_speed)

    # Region 2: Partial load (cubic relationship)
    partial_load_mask = (wind_speed >= cut_in_speed) & (wind_speed < rated_speed)
    if np.any(partial_load_mask):
        # Simplified cubic power curve
        v_partial = wind_speed[partial_load_mask]
        power[partial_load_mask] = rated_power * (
            (v_partial - cut_in_speed) / (rated_speed - cut_in_speed)
        ) ** 3

    # Region 3: Rated power
    rated_mask = (wind_speed >= rated_speed) & (wind_speed < cut_out_speed)
    power[rated_mask] = rated_power

    # Regions 1 and 4 already zero

    return power


def simulate_wind_generation(
    dates: pd.DatetimeIndex,
    installed_capacity_mw: float = 5000.0,
    mean_wind_speed: float = 7.0,
    wind_variability: float = 3.0,
    seasonal_amplitude: float = 1.5,
    random_seed: Optional[int] = None,
) -> pd.Series:
    """Simulate wind power generation for a region.

    Models realistic wind patterns:
    - Seasonal variation (higher in winter)
    - Diurnal variation (weak for wind)
    - Stochastic fluctuations (weather systems)
    - Power curve non-linearity

    Args:
        dates: Datetime index for simulation.
        installed_capacity_mw: Total installed wind capacity (MW).
        mean_wind_speed: Long-term average wind speed (m/s).
        wind_variability: Standard deviation of wind speed (m/s).
        seasonal_amplitude: Magnitude of seasonal variation (m/s).
        random_seed: Random seed for reproducibility.

    Returns:
        pd.Series: Wind power generation (MW) indexed by datetime.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Handle both DatetimeIndex and Series
    if isinstance(dates, pd.Series):
        dates = pd.DatetimeIndex(dates)

    n = len(dates)

    # Seasonal component (higher wind in winter)
    day_of_year = dates.dayofyear
    seasonal = seasonal_amplitude * np.cos(2 * np.pi * (day_of_year - 15) / 365)

    # Weak diurnal component (wind slightly higher during day due to convection)
    hour = dates.hour
    diurnal = 0.5 * np.sin(2 * np.pi * (hour - 6) / 24)

    # Autocorrelated wind speed (Ornstein-Uhlenbeck process)
    # Wind systems have ~6-12 hour persistence
    wind_speed = np.zeros(n)
    wind_speed[0] = mean_wind_speed

    dt = 1 / 24  # Hourly timesteps
    theta = 0.3  # Mean reversion speed (1/theta ≈ 3 hours half-life)

    for t in range(1, n):
        drift = theta * (mean_wind_speed + seasonal[t] + diurnal[t] - wind_speed[t - 1]) * dt
        diffusion = wind_variability * np.sqrt(dt) * np.random.randn()
        wind_speed[t] = wind_speed[t - 1] + drift + diffusion

    # Ensure non-negative and reasonable wind speeds
    wind_speed = np.clip(wind_speed, 0, 40)

    # Convert wind speed to power using power curve
    # Assume average turbine size 3 MW
    num_turbines = installed_capacity_mw / 3.0
    power_per_turbine = wind_power_curve(wind_speed)
    total_generation = power_per_turbine * num_turbines

    # Add availability factor (turbines occasionally offline for maintenance)
    availability = np.random.uniform(0.95, 1.0, n)
    total_generation = total_generation * availability

    return pd.Series(total_generation, index=dates, name="wind_generation_mw")


def solar_irradiance_model(
    dates: pd.DatetimeIndex,
    latitude: float = 48.8,
    clear_sky_irradiance: float = 1000.0,
    cloud_cover_mean: float = 0.5,
    cloud_cover_std: float = 0.2,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """Calculate solar irradiance with cloud cover effects.

    Models:
    - Solar geometry (elevation angle, day length)
    - Seasonal variation (summer > winter)
    - Cloud cover (stochastic)
    - Day/night cycle

    Args:
        dates: Datetime index.
        latitude: Location latitude (degrees North).
        clear_sky_irradiance: Maximum irradiance (W/m²).
        cloud_cover_mean: Average cloud cover fraction (0-1).
        cloud_cover_std: Cloud cover variability.
        random_seed: Random seed for reproducibility.

    Returns:
        np.ndarray: Solar irradiance (W/m²).
    """
    if random_seed is not None:
        np.random.seed(random_seed + 100)

    # Handle both DatetimeIndex and Series
    if isinstance(dates, pd.Series):
        dates = pd.DatetimeIndex(dates)

    n = len(dates)
    hour = dates.hour
    day_of_year = dates.dayofyear

    # Solar declination angle (degrees)
    # Maximum at summer solstice (~23.5°), minimum at winter solstice (~-23.5°)
    declination = 23.45 * np.sin(2 * np.pi * (day_of_year - 81) / 365)

    # Hour angle (0 at solar noon)
    hour_angle = 15 * (hour - 12)  # 15° per hour

    # Solar elevation angle (simplified)
    # elevation = arcsin(sin(lat) * sin(decl) + cos(lat) * cos(decl) * cos(hour_angle))
    lat_rad = np.radians(latitude)
    decl_rad = np.radians(declination)
    hour_angle_rad = np.radians(hour_angle)

    sin_elevation = (
        np.sin(lat_rad) * np.sin(decl_rad)
        + np.cos(lat_rad) * np.cos(decl_rad) * np.cos(hour_angle_rad)
    )
    sin_elevation = np.clip(sin_elevation, 0, 1)  # Below horizon = 0

    # Clear-sky irradiance (proportional to sin(elevation))
    irradiance_clear = clear_sky_irradiance * sin_elevation

    # Cloud cover (autocorrelated)
    cloud_cover = np.zeros(n)
    cloud_cover[0] = cloud_cover_mean

    for t in range(1, n):
        # Mean reversion with persistence
        drift = 0.1 * (cloud_cover_mean - cloud_cover[t - 1]) / 24
        diffusion = cloud_cover_std * np.sqrt(1 / 24) * np.random.randn()
        cloud_cover[t] = cloud_cover[t - 1] + drift + diffusion

    cloud_cover = np.clip(cloud_cover, 0, 1)

    # Irradiance with cloud attenuation
    # Cloud cover reduces irradiance (not linear, approximately quadratic)
    irradiance = irradiance_clear * (1 - cloud_cover**1.5)

    return irradiance


def simulate_solar_generation(
    dates: pd.DatetimeIndex,
    installed_capacity_mw: float = 3000.0,
    latitude: float = 48.8,
    panel_efficiency: float = 0.18,
    performance_ratio: float = 0.75,
    random_seed: Optional[int] = None,
) -> pd.Series:
    """Simulate solar PV power generation for a region.

    Args:
        dates: Datetime index for simulation.
        installed_capacity_mw: Total installed solar capacity (MW).
        latitude: Location latitude for solar geometry (Paris ≈ 48.8°N).
        panel_efficiency: PV panel efficiency (typically 15-20%).
        performance_ratio: System losses factor (inverter, wiring, soiling).
        random_seed: Random seed for reproducibility.

    Returns:
        pd.Series: Solar power generation (MW) indexed by datetime.
    """
    # Calculate irradiance
    irradiance = solar_irradiance_model(
        dates, latitude=latitude, random_seed=random_seed
    )

    # Convert irradiance to power
    # P = Irradiance × Area × Efficiency × Performance_Ratio
    # Installed capacity already accounts for area × efficiency, so:
    # P = (Irradiance / 1000) × Installed_Capacity × Performance_Ratio

    generation = (irradiance / 1000.0) * installed_capacity_mw * performance_ratio

    return pd.Series(generation, index=dates, name="solar_generation_mw")


def calculate_renewable_share(
    renewable_generation: pd.Series, total_load: pd.Series
) -> pd.Series:
    """Calculate instantaneous renewable energy share.

    Renewable share = Renewable Generation / Total Load

    High renewable share (>80%) can lead to:
    - Negative prices (over-supply)
    - Curtailment (wasted renewable energy)
    - Grid stability issues

    Args:
        renewable_generation: Total renewable generation (MW).
        total_load: Total electricity demand (MW).

    Returns:
        pd.Series: Renewable share (0-1, can exceed 1 if exports).
    """
    renewable_share = renewable_generation / (total_load + 1e-6)  # Avoid division by zero
    return renewable_share.rename("renewable_share")


def calculate_curtailment_risk(
    renewable_generation: pd.Series,
    total_load: pd.Series,
    must_run_generation: float = 0.2,
) -> pd.Series:
    """Calculate risk of renewable curtailment.

    Curtailment occurs when renewable generation exceeds demand minus must-run
    baseload (nuclear, combined heat-power).

    Curtailment Risk = max(0, Renewable - (Load - Must_Run)) / Renewable

    Args:
        renewable_generation: Total renewable generation (MW).
        total_load: Total electricity demand (MW).
        must_run_generation: Fraction of load that is inflexible baseload.

    Returns:
        pd.Series: Curtailment risk (0-1).
    """
    must_run_mw = total_load * must_run_generation
    available_capacity = total_load - must_run_mw

    curtailed = np.maximum(0, renewable_generation - available_capacity)
    curtailment_risk = curtailed / (renewable_generation + 1e-6)

    return curtailment_risk.rename("curtailment_risk")


def generate_renewable_features(
    dates: pd.DatetimeIndex,
    total_load: Optional[pd.Series] = None,
    wind_capacity_mw: float = 5000.0,
    solar_capacity_mw: float = 3000.0,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Generate comprehensive renewable energy feature set.

    This function simulates wind and solar generation and calculates derived
    features critical for price forecasting in high-renewable markets.

    Args:
        dates: Datetime index for simulation.
        total_load: Optional total electricity load (MW).
        wind_capacity_mw: Installed wind capacity.
        solar_capacity_mw: Installed solar capacity.
        random_seed: Random seed for reproducibility.

    Returns:
        pd.DataFrame: Renewable features including:
            - wind_generation_mw: Wind power output
            - solar_generation_mw: Solar power output
            - total_renewable_mw: Combined renewable generation
            - renewable_share: Renewable % of load (if load provided)
            - curtailment_risk: Risk of wasted renewable energy
            - wind_capacity_factor: Wind generation / installed capacity
            - solar_capacity_factor: Solar generation / installed capacity
    """
    # Simulate generation
    wind = simulate_wind_generation(
        dates, installed_capacity_mw=wind_capacity_mw, random_seed=random_seed
    )

    solar = simulate_solar_generation(
        dates, installed_capacity_mw=solar_capacity_mw, random_seed=random_seed
    )

    # Create DataFrame
    df = pd.DataFrame({"wind_generation_mw": wind, "solar_generation_mw": solar})

    # Total renewable generation
    df["total_renewable_mw"] = df["wind_generation_mw"] + df["solar_generation_mw"]

    # Capacity factors
    df["wind_capacity_factor"] = df["wind_generation_mw"] / wind_capacity_mw
    df["solar_capacity_factor"] = df["solar_generation_mw"] / solar_capacity_mw
    df["renewable_capacity_factor"] = df["total_renewable_mw"] / (
        wind_capacity_mw + solar_capacity_mw
    )

    # If load provided, calculate share and curtailment risk
    if total_load is not None:
        total_load_aligned = total_load.reindex(dates)
        df["renewable_share"] = calculate_renewable_share(
            df["total_renewable_mw"], total_load_aligned
        )
        df["curtailment_risk"] = calculate_curtailment_risk(
            df["total_renewable_mw"], total_load_aligned
        )

        # Net load (load minus renewables) - what conventional plants must serve
        df["net_load_mw"] = total_load_aligned - df["total_renewable_mw"]
        df["net_load_mw"] = np.maximum(df["net_load_mw"], 0)  # Can't be negative

    return df


def load_or_simulate_renewable_generation(
    dates: pd.DatetimeIndex,
    total_load: Optional[pd.Series] = None,
    use_real_data: bool = False,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Load real renewable data or simulate if not available.

    Args:
        dates: Datetime index.
        total_load: Optional electricity load for derived features.
        use_real_data: If True, attempt to load real data (not yet implemented).
        random_seed: Random seed for simulation.

    Returns:
        pd.DataFrame: Renewable generation features.
    """
    if use_real_data:
        # TODO: Implement real data loading from ENTSO-E API
        # - ENTSO-E Transparency Platform: Actual generation per production type
        # - Separate wind onshore, wind offshore, solar
        raise NotImplementedError(
            "Real renewable data loading not yet implemented.\n"
            "Use use_real_data=False for simulated data."
        )
    else:
        return generate_renewable_features(dates, total_load, random_seed=random_seed)
