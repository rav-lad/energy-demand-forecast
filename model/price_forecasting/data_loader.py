"""Data loading and price simulation for price forecasting models.

This module provides functions to load energy demand data and generate realistic
price data based on market fundamentals. In production, this would be replaced
with actual ENTSO-E day-ahead price data.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


def simulate_realistic_prices(
    load_data: pd.DataFrame,
    base_price: float = 50.0,
    load_sensitivity: float = 1.5,
    volatility: float = 0.15,
    spike_probability: float = 0.02,
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    """Simulate realistic electricity prices based on load data.

    This function generates synthetic electricity prices that exhibit realistic
    market characteristics including:
    - Load-dependent pricing (merit order effect)
    - Daily and seasonal patterns
    - Price spikes
    - Mean reversion
    - Volatility clustering

    Args:
        load_data: DataFrame with 'datetime_hour' and 'conso_elec_mw' columns.
        base_price: Base electricity price in EUR/MWh.
        load_sensitivity: Sensitivity of price to load changes.
        volatility: Price volatility parameter.
        spike_probability: Probability of price spikes occurring.
        random_seed: Random seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame with price data including:
            - datetime_hour: Timestamp
            - price: Simulated electricity price (EUR/MWh)
            - load_mw: Total load in MW
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    df = load_data.copy()

    # Aggregate load across regions if multiple regions exist
    if "insee_region" in df.columns:
        df = (
            df.groupby("datetime_hour")
            .agg({"conso_elec_mw": "sum"})
            .reset_index()
        )

    df = df.sort_values("datetime_hour").reset_index(drop=True)

    # Normalize load to [0, 1] range
    load_normalized = (df["conso_elec_mw"] - df["conso_elec_mw"].min()) / (
        df["conso_elec_mw"].max() - df["conso_elec_mw"].min()
    )

    # Base price component from load (merit order curve - convex relationship)
    load_price_component = base_price * (1 + load_sensitivity * load_normalized**2)

    # Add hour-of-day pattern (peak hours more expensive)
    hour = pd.to_datetime(df["datetime_hour"]).dt.hour
    hour_factor = 1 + 0.3 * np.sin(2 * np.pi * (hour - 6) / 24)
    hour_factor = np.maximum(hour_factor, 0.7)

    # Add day-of-week pattern (weekends cheaper)
    dow = pd.to_datetime(df["datetime_hour"]).dt.dayofweek
    dow_factor = np.where(dow >= 5, 0.85, 1.0)

    # Add seasonal pattern (winter more expensive)
    day_of_year = pd.to_datetime(df["datetime_hour"]).dt.dayofyear
    seasonal_factor = 1 + 0.2 * np.cos(2 * np.pi * (day_of_year - 15) / 365)

    # Combine all components
    base_component = load_price_component * hour_factor * dow_factor * seasonal_factor

    # Add autocorrelated noise (GARCH-like volatility clustering)
    n = len(df)
    noise = np.zeros(n)
    vol = np.ones(n) * volatility

    for t in range(1, n):
        # GARCH(1,1) style volatility
        vol[t] = np.sqrt(
            0.05 * volatility**2
            + 0.1 * noise[t - 1] ** 2
            + 0.85 * vol[t - 1] ** 2
        )
        noise[t] = vol[t] * np.random.randn()

    # Add price spikes (extreme events)
    spike_mask = np.random.rand(n) < spike_probability
    spike_magnitude = np.random.exponential(100, n) * spike_mask

    # Combine all components
    price = base_component * (1 + noise) + spike_magnitude

    # Ensure non-negative prices (but allow low prices for renewable flush)
    price = np.maximum(price, -5.0)

    # Create output DataFrame
    result = pd.DataFrame(
        {
            "datetime_hour": df["datetime_hour"],
            "price": price,
            "load_mw": df["conso_elec_mw"],
        }
    )

    return result


def load_price_and_load_data(
    data_path: str = "data/modified_data/energy_hourly_regional.csv",
    simulate_prices: bool = True,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Load energy load data and corresponding prices.

    In production, this would load actual ENTSO-E day-ahead prices.
    For development, it simulates realistic prices based on load data.

    Args:
        data_path: Path to hourly load data CSV file.
        simulate_prices: If True, simulate prices based on load data.
        random_seed: Random seed for price simulation.

    Returns:
        pd.DataFrame: Combined load and price data with features.
    """
    # Load load data
    if not Path(data_path).exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            f"Please run data collection scripts first."
        )

    load_df = pd.read_csv(data_path)
    load_df["datetime_hour"] = pd.to_datetime(load_df["datetime_hour"])

    if simulate_prices:
        # Simulate realistic prices
        price_df = simulate_realistic_prices(load_df, random_seed=random_seed)
        return price_df
    else:
        # In production, load actual price data from ENTSO-E API
        # For now, raise an error
        raise NotImplementedError(
            "Loading actual ENTSO-E price data not yet implemented.\n"
            "Set simulate_prices=True to use simulated data."
        )


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-based features to price data.

    Args:
        df: DataFrame with 'datetime_hour' column.

    Returns:
        pd.DataFrame: DataFrame with added calendar features.
    """
    df = df.copy()
    dt = pd.to_datetime(df["datetime_hour"])

    # Time features
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["day_of_month"] = dt.dt.day
    df["day_of_year"] = dt.dt.dayofyear
    df["week_of_year"] = dt.dt.isocalendar().week
    df["month"] = dt.dt.month
    df["quarter"] = dt.dt.quarter
    df["year"] = dt.dt.year

    # Cyclical encoding of time features
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # Binary features
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_peak_hour"] = ((df["hour"] >= 8) & (df["hour"] <= 20)).astype(int)
    df["is_night"] = ((df["hour"] >= 22) | (df["hour"] <= 6)).astype(int)

    # Seasonal indicators
    df["is_winter"] = df["month"].isin([12, 1, 2]).astype(int)
    df["is_summer"] = df["month"].isin([6, 7, 8]).astype(int)

    return df


def add_lag_features(
    df: pd.DataFrame, target_col: str = "price", lags: list = None
) -> pd.DataFrame:
    """Add lagged features for time series forecasting.

    Args:
        df: DataFrame sorted by datetime.
        target_col: Column name to create lags for.
        lags: List of lag periods (in hours).

    Returns:
        pd.DataFrame: DataFrame with added lag features.
    """
    if lags is None:
        lags = [1, 2, 3, 24, 48, 168]  # 1h, 2h, 3h, 1d, 2d, 1w

    df = df.copy()

    for lag in lags:
        df[f"{target_col}_lag_{lag}h"] = df[target_col].shift(lag)

    # Rolling statistics
    for window in [24, 168]:  # 1 day, 1 week
        df[f"{target_col}_roll_mean_{window}h"] = (
            df[target_col].rolling(window=window, min_periods=1).mean()
        )
        df[f"{target_col}_roll_std_{window}h"] = (
            df[target_col].rolling(window=window, min_periods=1).std()
        )
        df[f"{target_col}_roll_min_{window}h"] = (
            df[target_col].rolling(window=window, min_periods=1).min()
        )
        df[f"{target_col}_roll_max_{window}h"] = (
            df[target_col].rolling(window=window, min_periods=1).max()
        )

    return df


def prepare_price_forecasting_dataset(
    data_path: str = "data/modified_data/energy_hourly_regional.csv",
    target_col: str = "price",
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, list]:
    """Prepare complete dataset for price forecasting.

    Args:
        data_path: Path to hourly load data.
        target_col: Target variable column name.
        random_seed: Random seed for reproducibility.

    Returns:
        tuple: (DataFrame with all features, list of feature column names)
    """
    # Load price and load data
    df = load_price_and_load_data(data_path, random_seed=random_seed)

    # Add calendar features
    df = add_calendar_features(df)

    # Add lag features for price
    df = add_lag_features(df, target_col="price")

    # Add lag features for load
    df = add_lag_features(df, target_col="load_mw")

    # Drop rows with NaN values (from lagging)
    df = df.dropna().reset_index(drop=True)

    # Identify feature columns (exclude datetime and target)
    feature_cols = [
        col
        for col in df.columns
        if col not in ["datetime_hour", target_col]
    ]

    return df, feature_cols
