"""Data loading and price simulation for price forecasting models.

This module provides functions to load energy demand data and generate realistic
price data based on market fundamentals. In production, this would be replaced
with actual ENTSO-E day-ahead price data.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


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
    price_file: str = "data/raw_data/market_prices/day_ahead_prices_FR.csv",
) -> pd.DataFrame:
    """Load energy load data and corresponding prices.

    In production, this would load actual ENTSO-E day-ahead prices.
    For development, it simulates realistic prices based on load data.

    Args:
        data_path: Path to hourly load data CSV file.
        simulate_prices: If True, simulate prices based on load data.
        random_seed: Random seed for price simulation.
        price_file: Path to ENTSO-E price data CSV (used when simulate_prices=False).

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
        logger.info("Using simulated prices based on load data")
        price_df = simulate_realistic_prices(load_df, random_seed=random_seed)
        return price_df
    else:
        # Load actual ENTSO-E price data
        logger.info(f"Loading real ENTSO-E price data from {price_file}")

        if not Path(price_file).exists():
            raise FileNotFoundError(
                f"Price data file not found: {price_file}\n"
                f"Please collect price data first:\n"
                f"  python data_recuperation/data_market_prices.py --start_date 2023-01-01 --end_date 2024-12-31 --countries FR\n"
                f"Or set simulate_prices=True to use simulated data."
            )

        # Load price data
        price_df = pd.read_csv(price_file)

        # Handle different possible datetime column names
        datetime_col = None
        for col in ["datetime", "datetime_hour", "timestamp"]:
            if col in price_df.columns:
                datetime_col = col
                break

        if datetime_col is None:
            # Try to find index if it's a datetime
            if price_df.index.name in ["datetime", "timestamp"]:
                price_df = price_df.reset_index()
                datetime_col = price_df.columns[0]
            else:
                raise ValueError(
                    f"Could not find datetime column in {price_file}. "
                    f"Available columns: {price_df.columns.tolist()}"
                )

        # Rename to standard format
        if datetime_col != "datetime_hour":
            price_df["datetime_hour"] = pd.to_datetime(price_df[datetime_col])
        else:
            price_df["datetime_hour"] = pd.to_datetime(price_df["datetime_hour"])

        # Find price column (handle different naming conventions)
        price_col = None
        for col in ["price_FR", "price_eur_mwh", "price", "day_ahead_price"]:
            if col in price_df.columns:
                price_col = col
                break

        if price_col is None:
            raise ValueError(
                f"Could not find price column in {price_file}. "
                f"Available columns: {price_df.columns.tolist()}"
            )

        # Rename to standard format
        if price_col != "price":
            price_df["price"] = price_df[price_col]

        # Aggregate load data across regions if necessary
        if "insee_region" in load_df.columns:
            load_df = (
                load_df.groupby("datetime_hour")
                .agg({"conso_elec_mw": "sum"})
                .reset_index()
            )

        # Merge price and load data on datetime
        merged_df = pd.merge(
            load_df,
            price_df[["datetime_hour", "price"]],
            on="datetime_hour",
            how="inner",
        )

        # Add load column (rename from conso_elec_mw)
        merged_df["load_mw"] = merged_df["conso_elec_mw"]

        logger.info(
            f"Loaded {len(merged_df)} records with real prices "
            f"(range: {merged_df['price'].min():.2f} - {merged_df['price'].max():.2f} EUR/MWh)"
        )

        # Select relevant columns
        result_df = merged_df[["datetime_hour", "price", "load_mw"]].copy()

        return result_df


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

    # Identify feature columns (exclude datetime, target, and contemporaneous load)
    # IMPORTANT: Exclude 'load_mw' to avoid data leakage!
    # We only use lagged load (load_mw_lag_*) which are known at prediction time
    feature_cols = [
        col
        for col in df.columns
        if col not in ["datetime_hour", target_col, "load_mw"]
    ]

    logger.info(f"Total features: {len(feature_cols)}")
    logger.info(f"Features include load lags but NOT contemporaneous load (avoiding leakage)")

    return df, feature_cols


def prepare_price_forecasting_with_fuel_prices(
    data_path: str = "data/modified_data/energy_hourly_regional.csv",
    target_col: str = "price",
    include_fuel_prices: bool = True,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, list]:
    """Prepare complete dataset for price forecasting with fuel price features.

    This function extends prepare_price_forecasting_dataset by adding fuel price
    features (TTF gas, EUA carbon, coal) and derived spreads (spark, dark, clean).

    Args:
        data_path: Path to hourly load data.
        target_col: Target variable column name.
        include_fuel_prices: Whether to include fuel price features.
        random_seed: Random seed for reproducibility.

    Returns:
        tuple: (DataFrame with all features including fuel prices, list of feature column names)
    """
    # First, get base price forecasting dataset
    df, _ = prepare_price_forecasting_dataset(data_path, target_col, random_seed)

    if include_fuel_prices:
        from data_collection.fuel_prices import generate_fuel_price_features

        # Generate fuel price features
        fuel_df = generate_fuel_price_features(
            dates=df["datetime_hour"],
            power_prices=df[target_col],
            random_seed=random_seed,
        )

        # Validate before merge
        if len(df) != len(fuel_df):
            logger.warning(
                f"Length mismatch before merge: df={len(df)}, fuel_df={len(fuel_df)}. "
                "Resetting indices to ensure proper alignment."
            )
            df = df.reset_index(drop=True)
            fuel_df = fuel_df.reset_index(drop=True)
            # Take minimum length to avoid misalignment
            min_len = min(len(df), len(fuel_df))
            df = df.iloc[:min_len]
            fuel_df = fuel_df.iloc[:min_len]

        # Check for duplicate columns
        duplicate_cols = set(df.columns) & set(fuel_df.columns)
        if duplicate_cols:
            logger.warning(f"Duplicate columns found: {duplicate_cols}. Dropping from fuel_df.")
            fuel_df = fuel_df.drop(columns=list(duplicate_cols))

        # Merge with main dataset
        df = pd.concat([df, fuel_df], axis=1)

        # Add lag features for fuel prices (they affect future electricity prices)
        for fuel_col in ["ttf_gas_price", "eua_carbon_price", "coal_price"]:
            if fuel_col in df.columns:
                for lag in [1, 24, 168]:  # 1h, 1day, 1week
                    df[f"{fuel_col}_lag_{lag}h"] = df[fuel_col].shift(lag)

        # Add lag features for spreads (important predictors)
        for spread_col in ["spark_spread", "dark_spread"]:
            if spread_col in df.columns:
                for lag in [1, 24]:
                    df[f"{spread_col}_lag_{lag}h"] = df[spread_col].shift(lag)

        # Drop rows with NaN from new lags
        df = df.dropna().reset_index(drop=True)

    # Identify all feature columns
    # IMPORTANT: Exclude contemporaneous variables to avoid data leakage
    # We only use lagged versions which are known at prediction time
    exclude_cols = [
        "datetime_hour",
        target_col,
        "load_mw",  # Use load_mw_lag_* instead
        # Fuel prices: we use lags, so exclude contemporaneous values
        "ttf_gas_price",
        "eua_carbon_price",
        "coal_price",
        "spark_spread",
        "dark_spread",
        "clean_spark_spread",
    ]

    feature_cols = [
        col for col in df.columns if col not in exclude_cols
    ]

    logger.info(f"Total features with fuel prices: {len(feature_cols)}")
    logger.info(f"Excluded contemporaneous variables to avoid leakage")

    return df, feature_cols
