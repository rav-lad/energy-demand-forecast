"""
Build the unified dataset for model training and inference.

Merges:
  - ENTSO-E day-ahead spot prices (FR)          -> data/raw/fr_spot_hourly.csv
  - ODRE electricity consumption (national)      -> data/raw/energy_consumption_2023-2026.csv
  - Open-Meteo weather (Paris)                   -> data/external/weather_hourly_*.csv

Output: data/dataset.csv  (one row per hour, all features ready for ML)

Run daily after download_data.py:
  python scripts/build_dataset.py
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RAW = PROJECT_ROOT / "data" / "raw"
EXTERNAL = PROJECT_ROOT / "data" / "external"
OUT = PROJECT_ROOT / "data" / "dataset.csv"

# France renewable installed capacity (RTE Annual Electricity Report 2024)
WIND_CAPACITY_MW = 22_000
SOLAR_CAPACITY_MW = 17_000


# ---------------------------------------------------------------------------
# 1. Spot prices
# ---------------------------------------------------------------------------
def load_prices() -> pd.DataFrame:
    df = pd.read_csv(RAW / "fr_spot_hourly.csv")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df[["datetime", "price"]].sort_values("datetime").drop_duplicates("datetime")
    return df


# ---------------------------------------------------------------------------
# 2. Consumption (ODRE half-hourly -> hourly national)
# ---------------------------------------------------------------------------
def load_consumption() -> pd.DataFrame:
    df = pd.read_csv(
        RAW / "energy_consumption_2023-2026.csv",
        sep=";",
        usecols=["date_heure", "consommation_brute_electricite_rte"],
    )
    df["datetime"] = pd.to_datetime(df["date_heure"], utc=True, errors="coerce")
    df["datetime"] = df["datetime"].dt.tz_localize(None)
    df = df.dropna(subset=["datetime"])

    # Aggregate regions and resample to hourly
    national = (
        df.groupby("datetime")["consommation_brute_electricite_rte"]
        .sum()
        .reset_index()
        .rename(columns={"consommation_brute_electricite_rte": "load_mw"})
    )
    national = national.set_index("datetime").resample("h").mean().reset_index()
    national = national.dropna(subset=["load_mw"])
    return national


# ---------------------------------------------------------------------------
# 3. Weather
# ---------------------------------------------------------------------------
def load_weather() -> pd.DataFrame:
    # Pick the most recent weather file
    files = sorted(EXTERNAL.glob("weather_hourly_*.csv"))
    if not files:
        raise FileNotFoundError("No weather file found in data/external/")
    df = pd.read_csv(files[-1])
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").drop_duplicates("datetime")
    return df


# ---------------------------------------------------------------------------
# 4. Derived features
# ---------------------------------------------------------------------------
def add_renewable_features(df: pd.DataFrame) -> pd.DataFrame:
    """Wind and solar production estimates from weather."""
    wind = df["wind_speed_10m"].values
    # Simple wind power curve: cubic between cut-in (3) and rated (12), flat to cut-out (25)
    p = np.zeros_like(wind)
    mask_mid = (wind >= 3) & (wind < 12)
    mask_rated = (wind >= 12) & (wind <= 25)
    p[mask_mid] = ((wind[mask_mid] - 3) / (12 - 3)) ** 3
    p[mask_rated] = 1.0
    df["wind_production_mw"] = p * WIND_CAPACITY_MW

    rad = df["shortwave_radiation"].clip(lower=0).values
    df["solar_production_mw"] = (rad / 1000) * SOLAR_CAPACITY_MW * 0.17  # 17% efficiency

    df["renewable_production_mw"] = df["wind_production_mw"] + df["solar_production_mw"]
    df["net_load_mw"] = (df["load_mw"] - df["renewable_production_mw"]).clip(lower=0)
    df["renewable_penetration_pct"] = (
        df["renewable_production_mw"] / df["load_mw"].replace(0, np.nan)
    ).clip(0, 1)
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    dt = df["datetime"]
    df["hour"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["day_of_month"] = dt.dt.day
    df["day_of_year"] = dt.dt.dayofyear
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["month"] = dt.dt.month
    df["quarter"] = dt.dt.quarter
    df["year"] = dt.dt.year

    # Cyclical encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_peak_hour"] = df["hour"].between(8, 20).astype(int)
    df["is_night"] = (df["hour"] < 6).astype(int)
    df["is_winter"] = df["month"].isin([12, 1, 2]).astype(int)
    df["is_summer"] = df["month"].isin([6, 7, 8]).astype(int)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    # Time-based merge to handle DST gaps correctly (shift() by position is wrong on gappy data)
    indexed = df.set_index("datetime")[["price", "load_mw"]]
    for lag in [24, 48, 72, 168]:
        lag_index = df["datetime"] - pd.Timedelta(hours=lag)
        df[f"price_lag_{lag}h"] = indexed["price"].reindex(lag_index).values
        df[f"load_lag_{lag}h"]  = indexed["load_mw"].reindex(lag_index).values

    # Rolling stats: shift(24h) before rolling to avoid leakage
    df_tmp = df.set_index("datetime").sort_index()
    for window in [24, 168]:
        df[f"price_roll_mean_{window}h"] = (
            df_tmp["price"].shift(1, freq="h").rolling(f"{window}h").mean().reindex(df["datetime"]).values
        )
        df[f"price_roll_std_{window}h"] = (
            df_tmp["price"].shift(1, freq="h").rolling(f"{window}h").std().reindex(df["datetime"]).values
        )
        df[f"load_roll_mean_{window}h"] = (
            df_tmp["load_mw"].shift(1, freq="h").rolling(f"{window}h").mean().reindex(df["datetime"]).values
        )
        df[f"load_roll_std_{window}h"] = (
            df_tmp["load_mw"].shift(1, freq="h").rolling(f"{window}h").std().reindex(df["datetime"]).values
        )
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build() -> pd.DataFrame:
    print("Loading prices...")
    prices = load_prices()
    print(f"  {len(prices)} hours | {prices['datetime'].min().date()} -> {prices['datetime'].max().date()}")

    print("Loading consumption...")
    consumption = load_consumption()
    print(f"  {len(consumption)} hours | {consumption['datetime'].min().date()} -> {consumption['datetime'].max().date()}")

    print("Loading weather...")
    weather = load_weather()
    print(f"  {len(weather)} hours | {weather['datetime'].min().date()} -> {weather['datetime'].max().date()}")

    print("Merging...")
    # Left join on prices so recent hours without ODRE data are kept
    df = prices.merge(consumption, on="datetime", how="left")
    df = df.merge(weather, on="datetime", how="inner")
    print(f"  After merge: {len(df)} hours")

    # Forward-fill load_mw for hours where ODRE hasn't published yet (recent weeks)
    # and for DST clock-back gaps (1 missing hour per year). This keeps recent rows
    # usable for inference while using the most recent known load as a proxy.
    n_nan = df["load_mw"].isnull().sum()
    if n_nan > 0:
        df["load_mw"] = df["load_mw"].ffill()
        print(f"  Forward-filled {n_nan} missing load_mw values")

    print("Adding derived features...")
    df = add_renewable_features(df)
    df = add_calendar_features(df)
    df = add_lag_features(df)

    # Forward-fill rolling stat NaNs caused by DST spring-forward gaps (7 rows/year)
    roll_cols = [c for c in df.columns if c.startswith("price_roll_") or c.startswith("load_roll_")]
    df[roll_cols] = df[roll_cols].ffill()

    # Drop rows where price is NaN or price lag features are NaN (first 168h warmup)
    price_lag_cols = [c for c in df.columns if c.startswith("price_lag_")]
    df = df.dropna(subset=["price"] + price_lag_cols).reset_index(drop=True)

    print(f"  Final dataset: {len(df)} hours | {df['datetime'].min().date()} -> {df['datetime'].max().date()}")
    print(f"  Columns: {len(df.columns)}")

    df.to_csv(OUT, index=False)
    print(f"\nSaved -> {OUT}")
    return df


if __name__ == "__main__":
    build()
