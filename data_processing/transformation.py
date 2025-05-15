import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from pathlib import Path
import holidays 
import joblib


BASE_DIR = Path(__file__).resolve().parents[1]
TRANSFORMED_DIR = BASE_DIR / "data" / "transformed_data"
TRANSFORMED_DIR.mkdir(parents=True, exist_ok=True)
SCALER_DIR = BASE_DIR / "models" / "scalers"
SCALER_DIR.mkdir(parents=True, exist_ok=True)

def add_holiday_column(df):
    fr_holidays = holidays.France(years=range(df['date'].dt.year.min(), df['date'].dt.year.max() + 1))
    df["ferie"] = df["date"].dt.date.isin(fr_holidays)
    return df


def transform_regression_and_xgb(
    df: pd.DataFrame,
    frequency: str = "daily",
    fit_scaler: bool = True,
    save: bool = True,
    scaler_path: Path | None = None,
    verbose=False
):
    """
    Transformation for linear regression and XGBoost:
    - Temporal feature extraction
    - Derived variable creation
    - Encoding
    - Normalization (with scaler management)
    """
    df = df.copy()

    # Date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = add_holiday_column(df)
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day
        df["weekday"] = df["date"].dt.weekday
        df["dayofyear"] = df["date"].dt.dayofyear
        df["sin_day"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
        df["cos_day"] = np.cos(2 * np.pi * df["dayofyear"] / 365)
        df.drop(columns=["date"], inplace=True)

    # Temperatures
    df["temp_mean"] = (df["temperature_2m_max"] + df["temperature_2m_min"]) / 2
    df["apparent_temp_mean"] = (df["apparent_temperature_max"] + df["apparent_temperature_min"]) / 2
    df["temp_range"] = df["temperature_2m_max"] - df["temperature_2m_min"]

    # Wind
    df["wind_range"] = df["wind_gusts_10m_max"] - df["wind_speed_10m_max"]
    df["wind_sector"] = pd.cut(
        df["wind_direction_10m_dominant"],
        bins=[0, 90, 180, 270, 360],
        labels=["N-E", "E-S", "S-W", "W-N"],
        include_lowest=True
    )

    # Convert sunrise/sunset to seconds
    for col in ["sunrise", "sunset"]:
        if col in df.columns:
            dt_col = pd.to_datetime(df[col], errors='coerce')
            df[col] = dt_col.dt.hour * 3600 + dt_col.dt.minute * 60

    # Radiation and ratios
    df["radiation_et0_ratio"] = df["shortwave_radiation_sum"] / (df["et0_fao_evapotranspiration"] + 1e-5)
    df["sunshine_ratio"] = df["sunshine_duration"] / (df["daylight_duration"] + 1e-5)

    # One-hot encoding
    df = pd.get_dummies(df, columns=["weather_code", "wind_sector", "insee_region"], drop_first=True)

    # Normalization
    target_cols = ["conso_elec_mw", "conso_gaz_mw"]
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns.difference(target_cols)

    if scaler_path is None:
        scaler_path = SCALER_DIR / f"scaler_{frequency}_reglin_xgboost.pkl"

    if fit_scaler:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        joblib.dump(scaler, scaler_path)
    else:
        scaler = joblib.load(scaler_path)
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    # Save transformed data
    if save:
        TRANSFORMED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(TRANSFORMED_DIR / f"train_{frequency}_reglin_xgboost.csv", index=False)
    if verbose:
        print(f"Transformation complete. Shape = {df.shape}, Features = {df.columns.tolist()[:5]}...")
    return df


def transform_dl(df: pd.DataFrame, seq_len=24, filter_too_short=False):
    """
    Transformation for deep learning (TFT, CNN/LSTM).
    - Adds time_idx if needed
    - Ensures correct types for each column
    - Optionally filters out too-short sequences
    """
    df = df.copy()

    # Ensure date is datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    # Ensure time_idx exists
    if "time_idx" not in df.columns:
        df = df.sort_values(["insee_region", "date"])
        df["time_idx"] = df.groupby("insee_region").cumcount()

    # Convert categorical columns
    for col in ["insee_region"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    df["time_idx"] = df["time_idx"].astype(int)

    # Convert sunrise/sunset to seconds
    for col in ["sunrise", "sunset"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df[col] = df[col].dt.hour * 3600 + df[col].dt.minute * 60 + df[col].dt.second

    if filter_too_short:
        min_length = 48  # max_encoder_length + max_prediction_length
        group_lengths = df.groupby("insee_region")["time_idx"].count()
        valid_regions = group_lengths[group_lengths >= min_length].index
        df = df[df["insee_region"].isin(valid_regions)].reset_index(drop=True)

    return df


def transform_time_series(df: pd.DataFrame):
    """
    Transformation for time series models.
    - Ensures data is sorted and indexed by time
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(by=["insee_region", "date"])
    df.set_index("date", inplace=True)
    
    return df


def transform_lightgbm_quantile(df: pd.DataFrame, frequency="daily", save=True, lags=True):
    """
    Transformation for LightGBM quantile models:
    - Temporal, meteorological, and advanced interaction features
    - No normalization
    - Optional: skip lag/rolling columns if lags=False
    """
    df = df.copy()

    # Date and temporal features
    df["date"] = pd.to_datetime(df["date"])
    df = add_holiday_column(df)
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["weekday"] = df["date"].dt.weekday
    df["dayofyear"] = df["date"].dt.dayofyear
    df["week_number"] = df["date"].dt.isocalendar().week.astype(int)
    df["sin_day"] = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["cos_day"] = np.cos(2 * np.pi * df["dayofyear"] / 365)
    df["sin_week"] = np.sin(2 * np.pi * df["week_number"] / 52)
    df["cos_week"] = np.cos(2 * np.pi * df["week_number"] / 52)

    # Lags and rolling windows (by region)
    if lags:
        df = df.sort_values(by=["insee_region", "date"])
        for lag in [1, 7]:
            df[f"conso_elec_mw_lag{lag}"] = df.groupby("insee_region")["conso_elec_mw"].shift(lag)
            df[f"conso_gaz_mw_lag{lag}"] = df.groupby("insee_region")["conso_gaz_mw"].shift(lag)
            df[f"temp_mean_lag{lag}"] = df.groupby("insee_region")["temperature_2m_max"].shift(lag)
            df[f"rain_sum_lag{lag}"] = df.groupby("insee_region")["rain_sum"].shift(lag)

        df["rolling_conso_elec_3"] = df.groupby("insee_region")["conso_elec_mw"].transform(lambda x: x.shift(1).rolling(3).mean())
        df["rolling_conso_elec_7"] = df.groupby("insee_region")["conso_elec_mw"].transform(lambda x: x.shift(1).rolling(7).mean())
        df["rolling_temp_max_3"] = df.groupby("insee_region")["temperature_2m_max"].transform(lambda x: x.shift(1).rolling(3).max())

    # Interactions
    df["temp_ferie_interaction"] = df["temperature_2m_max"] * df["ferie"].astype(int)
    df["is_off"] = (df["ferie"] | df["weekday"].isin([5, 6])).astype(int)

    # Derived meteorological variables
    df["temp_mean"] = (df["temperature_2m_max"] + df["temperature_2m_min"]) / 2
    df["apparent_temp_mean"] = (df["apparent_temperature_max"] + df["apparent_temperature_min"]) / 2
    df["temp_range"] = df["temperature_2m_max"] - df["temperature_2m_min"]
    df["wind_range"] = df["wind_gusts_10m_max"] - df["wind_speed_10m_max"]
    df["temp_radiation_interaction"] = df["temp_mean"] * df["shortwave_radiation_sum"]

    # Wind direction sector
    df["wind_sector"] = pd.cut(
        df["wind_direction_10m_dominant"],
        bins=[0, 90, 180, 270, 360],
        labels=["N-E", "E-S", "S-W", "W-N"],
        include_lowest=True
    )

    # Convert sunrise/sunset to seconds
    for col in ["sunrise", "sunset"]:
        dt_col = pd.to_datetime(df[col], errors="coerce")
        df[col] = dt_col.dt.hour * 3600 + dt_col.dt.minute * 60

    df["sunrise_sunset_diff"] = df["sunset"] - df["sunrise"]
    df["radiation_et0_ratio"] = df["shortwave_radiation_sum"] / (df["et0_fao_evapotranspiration"] + 1e-5)
    df["sunshine_ratio"] = df["sunshine_duration"] / (df["daylight_duration"] + 1e-5)
    df["sun_week_interaction"] = df["sunshine_ratio"] * df["weekday"]

    if lags and "temp_mean_lag1" in df.columns:
        df["delta_temp"] = df["temp_mean"] - df["temp_mean_lag1"]

    # Fill and clean lag columns
    if lags:
        lag_cols = [c for c in df.columns if "lag" in c or "rolling" in c]
        df[lag_cols] = df.groupby("insee_region")[lag_cols].apply(lambda g: g.bfill().ffill())
        df.dropna(subset=lag_cols, inplace=True)

    # Categorical encoding
    df[["weather_code", "wind_sector", "insee_region"]] = df[["weather_code", "wind_sector", "insee_region"]].astype(str)
    df = pd.get_dummies(df, columns=["weather_code", "wind_sector", "insee_region"], drop_first=True)

    # Move date column to the end
    if "date" in df.columns:
        date_col = df["date"]
        df.drop(columns=["date"], inplace=True)
        df = df.sort_index()
        df["date"] = date_col.values
    else:
        df = df.sort_index()

    # Save CSV
    if save:
        BASE_DIR = Path(__file__).resolve().parents[1]
        TRANSFORMED_DIR = BASE_DIR / "data" / "transformed_data"
        TRANSFORMED_DIR.mkdir(parents=True, exist_ok=True)
        suffix = "withlags" if lags else "withoutlags"
        df.to_csv(TRANSFORMED_DIR / f"train_{frequency}_lightgbm_quantile_{suffix}.csv", index=False)

    return df
