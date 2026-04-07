"""Load (electricity demand) forecast for price forecasting.

This module provides functions to:
1. Fetch official load forecasts from ENTSO-E/RTE (if available)
2. Generate load forecasts using a simple model (weather + calendar)
3. Simulate realistic forecast errors for backtesting
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def generate_load_forecast_simple(
    df: pd.DataFrame, historical_load: pd.Series, forecast_horizon_hours: int = 24
) -> pd.DataFrame:
    """Generate simple load forecast based on weather + calendar patterns.

    This is a FALLBACK when official RTE/ENTSO-E forecasts are not available.

    Method:
    - Uses temperature, day type, hour patterns
    - Calibrated on historical load data
    - Adds realistic forecast errors

    Args:
        df: DataFrame with datetime_hour and weather forecast
        historical_load: Series with actual historical load (for training)
        forecast_horizon_hours: Forecast horizon

    Returns:
        DataFrame with load_forecast column
    """
    df = df.copy()

    # Extract calendar features
    dt = pd.to_datetime(df["datetime_hour"])
    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_holiday"] = 0  # TODO: Add French holiday calendar

    # Use temperature forecast if available
    if "temp_forecast" in df.columns:
        df["temp"] = df["temp_forecast"]
    elif "temp_actual" in df.columns:
        df["temp"] = df["temp_actual"]
    else:
        logger.warning("No temperature data available, using dummy values")
        df["temp"] = 15.0

    # Simple heuristic model (can be replaced with trained model)
    # Base load pattern by hour
    hour_factors = {
        0: 0.75,
        1: 0.70,
        2: 0.68,
        3: 0.67,
        4: 0.68,
        5: 0.72,
        6: 0.80,
        7: 0.90,
        8: 0.95,
        9: 1.00,
        10: 1.02,
        11: 1.03,
        12: 1.00,
        13: 0.98,
        14: 0.97,
        15: 0.96,
        16: 0.98,
        17: 1.05,
        18: 1.10,
        19: 1.12,
        20: 1.10,
        21: 1.05,
        22: 0.95,
        23: 0.85,
    }

    df["hour_factor"] = df["hour"].map(hour_factors)

    # Weekend reduction
    df["dow_factor"] = np.where(df["is_weekend"], 0.85, 1.0)

    # Temperature effect (heating/cooling)
    # Heating degree days (HDD) below 15°C
    df["hdd"] = np.maximum(0, 15 - df["temp"])
    # Cooling degree days (CDD) above 24°C
    df["cdd"] = np.maximum(0, df["temp"] - 24)

    # Estimate base load from historical data
    if historical_load is not None and len(historical_load) > 0:
        base_load = historical_load.mean()
        temp_sensitivity = 300  # MW per degree (rough estimate for France)
    else:
        base_load = 50000  # MW (typical for France)
        temp_sensitivity = 300

    # Forecast formula
    df["load_forecast"] = base_load * df["hour_factor"] * df[
        "dow_factor"
    ] + temp_sensitivity * (
        df["hdd"] * 1.2 + df["cdd"] * 0.8
    )  # Heating stronger than cooling

    # Add forecast error (increases with horizon)
    forecast_error_std = 1000 + (forecast_horizon_hours / 24) * 500  # MW
    forecast_error = np.random.normal(0, forecast_error_std, len(df))

    # Add autocorrelation to errors
    forecast_error = pd.Series(forecast_error).ewm(alpha=0.4).mean().values

    df["load_forecast"] = df["load_forecast"] + forecast_error

    # Ensure non-negative
    df["load_forecast"] = np.maximum(df["load_forecast"], 10000)

    logger.info(
        f"[OK] Generated simple load forecast\n"
        f"   Mean forecast: {df['load_forecast'].mean():.0f} MW\n"
        f"   Range: [{df['load_forecast'].min():.0f}, {df['load_forecast'].max():.0f}] MW"
    )

    # Clean up intermediate columns
    cols_to_keep = ["datetime_hour", "load_forecast"]
    return df[cols_to_keep]


def generate_load_forecast_ml(
    df_forecast: pd.DataFrame,
    df_historical: pd.DataFrame,
    forecast_horizon_hours: int = 24,
) -> pd.DataFrame:
    """Generate load forecast using ML model trained on historical data.

    This is more accurate than simple heuristic, but requires training data.

    Args:
        df_forecast: DataFrame with datetime_hour and weather forecast features
        df_historical: DataFrame with historical load, weather, and calendar
        forecast_horizon_hours: Forecast horizon

    Returns:
        DataFrame with load_forecast column
    """
    logger.info("Training ML model for load forecasting...")

    # Prepare training data
    df_train = df_historical.copy()

    # Extract features
    dt_train = pd.to_datetime(df_train["datetime_hour"])
    X_train_features = pd.DataFrame(
        {
            "hour": dt_train.dt.hour,
            "dayofweek": dt_train.dt.dayofweek,
            "month": dt_train.dt.month,
            "is_weekend": (dt_train.dt.dayofweek >= 5).astype(int),
        }
    )

    # Add weather features if available
    weather_cols = ["temp_actual", "temp_forecast", "wind10m_actual", "humidity_actual"]
    for col in weather_cols:
        if col in df_train.columns:
            X_train_features[col] = df_train[col]

    # Add lag features
    if "load_actual" in df_train.columns or "load_mw" in df_train.columns:
        load_col = "load_actual" if "load_actual" in df_train.columns else "load_mw"
        for lag in [24, 48, 168]:
            X_train_features[f"load_lag_{lag}h"] = df_train[load_col].shift(lag)

    # Drop NaN from lags
    X_train_features = X_train_features.dropna()

    # Target
    y_train = df_train.loc[X_train_features.index, load_col]

    # Train model
    model = GradientBoostingRegressor(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_features)

    model.fit(X_train_scaled, y_train)

    logger.info(
        f"[OK] Trained ML load forecast model (R²: {model.score(X_train_scaled, y_train):.3f})"
    )

    # Generate forecasts
    df_pred = df_forecast.copy()
    dt_pred = pd.to_datetime(df_pred["datetime_hour"])

    X_pred_features = pd.DataFrame(
        {
            "hour": dt_pred.dt.hour,
            "dayofweek": dt_pred.dt.dayofweek,
            "month": dt_pred.dt.month,
            "is_weekend": (dt_pred.dt.dayofweek >= 5).astype(int),
        }
    )

    # Add weather forecasts
    for col in weather_cols:
        if col.replace("_actual", "_forecast") in df_pred.columns:
            X_pred_features[col] = df_pred[col.replace("_actual", "_forecast")]
        elif col in df_pred.columns:
            X_pred_features[col] = df_pred[col]
        else:
            X_pred_features[col] = 15.0  # Default

    # For lags, we'd need recent actual values - for now use dummy
    for lag in [24, 48, 168]:
        X_pred_features[f"load_lag_{lag}h"] = y_train.mean()

    # Predict
    X_pred_scaled = scaler.transform(X_pred_features)
    df_pred["load_forecast"] = model.predict(X_pred_scaled)

    # Add realistic forecast error
    forecast_error_std = 1000  # MW
    forecast_error = np.random.normal(0, forecast_error_std, len(df_pred))
    df_pred["load_forecast"] = df_pred["load_forecast"] + forecast_error

    return df_pred[["datetime_hour", "load_forecast"]]


def simulate_load_forecast_from_actual(
    df: pd.DataFrame, forecast_horizon_hours: int = 24, error_std_pct: float = 0.03
) -> pd.DataFrame:
    """Simulate load forecast from actual load with realistic errors.

    For BACKTESTING when we have actual load but want to simulate forecasts.

    Args:
        df: DataFrame with actual load
        forecast_horizon_hours: Forecast horizon
        error_std_pct: Standard deviation of forecast error as % of load

    Returns:
        DataFrame with load_forecast column added
    """
    df = df.copy()

    load_col = "load_actual" if "load_actual" in df.columns else "load_mw"

    if load_col not in df.columns:
        raise ValueError(f"No load column found. Need '{load_col}' in DataFrame")

    # Forecast error increases with horizon
    base_error_pct = error_std_pct
    horizon_factor = 1 + (forecast_horizon_hours / 24) * 0.5

    error_std = df[load_col] * base_error_pct * horizon_factor

    # Generate errors with autocorrelation
    errors = np.random.normal(0, 1, len(df)) * error_std

    # Add persistence (errors are autocorrelated)
    errors = pd.Series(errors).ewm(alpha=0.3).mean().values

    df["load_forecast"] = df[load_col] + errors

    # Ensure non-negative
    df["load_forecast"] = np.maximum(df["load_forecast"], 0)

    logger.info(
        f"[OK] Simulated load forecast from actual values\n"
        f"   Mean absolute error: {np.abs(df['load_forecast'] - df[load_col]).mean():.0f} MW "
        f"({np.abs(df['load_forecast'] - df[load_col]).mean() / df[load_col].mean() * 100:.1f}%)"
    )

    return df


def get_load_forecast_for_price_forecasting(
    df_weather: pd.DataFrame,
    df_historical_load: Optional[pd.DataFrame] = None,
    mode: str = "simple",
    forecast_horizon_hours: int = 24,
) -> pd.DataFrame:
    """Main function to get load forecasts for price forecasting.

    Args:
        df_weather: DataFrame with weather forecast data
        df_historical_load: DataFrame with historical load (for training ML model)
        mode: "simple" (heuristic), "ml" (machine learning), or "simulate" (from actual)
        forecast_horizon_hours: Forecast horizon

    Returns:
        DataFrame with load_forecast
    """
    if mode == "simple":
        historical_load = None
        if df_historical_load is not None and "load_mw" in df_historical_load.columns:
            historical_load = df_historical_load["load_mw"]

        return generate_load_forecast_simple(
            df_weather, historical_load, forecast_horizon_hours
        )

    elif mode == "ml":
        if df_historical_load is None:
            logger.warning(
                "No historical load data provided, falling back to simple model"
            )
            return get_load_forecast_for_price_forecasting(
                df_weather,
                None,
                mode="simple",
                forecast_horizon_hours=forecast_horizon_hours,
            )

        return generate_load_forecast_ml(
            df_weather, df_historical_load, forecast_horizon_hours
        )

    elif mode == "simulate":
        if df_historical_load is None:
            raise ValueError("Need historical load data for simulate mode")

        # Merge weather and load
        df_merged = df_weather.merge(
            df_historical_load[["datetime_hour", "load_mw"]],
            on="datetime_hour",
            how="left",
        )

        return simulate_load_forecast_from_actual(df_merged, forecast_horizon_hours)

    else:
        raise ValueError(f"Invalid mode: {mode}. Use 'simple', 'ml', or 'simulate'")


if __name__ == "__main__":
    # Test the module
    logging.basicConfig(level=logging.INFO)

    print("\n" + "=" * 80)
    print("TESTING LOAD FORECAST MODULE")
    print("=" * 80)

    # Create dummy weather data
    dates = pd.date_range("2024-01-01", "2024-01-07", freq="H")
    df_weather = pd.DataFrame(
        {
            "datetime_hour": dates,
            "temp_forecast": np.random.normal(10, 5, len(dates)),
        }
    )

    # Create dummy historical load
    df_historical = pd.DataFrame(
        {
            "datetime_hour": pd.date_range("2023-01-01", "2023-12-31", freq="H"),
            "load_mw": 50000
            + np.random.normal(
                0, 5000, len(pd.date_range("2023-01-01", "2023-12-31", freq="H"))
            ),
            "temp_actual": np.random.normal(
                12, 8, len(pd.date_range("2023-01-01", "2023-12-31", freq="H"))
            ),
        }
    )

    # Test simple mode
    print("\n1. Testing SIMPLE mode:")
    df_simple = get_load_forecast_for_price_forecasting(
        df_weather, df_historical, mode="simple"
    )
    print(df_simple.head())

    # Test simulate mode
    print("\n2. Testing SIMULATE mode:")
    df_weather_with_actual = df_weather.copy()
    df_weather_with_actual = df_weather_with_actual.merge(
        df_historical[["datetime_hour", "load_mw"]].head(len(df_weather)),
        on="datetime_hour",
        how="left",
    )

    df_sim = simulate_load_forecast_from_actual(df_weather_with_actual)
    print(df_sim.head())
    print(
        f"\nForecast accuracy: MAE = {np.abs(df_sim['load_forecast'] - df_sim['load_mw']).mean():.0f} MW"
    )
