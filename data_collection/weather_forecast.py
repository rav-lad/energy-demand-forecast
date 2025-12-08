"""Weather forecast data collection for price forecasting.

This module provides functions to collect weather forecast data from Open-Meteo API
and simulate realistic forecast errors for historical backtesting.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)


def get_weather_forecast_realtime(
    lat: float = 48.8566,
    lon: float = 2.3522,
    forecast_days: int = 2
) -> pd.DataFrame:
    """Get real-time weather forecast from Open-Meteo API.

    This is for PRODUCTION use - gets actual forecasts for trading.

    Args:
        lat: Latitude (default: Paris)
        lon: Longitude (default: Paris)
        forecast_days: Number of days to forecast (max 16)

    Returns:
        DataFrame with hourly weather forecasts
    """
    url = (
        "https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}"
        "&hourly=temperature_2m,relative_humidity_2m,precipitation,"
        "windspeed_10m,windspeed_100m,shortwave_radiation,cloudcover"
        f"&forecast_days={forecast_days}"
        "&timezone=Europe/Paris"
    )

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        data = response.json()

        df = pd.DataFrame(data["hourly"])
        df["datetime_hour"] = pd.to_datetime(df["time"])
        df = df.drop(columns=["time"])

        # Rename columns for consistency
        df = df.rename(columns={
            "temperature_2m": "temp_forecast",
            "windspeed_10m": "wind10m_forecast",
            "windspeed_100m": "wind100m_forecast",
            "shortwave_radiation": "solar_radiation_forecast",
            "relative_humidity_2m": "humidity_forecast",
            "precipitation": "precipitation_forecast",
            "cloudcover": "cloudcover_forecast"
        })

        logger.info(f"✅ Retrieved {len(df)} hours of weather forecast")
        return df

    except Exception as e:
        logger.error(f"Failed to get weather forecast: {e}")
        raise


def get_historical_weather_with_simulated_forecast(
    start_date: str,
    end_date: str,
    lat: float = 48.8566,
    lon: float = 2.3522,
    forecast_horizon_hours: int = 24
) -> pd.DataFrame:
    """Get historical weather data and simulate forecast errors for backtesting.

    This is for BACKTESTING - simulates what forecasts would have been.

    The key insight: we have historical ACTUAL weather, but for backtesting we need
    to simulate what the FORECAST would have been 24h earlier (with realistic errors).

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        lat: Latitude
        lon: Longitude
        forecast_horizon_hours: Forecast horizon (default 24h)

    Returns:
        DataFrame with:
        - Actual weather at time t
        - Simulated forecast made at t-24h for time t
    """
    # Get historical ACTUAL weather
    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        f"latitude={lat}&longitude={lon}"
        f"&start_date={start_date}&end_date={end_date}"
        "&hourly=temperature_2m,relative_humidity_2m,precipitation,"
        "windspeed_10m,windspeed_100m,shortwave_radiation,cloudcover"
        "&timezone=Europe/Paris"
    )

    try:
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        data = response.json()

        df = pd.DataFrame(data["hourly"])
        df["datetime_hour"] = pd.to_datetime(df["time"])
        df = df.drop(columns=["time"])
        df = df.set_index("datetime_hour")

        # Rename actual values
        df = df.rename(columns={
            "temperature_2m": "temp_actual",
            "windspeed_10m": "wind10m_actual",
            "windspeed_100m": "wind100m_actual",
            "shortwave_radiation": "solar_radiation_actual",
            "relative_humidity_2m": "humidity_actual",
            "precipitation": "precipitation_actual",
            "cloudcover": "cloudcover_actual"
        })

        # Simulate forecast errors (what forecast at t would predict for t+24h)
        df = add_realistic_forecast_errors(df, forecast_horizon_hours)

        logger.info(
            f"✅ Retrieved {len(df)} hours of historical weather with simulated forecasts\n"
            f"   Date range: {df.index.min()} to {df.index.max()}"
        )

        return df.reset_index()

    except Exception as e:
        logger.error(f"Failed to get historical weather: {e}")
        raise


def add_realistic_forecast_errors(
    df: pd.DataFrame,
    horizon_hours: int = 24
) -> pd.DataFrame:
    """Add realistic forecast errors to simulate what forecasts would have been.

    Forecast errors increase with horizon and depend on meteorological conditions.

    References:
    - Temperature: MAE ~2-3°C at 24h horizon
    - Wind speed: MAE ~2-3 m/s at 24h horizon
    - Solar radiation: MAE ~100-150 W/m² at 24h horizon

    Args:
        df: DataFrame with actual weather values
        horizon_hours: Forecast horizon

    Returns:
        DataFrame with added forecast columns
    """
    df = df.copy()

    # Temperature forecast error
    # - Systematic bias: forecasts slightly warmer in winter, cooler in summer
    # - Random error: increases with horizon, autocorrelated
    temp_bias = 0.5 * np.sin(2 * np.pi * df.index.dayofyear / 365)
    temp_random_error = np.random.normal(0, 2.5, len(df))

    # Add autocorrelation (forecast errors persist)
    temp_random_error = pd.Series(temp_random_error).ewm(alpha=0.3).mean().values

    df["temp_forecast"] = df["temp_actual"] + temp_bias + temp_random_error

    # Wind speed forecast error
    # - Higher relative error at low wind speeds
    # - Scale with actual wind speed
    wind10m_rel_error = np.random.normal(0, 0.25, len(df))  # 25% relative error
    wind10m_abs_error = df["wind10m_actual"] * wind10m_rel_error + np.random.normal(0, 1.5, len(df))
    df["wind10m_forecast"] = np.maximum(0, df["wind10m_actual"] + wind10m_abs_error)

    wind100m_rel_error = np.random.normal(0, 0.20, len(df))  # 20% relative error
    wind100m_abs_error = df["wind100m_actual"] * wind100m_rel_error + np.random.normal(0, 2.0, len(df))
    df["wind100m_forecast"] = np.maximum(0, df["wind100m_actual"] + wind100m_abs_error)

    # Solar radiation forecast error
    # - Depends on cloud cover uncertainty
    # - Zero at night (easy to predict)
    # - Higher error during partly cloudy conditions
    is_daytime = (df.index.hour >= 6) & (df.index.hour <= 20)
    solar_error = np.random.normal(0, 100, len(df)) * is_daytime
    df["solar_radiation_forecast"] = np.maximum(0, df["solar_radiation_actual"] + solar_error)

    # Humidity forecast error (small)
    humidity_error = np.random.normal(0, 5, len(df))
    df["humidity_forecast"] = np.clip(df["humidity_actual"] + humidity_error, 0, 100)

    # Precipitation forecast (binary + amount)
    # Very hard to predict! Miss/false alarm common
    precip_error = np.random.normal(0, 0.5, len(df))
    df["precipitation_forecast"] = np.maximum(0, df["precipitation_actual"] + precip_error)

    # Cloud cover forecast
    cloud_error = np.random.normal(0, 15, len(df))
    df["cloudcover_forecast"] = np.clip(df["cloudcover_actual"] + cloud_error, 0, 100)

    logger.info(
        f"✅ Added realistic forecast errors for {horizon_hours}h horizon\n"
        f"   Temp MAE: {np.abs(df['temp_forecast'] - df['temp_actual']).mean():.2f}°C\n"
        f"   Wind10m MAE: {np.abs(df['wind10m_forecast'] - df['wind10m_actual']).mean():.2f} m/s\n"
        f"   Solar MAE: {np.abs(df['solar_radiation_forecast'] - df['solar_radiation_actual']).mean():.2f} W/m²"
    )

    return df


def aggregate_regional_weather(
    df_weather: pd.DataFrame,
    regions: list = None
) -> pd.DataFrame:
    """Aggregate weather data across multiple regions for national forecast.

    Args:
        df_weather: DataFrame with regional weather data
        regions: List of region codes to aggregate (None = all)

    Returns:
        DataFrame with aggregated national weather
    """
    if "insee_region" not in df_weather.columns:
        # Already aggregated or single region
        return df_weather

    if regions is not None:
        df_weather = df_weather[df_weather["insee_region"].isin(regions)]

    # Aggregate by taking weighted mean (weight by population or area)
    # For now, simple mean
    agg_dict = {
        col: "mean" for col in df_weather.columns
        if col not in ["datetime_hour", "insee_region"]
    }

    df_agg = df_weather.groupby("datetime_hour").agg(agg_dict).reset_index()

    logger.info(f"✅ Aggregated weather data across {df_weather['insee_region'].nunique()} regions")

    return df_agg


def get_weather_for_price_forecasting(
    start_date: str,
    end_date: str,
    mode: str = "backtest",
    forecast_horizon_hours: int = 24
) -> pd.DataFrame:
    """Main function to get weather data for price forecasting.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        mode: "backtest" (with simulated errors) or "realtime" (actual forecasts)
        forecast_horizon_hours: Forecast horizon

    Returns:
        DataFrame with weather forecasts suitable for price modeling
    """
    if mode == "backtest":
        # Historical data with simulated forecast errors
        df = get_historical_weather_with_simulated_forecast(
            start_date, end_date,
            forecast_horizon_hours=forecast_horizon_hours
        )
    elif mode == "realtime":
        # Real-time forecasts for trading
        df = get_weather_forecast_realtime(forecast_days=2)
        # Filter to date range
        df = df[
            (df["datetime_hour"] >= start_date) &
            (df["datetime_hour"] <= end_date)
        ]
    else:
        raise ValueError(f"Invalid mode: {mode}. Use 'backtest' or 'realtime'")

    return df


if __name__ == "__main__":
    # Test the module
    logging.basicConfig(level=logging.INFO)

    # Test backtest mode
    print("\n" + "="*80)
    print("TESTING BACKTEST MODE (Historical + Simulated Forecasts)")
    print("="*80)

    df = get_weather_for_price_forecasting(
        start_date="2024-01-01",
        end_date="2024-01-07",
        mode="backtest",
        forecast_horizon_hours=24
    )

    print("\n📊 Sample data:")
    print(df.head())

    print("\n📈 Forecast accuracy:")
    print(f"Temperature MAE: {np.abs(df['temp_forecast'] - df['temp_actual']).mean():.2f}°C")
    print(f"Wind MAE: {np.abs(df['wind100m_forecast'] - df['wind100m_actual']).mean():.2f} m/s")

    # Test realtime mode
    print("\n" + "="*80)
    print("TESTING REALTIME MODE (Current Forecasts)")
    print("="*80)

    df_rt = get_weather_forecast_realtime(forecast_days=2)
    print("\n📊 Current forecast:")
    print(df_rt.head())
