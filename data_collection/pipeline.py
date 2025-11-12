"""
Unified Data Collection Pipeline for Energy Trading Platform

This module replaces multiple scattered scripts with a single, optimized pipeline:
- Weather data collection (historical + forecast)
- Energy consumption data
- Market prices
- Data merging and storage

All data is stored as CSV files in local directories.
"""

import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, date
import time
import logging
from typing import Literal, List, Optional
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# French regions mapping (INSEE code -> coordinates)
REGIONS = {
    11: {"city": "Paris", "lat": 48.8566, "lon": 2.3522},
    24: {"city": "Caen", "lat": 49.1829, "lon": -0.3707},
    27: {"city": "Rouen", "lat": 49.4431, "lon": 1.0993},
    28: {"city": "Orléans", "lat": 47.9025, "lon": 1.9090},
    32: {"city": "Dijon", "lat": 47.3220, "lon": 5.0415},
    44: {"city": "Strasbourg", "lat": 48.5734, "lon": 7.7521},
    52: {"city": "Lille", "lat": 50.6292, "lon": 3.0573},
    53: {"city": "Reims", "lat": 49.2583, "lon": 4.0317},
    75: {"city": "Bordeaux", "lat": 44.8378, "lon": -0.5792},
    76: {"city": "Toulouse", "lat": 43.6047, "lon": 1.4442},
    84: {"city": "Lyon", "lat": 45.7640, "lon": 4.8357},
    93: {"city": "Marseille", "lat": 43.2965, "lon": 5.3698},
    94: {"city": "Ajaccio", "lat": 41.9192, "lon": 8.7386},
}

# Weather variables
DAILY_WEATHER_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
]

HOURLY_WEATHER_VARS = [
    "temperature_2m",
    "precipitation",
    "relative_humidity_2m",
    "wind_speed_10m",
    "shortwave_radiation",
    "cloud_cover",
]

# Data paths
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw_data"
MODIFIED_DIR = DATA_DIR / "modified_data"

# Create directories
for dir_path in [RAW_DIR, MODIFIED_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Weather Data Collection (Open-Meteo)
# =============================================================================


class WeatherCollector:
    """Optimized weather data collector using Open-Meteo API."""

    def __init__(self, start_date: str = "2020-01-01", end_date: Optional[str] = None):
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")

    def fetch_historical(
        self, insee_code: int, frequency: Literal["daily", "hourly"] = "daily"
    ) -> pd.DataFrame:
        """
        Fetch historical weather data for a region.

        Args:
            insee_code: INSEE region code
            frequency: 'daily' or 'hourly'

        Returns:
            DataFrame with weather data
        """
        region = REGIONS[insee_code]
        city, lat, lon = region["city"], region["lat"], region["lon"]

        logger.info(f"Fetching {frequency} weather for {city} (INSEE {insee_code})...")

        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "timezone": "Europe/Paris",
        }

        # Add variables based on frequency
        if frequency == "daily":
            params["daily"] = ",".join(DAILY_WEATHER_VARS)
        else:
            params["hourly"] = ",".join(HOURLY_WEATHER_VARS)

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Parse response
            df = pd.DataFrame(data[frequency])
            time_col = "time"
            df["date" if frequency == "daily" else "datetime"] = pd.to_datetime(
                df[time_col]
            )
            df["insee_region"] = insee_code

            # Return only relevant columns
            time_key = "date" if frequency == "daily" else "datetime"
            vars_list = (
                DAILY_WEATHER_VARS if frequency == "daily" else HOURLY_WEATHER_VARS
            )

            return df[[time_key, "insee_region"] + vars_list]

        except Exception as e:
            logger.error(f"Failed to fetch weather for {city}: {e}")
            return pd.DataFrame()

    def fetch_forecast(self, insee_code: int, days: int = 7) -> pd.DataFrame:
        """
        Fetch weather forecast for a region.

        Args:
            insee_code: INSEE region code
            days: Number of forecast days

        Returns:
            DataFrame with forecast data
        """
        region = REGIONS[insee_code]
        city, lat, lon = region["city"], region["lat"], region["lon"]

        logger.info(f"Fetching {days}-day forecast for {city}...")

        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": ",".join(DAILY_WEATHER_VARS),
            "timezone": "Europe/Paris",
            "forecast_days": days,
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            df = pd.DataFrame(data["daily"])
            df["date"] = pd.to_datetime(df["time"])
            df["insee_region"] = insee_code

            return df[["date", "insee_region"] + DAILY_WEATHER_VARS]

        except Exception as e:
            logger.error(f"Failed to fetch forecast for {city}: {e}")
            return pd.DataFrame()

    def collect_all_regions(
        self, frequency: Literal["daily", "hourly"] = "daily", forecast: bool = False
    ) -> pd.DataFrame:
        """
        Collect weather data for all regions.

        Args:
            frequency: 'daily' or 'hourly'
            forecast: If True, collect forecast instead of historical

        Returns:
            Combined DataFrame for all regions
        """
        all_data = []

        for insee_code in REGIONS.keys():
            if forecast:
                df = self.fetch_forecast(insee_code)
            else:
                df = self.fetch_historical(insee_code, frequency)

            if not df.empty:
                all_data.append(df)

            # Rate limiting
            time.sleep(1.2)

        if not all_data:
            logger.warning("No weather data collected")
            return pd.DataFrame()

        # Combine and sort
        df_combined = pd.concat(all_data, ignore_index=True)
        time_col = "date" if frequency == "daily" or forecast else "datetime"
        df_combined = df_combined.sort_values([time_col, "insee_region"]).reset_index(
            drop=True
        )

        logger.info(f"Collected {len(df_combined):,} weather records")
        return df_combined


# =============================================================================
# Data Merger
# =============================================================================


class DataMerger:
    """Merge energy, weather, and market data."""

    @staticmethod
    def merge_daily(
        energy_path: Path, weather_path: Path, output_path: Path
    ) -> pd.DataFrame:
        """
        Merge daily energy and weather data.

        Args:
            energy_path: Path to energy CSV
            weather_path: Path to weather CSV
            output_path: Path to save merged data

        Returns:
            Merged DataFrame
        """
        logger.info("Merging daily data...")

        df_energy = pd.read_csv(energy_path)
        df_weather = pd.read_csv(weather_path)

        # Ensure datetime format
        df_energy["date"] = pd.to_datetime(df_energy["date"])
        df_weather["date"] = pd.to_datetime(df_weather["date"])

        # Merge on date + region
        df_merged = pd.merge(
            df_energy, df_weather, on=["date", "insee_region"], how="inner"
        )

        # Sort chronologically
        df_merged = df_merged.sort_values(["date", "insee_region"]).reset_index(
            drop=True
        )

        # Save
        df_merged.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df_merged):,} merged daily records to {output_path}")

        return df_merged

    @staticmethod
    def merge_hourly(
        energy_path: Path,
        weather_path: Path,
        output_path: Path,
        cutoff_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Merge hourly energy and weather data.

        Args:
            energy_path: Path to energy CSV
            weather_path: Path to weather CSV
            output_path: Path to save merged data
            cutoff_date: Optional cutoff date (YYYY-MM-DD)

        Returns:
            Merged DataFrame
        """
        logger.info("Merging hourly data...")

        df_energy = pd.read_csv(energy_path)
        df_weather = pd.read_csv(weather_path)

        # Ensure datetime format
        df_energy["datetime"] = pd.to_datetime(df_energy["datetime"])
        df_weather["datetime"] = pd.to_datetime(df_weather["datetime"])

        # Apply cutoff if specified
        if cutoff_date:
            cutoff = pd.to_datetime(cutoff_date)
            df_energy = df_energy[df_energy["datetime"] <= cutoff]
            df_weather = df_weather[df_weather["datetime"] <= cutoff]

        # Merge
        df_merged = pd.merge(
            df_energy, df_weather, on=["datetime", "insee_region"], how="inner"
        )

        # Sort
        df_merged = df_merged.sort_values(["datetime", "insee_region"]).reset_index(
            drop=True
        )

        # Save
        df_merged.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df_merged):,} merged hourly records to {output_path}")

        return df_merged


# =============================================================================
# Main Pipeline
# =============================================================================


def collect_weather_historical(frequency: Literal["daily", "hourly"] = "daily"):
    """Collect historical weather data for all regions."""
    collector = WeatherCollector(start_date="2020-01-01")
    df = collector.collect_all_regions(frequency=frequency)

    output_path = MODIFIED_DIR / f"weather_{frequency}.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved weather data to {output_path}")

    return df


def collect_weather_forecast(days: int = 7):
    """Collect weather forecast for all regions."""
    collector = WeatherCollector()
    df = collector.collect_all_regions(forecast=True)

    output_path = MODIFIED_DIR / "weather_forecast.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved forecast to {output_path}")

    return df


def merge_all_data(frequency: Literal["daily", "hourly"] = "daily"):
    """Merge energy and weather data."""
    energy_path = MODIFIED_DIR / f"energy_{frequency}.csv"
    weather_path = MODIFIED_DIR / f"weather_{frequency}.csv"
    output_path = MODIFIED_DIR / f"merged_{frequency}.csv"

    if not energy_path.exists():
        logger.error(f"Energy data not found: {energy_path}")
        return None

    if not weather_path.exists():
        logger.error(f"Weather data not found: {weather_path}")
        return None

    if frequency == "daily":
        return DataMerger.merge_daily(energy_path, weather_path, output_path)
    else:
        return DataMerger.merge_hourly(energy_path, weather_path, output_path)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Energy Trading Data Collection Pipeline"
    )
    parser.add_argument(
        "action",
        choices=["weather-historical", "weather-forecast", "merge"],
        help="Action to perform",
    )
    parser.add_argument(
        "--frequency",
        choices=["daily", "hourly"],
        default="daily",
        help="Data frequency",
    )
    parser.add_argument(
        "--forecast-days", type=int, default=7, help="Number of forecast days"
    )

    args = parser.parse_args()

    if args.action == "weather-historical":
        collect_weather_historical(args.frequency)
    elif args.action == "weather-forecast":
        collect_weather_forecast(args.forecast_days)
    elif args.action == "merge":
        merge_all_data(args.frequency)

    logger.info("Pipeline completed!")
