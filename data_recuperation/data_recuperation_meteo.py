import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
import time
from data_recuperation_energy import load_regional_energy_daily, load_regional_energy_hourly
from data_recuperation_meteo import fetch_meteo_for_region, REGIONS

# Mapping INSEE region code -> (city, latitude, longitude)
INSEE_REGION_TO_CITY = {
    11: ("Paris", 48.8566, 2.3522),
    24: ("Caen", 49.1829, -0.3707),
    27: ("Rouen", 49.4431, 1.0993),
    28: ("Orléans", 47.9025, 1.9090),
    32: ("Dijon", 47.3220, 5.0415),
    44: ("Strasbourg", 48.5734, 7.7521),
    52: ("Lille", 50.6292, 3.0573),
    53: ("Reims", 49.2583, 4.0317),
    75: ("Bordeaux", 44.8378, -0.5792),
    76: ("Toulouse", 43.6047, 1.4442),
    84: ("Lyon", 45.7640, 4.8357),
    93: ("Marseille", 43.2965, 5.3698),
    94: ("Ajaccio", 41.9192, 8.7386)
}

START_DATE = "2013-01-01"
END_DATE = "2024-12-31"
RAW_DAILY_DIR = Path("data/raw_data/meteo_daily_by_region")
RAW_HOURLY_DIR = Path("data/raw_data/meteo_hourly_by_region")
RAW_DAILY_DIR.mkdir(parents=True, exist_ok=True)
RAW_HOURLY_DIR.mkdir(parents=True, exist_ok=True)

DAILY_VARS = [
    "temperature_2m_max", "temperature_2m_min", "precipitation_sum", "weather_code",
    "apparent_temperature_max", "apparent_temperature_min", "rain_sum", "snowfall_sum",
    "precipitation_hours", "sunrise", "sunset", "sunshine_duration", "daylight_duration",
    "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant",
    "shortwave_radiation_sum", "et0_fao_evapotranspiration"
]

HOURLY_VARS = [
    "temperature_2m", "precipitation", "relative_humidity_2m", "dew_point_2m", "apparent_temperature",
    "pressure_msl", "surface_pressure", "rain", "snowfall", "cloud_cover", "cloud_cover_low",
    "cloud_cover_mid", "cloud_cover_high", "shortwave_radiation", "direct_radiation",
    "direct_normal_irradiance", "diffuse_radiation", "global_tilted_irradiance", "sunshine_duration",
    "wind_speed_10m", "wind_speed_100m", "wind_direction_10m", "wind_direction_100m", "wind_gusts_10m",
    "et0_fao_evapotranspiration", "weather_code", "snow_depth", "vapour_pressure_deficit",
    "soil_temperature_0_to_7cm", "soil_moisture_0_to_7cm"
]

def fetch_meteo_for_region(insee_code: int, city: str, lat: float, lon: float, frequency: str = "daily") -> pd.DataFrame:
    print(f"Downloading {frequency} weather data for {city} (INSEE {insee_code})...")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "timezone": "Europe/Paris",
    }

    if frequency == "daily":
        params["daily"] = ",".join(DAILY_VARS)
    elif frequency == "hourly":
        params["hourly"] = ",".join(HOURLY_VARS)
    else:
        raise ValueError("frequency must be either 'daily' or 'hourly'")

    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(data[frequency])
    if frequency == "daily":
        df["date"] = pd.to_datetime(df["time"])
        df["insee_region"] = insee_code
        return df[["date", "insee_region"] + DAILY_VARS]
    else:
        df["datetime"] = pd.to_datetime(df["time"])
        df["insee_region"] = insee_code
        return df[["datetime", "insee_region"] + HOURLY_VARS]


def fetch_all_meteo_regions():
    for insee, info in REGIONS.items():
        city, lat, lon = info["city"], info["lat"], info["lon"]
        print(f"Downloading weather data for {city} (INSEE {insee})...")

        daily_path = RAW_DAILY_DIR / f"{insee}_{city}_daily.csv"
        hourly_path = RAW_HOURLY_DIR / f"{insee}_{city}_hourly.csv"

        if not daily_path.exists():
            df_daily = fetch_meteo_for_region(insee, city, lat, lon, frequency="daily")
            df_daily.to_csv(daily_path, index=False)
            time.sleep(1.2)

        if not hourly_path.exists():
            df_hourly = fetch_meteo_for_region(insee, city, lat, lon, frequency="hourly")
            df_hourly.to_csv(hourly_path, index=False)
            time.sleep(1.2)

    print("Weather download completed for all regions.")
