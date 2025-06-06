# data_recuperation_actual_meteo.py
import requests
import pandas as pd
from pathlib import Path
from datetime import date

# === Config ===
FORECAST_DAYS = 7
OUTPUT_PATH = Path("data/modified_data/meteo_forecast_daily.csv")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

DAILY_VARS = [
    "temperature_2m_max", "temperature_2m_min", "precipitation_sum", "weather_code",
    "apparent_temperature_max", "apparent_temperature_min", "rain_sum", "snowfall_sum",
    "precipitation_hours", "sunrise", "sunset", "sunshine_duration", "daylight_duration",
    "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant",
    "shortwave_radiation_sum", "et0_fao_evapotranspiration"
]

# === Mapping INSEE -> Coordinates ===
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

# === Open-Meteo Forecast API Call ===
def fetch_forecast(insee_code, city, lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": ",".join(DAILY_VARS),
        "timezone": "Europe/Paris",
        "forecast_days": FORECAST_DAYS,
    }

    print(f" Fetching forecast for {city} (INSEE {insee_code})...")
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(data["daily"])
    df["date"] = pd.to_datetime(df["time"])
    df["insee_region"] = insee_code
    return df[["date", "insee_region"] + DAILY_VARS]


def main():
    all_regions = []
    
    # Loop through all INSEE codes and their corresponding city and coordinates
    for insee, (city, lat, lon) in INSEE_REGION_TO_CITY.items():
        df = fetch_forecast(insee, city, lat, lon)
        all_regions.append(df)

    # Concatenate all dataframes into one
    df_all = pd.concat(all_regions, ignore_index=True)

    # Define output path relative to the project base directory
    BASE_DIR = Path(__file__).resolve().parents[1]
    OUTPUT_PATH = BASE_DIR / "data" / "modified_data" / "meteo_forecast_daily.csv"
    
    # Save the final dataframe to CSV
    df_all.to_csv(OUTPUT_PATH, index=False)
    print(f"Weather forecast saved ({len(df_all)} rows) to {OUTPUT_PATH.resolve()}")

if __name__ == "__main__":
    main()

