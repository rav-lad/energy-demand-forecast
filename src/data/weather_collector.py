"""Weather data collector using Open-Meteo API."""

import pandas as pd
import requests
from datetime import datetime


class WeatherCollector:
    """Collect historical weather data from Open-Meteo API."""

    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

    def get_historical_weather(
        self, latitude: float, longitude: float, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Get historical weather data.

        Args:
            latitude: Latitude
            longitude: Longitude
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with weather data
        """
        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "daily": [
                "temperature_2m_max",
                "temperature_2m_min",
                "precipitation_sum",
                "wind_speed_10m_max",
                "shortwave_radiation_sum",
                "et0_fao_evapotranspiration",
            ],
            "timezone": "Europe/Paris",
        }

        response = requests.get(self.BASE_URL, params=params)
        response.raise_for_status()

        data = response.json()

        df = pd.DataFrame(
            {
                "date": data["daily"]["time"],
                "temperature_2m_max": data["daily"]["temperature_2m_max"],
                "temperature_2m_min": data["daily"]["temperature_2m_min"],
                "precipitation_sum": data["daily"]["precipitation_sum"],
                "wind_speed_10m_max": data["daily"]["wind_speed_10m_max"],
                "shortwave_radiation_sum": data["daily"]["shortwave_radiation_sum"],
                "et0_fao_evapotranspiration": data["daily"][
                    "et0_fao_evapotranspiration"
                ],
            }
        )

        return df
