"""
Unified data collection pipeline for energy trading.

This module provides a clean, optimized pipeline for collecting:
- Weather data (historical and forecast) from Open-Meteo
- Energy consumption data
- Market prices
- Merging and storage as CSV
"""

from .pipeline import (
    DataMerger,
    WeatherCollector,
    collect_weather_forecast,
    collect_weather_historical,
    merge_all_data,
)

__all__ = [
    "WeatherCollector",
    "DataMerger",
    "collect_weather_historical",
    "collect_weather_forecast",
    "merge_all_data",
]
