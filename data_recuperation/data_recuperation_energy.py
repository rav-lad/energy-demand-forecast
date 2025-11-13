"""Energy consumption data loading and processing module.

This module provides functions to load and process regional energy consumption
data for electricity and gas from CSV files. Data is cleaned, aggregated, and
saved for further analysis.
"""

from pathlib import Path

import pandas as pd


def load_regional_energy_hourly(
    csv_path: str, output_path: str = "data/modified_data/energy_hourly_regional.csv"
) -> pd.DataFrame:
    """Load and process regional energy consumption data at hourly frequency.

    This function reads raw energy consumption data, cleans column names,
    converts timestamps to Europe/Paris timezone, and aggregates the data
    to hourly frequency by region.

    Args:
        csv_path: Path to the input CSV file containing raw energy data
        output_path: Path where the processed hourly data will be saved.
            Defaults to "data/modified_data/energy_hourly_regional.csv"

    Returns:
        pd.DataFrame: Processed hourly energy consumption data with columns:
            - datetime_hour: Hourly timestamp
            - insee_region: INSEE region code
            - conso_elec_mw: Electricity consumption in MW
            - conso_gaz_mw: Gas consumption in MW
    """
    df = pd.read_csv(csv_path, sep=";", encoding="utf-8")

    # Clean column names
    df.columns = df.columns.str.strip().str.replace("\xa0", " ").str.replace("\u202f", " ")

    # Parse and convert datetime to Europe/Paris timezone
    df["datetime"] = pd.to_datetime(df["Date - Heure"], utc=True).dt.tz_convert("Europe/Paris")

    # Select and rename columns
    df = df[
        [
            "datetime",
            "Code INSEE région",
            "Consommation brute électricité (MW) - RTE",
            "Consommation brute gaz totale (MW PCS 0°C)",
        ]
    ]
    df.columns = ["datetime", "insee_region", "conso_elec_mw", "conso_gaz_mw"]

    # Convert to numeric
    df["conso_elec_mw"] = pd.to_numeric(df["conso_elec_mw"], errors="coerce")
    df["conso_gaz_mw"] = pd.to_numeric(df["conso_gaz_mw"], errors="coerce")

    # Filter out data after the threshold date
    df = df[df["datetime"] < "2024-04-30"]

    # Remove timezone information and round to the nearest hour
    df["datetime"] = df["datetime"].dt.tz_localize(None)
    df["datetime_hour"] = df["datetime"].dt.ceil("h")

    # Aggregate to hourly consumption per region
    df_hour = df.groupby(["datetime_hour", "insee_region"]).sum(numeric_only=True).reset_index()

    Path("data/modified_data").mkdir(parents=True, exist_ok=True)
    df_hour.to_csv(output_path, index=False)
    return df_hour


def load_regional_energy_daily(
    _, output_path: str = "data/modified_data/energy_daily_regional.csv"
) -> pd.DataFrame:
    """Load and aggregate hourly energy data to daily frequency.

    This function reads previously processed hourly energy data and aggregates
    it to daily frequency by region.

    Args:
        _: Unused parameter (placeholder for consistency with other functions)
        output_path: Path where the daily aggregated data will be saved.
            Defaults to "data/modified_data/energy_daily_regional.csv"

    Returns:
        pd.DataFrame: Daily energy consumption data with columns:
            - date: Date
            - insee_region: INSEE region code
            - conso_elec_mw: Total daily electricity consumption in MW
            - conso_gaz_mw: Total daily gas consumption in MW
    """
    df_hour = pd.read_csv("data/modified_data/energy_hourly_regional.csv")
    df_hour["datetime_hour"] = pd.to_datetime(df_hour["datetime_hour"])

    # Filter out data after the threshold date
    df_hour = df_hour[df_hour["datetime_hour"] < "2024-04-30"]

    # Extract date from datetime
    df_hour["date"] = df_hour["datetime_hour"].dt.date

    # Aggregate to daily consumption per region
    df_day = df_hour.groupby(["date", "insee_region"]).sum(numeric_only=True).reset_index()

    Path("data/modified_data").mkdir(parents=True, exist_ok=True)
    df_day.to_csv(output_path, index=False)
    return df_day
