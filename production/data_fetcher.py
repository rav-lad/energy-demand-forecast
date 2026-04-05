"""Automated incremental data fetcher for production pipeline.

Fetches new data from ENTSO-E, Open-Meteo, and ODRE since the last update.
Uses the existing data collection modules with caching and validation.
"""

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

LAST_UPDATE_FILE = project_root / "data" / "cache" / ".last_update.json"
RAW_DATA_DIR = project_root / "data" / "raw_data"
PROCESSED_DIR = project_root / "data" / "processed"


def _ensure_dirs():
    """Create data directories if they don't exist."""
    for d in [RAW_DATA_DIR, PROCESSED_DIR, LAST_UPDATE_FILE.parent]:
        d.mkdir(parents=True, exist_ok=True)
    for sub in ["energy", "weather", "market_prices", "fundamentals"]:
        (RAW_DATA_DIR / sub).mkdir(parents=True, exist_ok=True)


def _load_last_update() -> dict:
    """Load last update timestamps."""
    if LAST_UPDATE_FILE.exists():
        with open(LAST_UPDATE_FILE) as f:
            return json.load(f)
    return {}


def _save_last_update(updates: dict):
    """Save last update timestamps."""
    _ensure_dirs()
    with open(LAST_UPDATE_FILE, "w") as f:
        json.dump(updates, f, indent=2, default=str)


def fetch_entsoe_prices(
    start_date: Optional[str] = None,
    api_key: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Fetch day-ahead prices from ENTSO-E (incremental).

    Args:
        start_date: Override start date (default: day after last fetch).
        api_key: ENTSO-E API key (default: from env).

    Returns:
        DataFrame with new price data, or None if fetch fails.
    """
    import os

    api_key = api_key or os.getenv("ENTSOE_API_KEY")
    if not api_key:
        logger.warning("ENTSOE_API_KEY not set, skipping price fetch")
        return None

    updates = _load_last_update()
    if start_date is None:
        last = updates.get("entsoe_prices")
        start_date = (
            (datetime.fromisoformat(last) + timedelta(days=1)).strftime("%Y-%m-%d")
            if last
            else (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        )

    end_date = datetime.now().strftime("%Y-%m-%d")

    if start_date >= end_date:
        logger.info("ENTSO-E prices already up to date")
        return None

    try:
        from data_collection.entsoe_connector import EntsoeConnector

        connector = EntsoeConnector(api_key=api_key)
        df = connector.fetch_day_ahead_prices(
            start_date=start_date,
            end_date=end_date,
            country_code="FR",
        )

        if df is not None and len(df) > 0:
            # Append to existing file
            output_file = RAW_DATA_DIR / "market_prices" / "day_ahead_prices_FR.csv"
            if output_file.exists():
                existing = pd.read_csv(output_file)
                df = pd.concat([existing, df]).drop_duplicates(
                    subset=["datetime"], keep="last"
                )
            df.to_csv(output_file, index=False)

            updates["entsoe_prices"] = end_date
            _save_last_update(updates)
            logger.info(f"Fetched {len(df)} ENTSO-E price records up to {end_date}")

        return df

    except Exception as e:
        logger.error(f"ENTSO-E price fetch failed: {e}")
        return None


def fetch_weather_data(
    start_date: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Fetch weather data from Open-Meteo (incremental).

    Args:
        start_date: Override start date.

    Returns:
        DataFrame with new weather data, or None if fetch fails.
    """
    updates = _load_last_update()
    if start_date is None:
        last = updates.get("weather")
        start_date = (
            (datetime.fromisoformat(last) + timedelta(days=1)).strftime("%Y-%m-%d")
            if last
            else (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        )

    end_date = datetime.now().strftime("%Y-%m-%d")

    if start_date >= end_date:
        logger.info("Weather data already up to date")
        return None

    try:
        from data_collection.weather_collector import fetch_weather_history

        df = fetch_weather_history(
            start_date=start_date,
            end_date=end_date,
        )

        if df is not None and len(df) > 0:
            output_file = RAW_DATA_DIR / "weather" / "weather_france.csv"
            if output_file.exists():
                existing = pd.read_csv(output_file)
                df = pd.concat([existing, df]).drop_duplicates(keep="last")
            df.to_csv(output_file, index=False)

            updates["weather"] = end_date
            _save_last_update(updates)
            logger.info(f"Fetched weather data up to {end_date}")

        return df

    except Exception as e:
        logger.error(f"Weather fetch failed: {e}")
        return None


def fetch_odre_load(
    start_date: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Fetch electricity load data from ODRE (incremental).

    Args:
        start_date: Override start date.

    Returns:
        DataFrame with new load data, or None if fetch fails.
    """
    updates = _load_last_update()
    if start_date is None:
        last = updates.get("odre_load")
        start_date = (
            (datetime.fromisoformat(last) + timedelta(days=1)).strftime("%Y-%m-%d")
            if last
            else (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        )

    end_date = datetime.now().strftime("%Y-%m-%d")

    if start_date >= end_date:
        logger.info("ODRE load data already up to date")
        return None

    try:
        from data_collection.odre_collector import OdreCollector

        collector = OdreCollector()
        df = collector.fetch_consumption(
            start_date=start_date,
            end_date=end_date,
        )

        if df is not None and len(df) > 0:
            output_file = RAW_DATA_DIR / "energy" / "energy_hourly_regional.csv"
            if output_file.exists():
                existing = pd.read_csv(output_file)
                df = pd.concat([existing, df]).drop_duplicates(keep="last")
            df.to_csv(output_file, index=False)

            updates["odre_load"] = end_date
            _save_last_update(updates)
            logger.info(f"Fetched ODRE load data up to {end_date}")

        return df

    except Exception as e:
        logger.error(f"ODRE load fetch failed: {e}")
        return None


def fetch_all(alert_service=None) -> dict:
    """Run all data fetchers with error handling and alerting.

    Args:
        alert_service: Optional AlertService instance for failure notifications.

    Returns:
        Dictionary with fetch status for each source.
    """
    _ensure_dirs()
    results = {}

    # ENTSO-E prices
    try:
        df = fetch_entsoe_prices()
        results["entsoe_prices"] = "ok" if df is not None else "up_to_date"
    except Exception as e:
        results["entsoe_prices"] = f"error: {e}"
        if alert_service:
            alert_service.alert_data_failure("ENTSO-E", str(e))

    # Weather
    try:
        df = fetch_weather_data()
        results["weather"] = "ok" if df is not None else "up_to_date"
    except Exception as e:
        results["weather"] = f"error: {e}"
        if alert_service:
            alert_service.alert_data_failure("Open-Meteo", str(e))

    # ODRE load
    try:
        df = fetch_odre_load()
        results["odre_load"] = "ok" if df is not None else "up_to_date"
    except Exception as e:
        results["odre_load"] = f"error: {e}"
        if alert_service:
            alert_service.alert_data_failure("ODRE", str(e))

    logger.info(f"Data fetch complete: {results}")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    results = fetch_all()
    print(f"\nFetch results: {json.dumps(results, indent=2)}")
