"""
Complete Data Collection Pipeline for Energy Trading Platform

This script collects all necessary data:
1. Energy consumption (from ODRE - French energy data)
2. Market prices (from ENTSO-E - European electricity markets)
3. Weather data (from Open-Meteo)

Usage:
    python collect_all_data.py --start_date 2023-01-01 --end_date 2024-12-31
"""

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Data directories
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw_data"
MODIFIED_DIR = DATA_DIR / "modified_data"

# Create directories
for dir_path in [RAW_DIR, MODIFIED_DIR, RAW_DIR / "market_prices"]:
    dir_path.mkdir(parents=True, exist_ok=True)


def collect_market_prices(country: str, start_date: str, end_date: str):
    """Collect electricity market prices from ENTSO-E.

    ENTSO-E API has limits on period length, so we split into chunks.
    """
    from data_collection.entsoe_connector import EntsoeConnector

    logger.info(f"=== Collecting Market Prices for {country} ===")
    logger.info(f"Period: {start_date} to {end_date}")

    try:
        connector = EntsoeConnector()

        # Split into monthly chunks (ENTSO-E API limitation)
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        all_data = []
        current_start = start_dt

        while current_start < end_dt:
            # Collect data for 1 month at a time
            current_end = min(
                current_start + pd.DateOffset(months=1),
                end_dt
            )

            logger.info(f"  Fetching {current_start.date()} to {current_end.date()}...")

            # Collect prices and load data for this chunk
            df_chunk = connector.get_market_data(
                country=country,
                start=current_start.strftime("%Y-%m-%d"),
                end=current_end.strftime("%Y-%m-%d")
            )

            if not df_chunk.empty:
                all_data.append(df_chunk)
                logger.info(f"    ✓ Collected {len(df_chunk)} records")

            current_start = current_end

        if not all_data:
            logger.warning(f"No data collected for {country}")
            return None

        # Combine all chunks
        df = pd.concat(all_data, ignore_index=True)
        df = df.sort_values("datetime").reset_index(drop=True)

        # Save to CSV
        output_path = RAW_DIR / "market_prices" / f"{country}_{start_date}_{end_date}.csv"
        df.to_csv(output_path, index=False)

        logger.info(f"✓ Saved {len(df)} total records to {output_path}")
        logger.info(f"  Mean price: {df['price_eur_mwh'].mean():.2f} EUR/MWh")
        logger.info(f"  Mean load: {df['load_mw'].mean():.2f} MW")

        return df

    except Exception as e:
        logger.error(f"Failed to collect market prices: {e}")
        raise


def collect_energy_consumption(start_date: str, end_date: str):
    """Collect French energy consumption from ODRE."""
    logger.info(f"=== Collecting Energy Consumption (ODRE) ===")
    logger.info(f"Period: {start_date} to {end_date}")

    try:
        import subprocess

        output_path = RAW_DIR / f"energy_consumption_{start_date}_{end_date}.csv"

        cmd = [
            "python", "data_collection/odre_collector.py",
            "--start_date", start_date,
            "--end_date", end_date,
            "--output", str(output_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            logger.info(f"✓ Energy consumption data saved to {output_path}")
            df = pd.read_csv(output_path)
            logger.info(f"  Collected {len(df)} records")
            return df
        else:
            logger.error(f"ODRE collection failed: {result.stderr}")
            return None

    except Exception as e:
        logger.error(f"Failed to collect energy consumption: {e}")
        raise


def collect_weather_data(start_date: str, end_date: str, frequency: str = "daily"):
    """Collect weather data from Open-Meteo."""
    from data_collection.pipeline import WeatherCollector

    logger.info(f"=== Collecting Weather Data ({frequency}) ===")
    logger.info(f"Period: {start_date} to {end_date}")

    try:
        collector = WeatherCollector(start_date=start_date, end_date=end_date)
        df = collector.collect_all_regions(frequency=frequency)

        if df.empty:
            logger.warning("No weather data collected")
            return None

        # Save to CSV
        output_path = MODIFIED_DIR / f"weather_{frequency}_{start_date}_{end_date}.csv"
        df.to_csv(output_path, index=False)

        logger.info(f"✓ Saved {len(df)} weather records to {output_path}")

        return df

    except Exception as e:
        logger.error(f"Failed to collect weather data: {e}")
        raise


def main():
    """Main data collection pipeline."""
    parser = argparse.ArgumentParser(
        description="Complete data collection for energy trading platform"
    )
    parser.add_argument(
        "--start_date",
        type=str,
        default="2023-01-01",
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end_date",
        type=str,
        default="2024-12-31",
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--countries",
        nargs="+",
        default=["FR"],
        help="Countries to collect market data for (default: FR)"
    )
    parser.add_argument(
        "--skip-prices",
        action="store_true",
        help="Skip market prices collection"
    )
    parser.add_argument(
        "--skip-energy",
        action="store_true",
        help="Skip energy consumption collection"
    )
    parser.add_argument(
        "--skip-weather",
        action="store_true",
        help="Skip weather data collection"
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("ENERGY TRADING DATA COLLECTION")
    logger.info("=" * 70)
    logger.info(f"Period: {args.start_date} to {args.end_date}")
    logger.info(f"Countries: {', '.join(args.countries)}")
    logger.info("=" * 70)

    try:
        # 1. Collect market prices
        if not args.skip_prices:
            for country in args.countries:
                collect_market_prices(country, args.start_date, args.end_date)
        else:
            logger.info("Skipping market prices collection")

        # 2. Collect energy consumption
        if not args.skip_energy and "FR" in args.countries:
            collect_energy_consumption(args.start_date, args.end_date)
        else:
            logger.info("Skipping energy consumption collection")

        # 3. Collect weather data
        if not args.skip_weather:
            collect_weather_data(args.start_date, args.end_date, frequency="daily")
            collect_weather_data(args.start_date, args.end_date, frequency="hourly")
        else:
            logger.info("Skipping weather data collection")

        logger.info("=" * 70)
        logger.info("✓ DATA COLLECTION COMPLETE")
        logger.info("=" * 70)
        logger.info("\nNext steps:")
        logger.info("1. Check collected data in data/raw_data/ and data/modified_data/")
        logger.info("2. Run data processing: python data_processing/transform_initial.py")
        logger.info("3. Train models: python scripts/train_pipeline.py")

    except KeyboardInterrupt:
        logger.warning("\nData collection interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nData collection failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
