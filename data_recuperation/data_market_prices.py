"""
Market Price Data Collection from ENTSO-E Transparency Platform

This script collects day-ahead electricity prices from the ENTSO-E API.

Requirements:
    - Free API key from https://transparency.entsoe.eu/
    - Set ENTSOE_API_KEY in your .env file

Usage:
    python data_market_prices.py --start_date 2020-01-01 --end_date 2024-12-31 --countries FR DE ES
"""

import os
import sys
import argparse
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import time
import logging

try:
    from entsoe import EntsoePandasClient
except ImportError:
    print("ERROR: entsoe-py library not installed.")
    print("Install it with: pip install entsoe-py")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
COUNTRIES_MAPPING = {
    "FR": "10YFR-RTE------C",  # France
    "DE": "10Y1001A1001A83F",  # Germany
    "ES": "10YES-REE------0",  # Spain
    "IT": "10YIT-GRTN-----B",  # Italy
    "BE": "10YBE----------2",  # Belgium
    "NL": "10YNL----------L",  # Netherlands
    "CH": "10YCH-SWISSGRIDZ",  # Switzerland
    "AT": "10YAT-APG------L",  # Austria
    "GB": "10YGB----------A",  # Great Britain
}


def get_entsoe_client():
    """
    Initialize ENTSO-E API client with API key from environment.

    Returns:
        EntsoePandasClient: Initialized client

    Raises:
        ValueError: If API key not found
    """
    api_key = os.getenv("ENTSOE_API_KEY")

    if not api_key:
        raise ValueError(
            "ENTSOE_API_KEY not found in environment variables.\n"
            "Please:\n"
            "1. Get a free API key from https://transparency.entsoe.eu/\n"
            "2. Create a .env file (copy from .env.example)\n"
            "3. Set ENTSOE_API_KEY=your_key_here"
        )

    return EntsoePandasClient(api_key=api_key)


def fetch_day_ahead_prices(client, country_code, start_date, end_date):
    """
    Fetch day-ahead electricity prices for a specific country.

    Args:
        client: EntsoePandasClient instance
        country_code: Two-letter country code (e.g., 'FR', 'DE')
        start_date: Start date (datetime or string 'YYYY-MM-DD')
        end_date: End date (datetime or string 'YYYY-MM-DD')

    Returns:
        pd.DataFrame: DataFrame with datetime index and price column
    """
    if isinstance(start_date, str):
        start_date = pd.Timestamp(start_date, tz="Europe/Paris")
    if isinstance(end_date, str):
        end_date = pd.Timestamp(end_date, tz="Europe/Paris")

    country_domain = COUNTRIES_MAPPING.get(country_code)
    if not country_domain:
        raise ValueError(
            f"Country code {country_code} not supported. Available: {list(COUNTRIES_MAPPING.keys())}"
        )

    logger.info(
        f"Fetching day-ahead prices for {country_code} from {start_date.date()} to {end_date.date()}"
    )

    try:
        # Fetch day-ahead prices
        prices = client.query_day_ahead_prices(
            country_domain, start=start_date, end=end_date
        )

        # Convert to DataFrame if Series
        if isinstance(prices, pd.Series):
            df = pd.DataFrame(prices, columns=[f"price_{country_code}"])
        else:
            df = prices
            df.columns = [f"price_{country_code}"]

        # Add metadata columns
        df["country"] = country_code

        logger.info(f"Successfully fetched {len(df)} records for {country_code}")
        return df

    except Exception as e:
        logger.error(f"Error fetching data for {country_code}: {str(e)}")
        return None


def fetch_intraday_prices(client, country_code, start_date, end_date):
    """
    Fetch intraday electricity prices (if available).

    Note: Not all countries provide intraday prices through ENTSO-E.
    This is optional and may fail for some countries.

    Args:
        client: EntsoePandasClient instance
        country_code: Two-letter country code
        start_date: Start date
        end_date: End date

    Returns:
        pd.DataFrame or None: Intraday prices if available
    """
    country_domain = COUNTRIES_MAPPING.get(country_code)
    if not country_domain:
        return None

    logger.info(f"Attempting to fetch intraday prices for {country_code}")

    try:
        # Note: This method may not exist or work for all countries
        # You may need to use alternative methods depending on the country
        prices = client.query_intraday_offered_capacity(
            country_domain, start=start_date, end=end_date
        )
        logger.info(f"Successfully fetched intraday data for {country_code}")
        return prices
    except Exception as e:
        logger.warning(f"Intraday prices not available for {country_code}: {str(e)}")
        return None


def save_prices_data(df, country_code, output_dir):
    """
    Save prices data to CSV file.

    Args:
        df: DataFrame with prices
        country_code: Country code
        output_dir: Output directory path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save to CSV
    output_file = output_dir / f"day_ahead_prices_{country_code}.csv"
    df.to_csv(output_file)
    logger.info(f"Saved prices to {output_file}")

    # Basic statistics
    logger.info(f"\nPrice statistics for {country_code}:")
    logger.info(f"  Mean: {df[f'price_{country_code}'].mean():.2f} EUR/MWh")
    logger.info(f"  Std: {df[f'price_{country_code}'].std():.2f} EUR/MWh")
    logger.info(f"  Min: {df[f'price_{country_code}'].min():.2f} EUR/MWh")
    logger.info(f"  Max: {df[f'price_{country_code}'].max():.2f} EUR/MWh")


def fetch_multiple_countries(countries, start_date, end_date, output_dir, sleep_time=2):
    """
    Fetch day-ahead prices for multiple countries.

    Args:
        countries: List of country codes
        start_date: Start date
        end_date: End date
        output_dir: Output directory
        sleep_time: Seconds to wait between API calls (to respect rate limits)

    Returns:
        dict: Dictionary mapping country codes to DataFrames
    """
    client = get_entsoe_client()
    results = {}

    for country in countries:
        try:
            df = fetch_day_ahead_prices(client, country, start_date, end_date)
            if df is not None:
                save_prices_data(df, country, output_dir)
                results[country] = df

            # Sleep to respect API rate limits
            if country != countries[-1]:  # Don't sleep after last country
                logger.info(f"Sleeping {sleep_time}s to respect API rate limits...")
                time.sleep(sleep_time)

        except Exception as e:
            logger.error(f"Failed to process {country}: {str(e)}")
            continue

    return results


def merge_country_prices(results, output_dir):
    """
    Merge prices from multiple countries into a single DataFrame.

    Args:
        results: Dictionary of country_code -> DataFrame
        output_dir: Output directory

    Returns:
        pd.DataFrame: Merged DataFrame with all countries
    """
    if not results:
        logger.warning("No data to merge")
        return None

    logger.info("Merging prices from all countries...")

    # Start with first country
    merged = None
    for country, df in results.items():
        price_col = f"price_{country}"
        df_temp = df[[price_col]].copy()

        if merged is None:
            merged = df_temp
        else:
            merged = merged.join(df_temp, how="outer")

    # Save merged data
    output_file = Path(output_dir) / "day_ahead_prices_all_countries.csv"
    merged.to_csv(output_file)
    logger.info(f"Saved merged prices to {output_file}")

    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Fetch electricity market prices from ENTSO-E",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch prices for France for 2023
  python data_market_prices.py --start_date 2023-01-01 --end_date 2023-12-31 --countries FR

  # Fetch for multiple countries
  python data_market_prices.py --start_date 2020-01-01 --end_date 2024-12-31 --countries FR DE ES IT

Available countries:
  FR (France), DE (Germany), ES (Spain), IT (Italy), BE (Belgium),
  NL (Netherlands), CH (Switzerland), AT (Austria), GB (Great Britain)
        """,
    )

    parser.add_argument(
        "--start_date", type=str, default="2020-01-01", help="Start date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--end_date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--countries",
        nargs="+",
        default=["FR"],
        help="List of country codes (e.g., FR DE ES)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw_data/market_prices",
        help="Output directory for CSV files",
    )

    parser.add_argument(
        "--sleep_time", type=int, default=2, help="Seconds to sleep between API calls"
    )

    args = parser.parse_args()

    # Validate dates
    try:
        start_date = pd.Timestamp(args.start_date, tz="Europe/Paris")
        end_date = pd.Timestamp(args.end_date, tz="Europe/Paris")

        if end_date <= start_date:
            raise ValueError("End date must be after start date")

    except Exception as e:
        logger.error(f"Invalid date format: {str(e)}")
        sys.exit(1)

    # Validate countries
    invalid_countries = [c for c in args.countries if c not in COUNTRIES_MAPPING]
    if invalid_countries:
        logger.error(f"Invalid country codes: {invalid_countries}")
        logger.error(f"Available countries: {list(COUNTRIES_MAPPING.keys())}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("ENTSO-E Market Prices Data Collection")
    logger.info("=" * 60)
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")
    logger.info(f"Countries: {', '.join(args.countries)}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 60)

    # Fetch data
    results = fetch_multiple_countries(
        args.countries, start_date, end_date, args.output_dir, args.sleep_time
    )

    # Merge all countries
    if len(results) > 1:
        merge_country_prices(results, args.output_dir)

    logger.info("=" * 60)
    logger.info(f"Completed! Fetched data for {len(results)} countries")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
