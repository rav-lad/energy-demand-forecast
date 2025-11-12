"""
Fundamental Energy Data Collection from ENTSO-E Transparency Platform

This script collects fundamental energy market data:
- Generation by production type (nuclear, solar, wind, gas, etc.)
- Actual load vs forecasted load
- Cross-border flows (imports/exports)
- Installed generation capacity

Requirements:
    - Free API key from https://transparency.entsoe.eu/
    - Set ENTSOE_API_KEY in your .env file

Usage:
    python data_fundamentals.py --start_date 2020-01-01 --end_date 2024-12-31 --country FR --data_type generation
"""

import os
import sys
import argparse
from datetime import datetime
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
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
COUNTRIES_MAPPING = {
    'FR': '10YFR-RTE------C',
    'DE': '10Y1001A1001A83F',
    'ES': '10YES-REE------0',
    'IT': '10YIT-GRTN-----B',
    'BE': '10YBE----------2',
    'NL': '10YNL----------L',
    'CH': '10YCH-SWISSGRIDZ',
    'AT': '10YAT-APG------L',
    'GB': '10YGB----------A',
}

# Production types mapping
PRODUCTION_TYPES = {
    'B01': 'Biomass',
    'B02': 'Fossil Brown coal/Lignite',
    'B03': 'Fossil Coal-derived gas',
    'B04': 'Fossil Gas',
    'B05': 'Fossil Hard coal',
    'B06': 'Fossil Oil',
    'B07': 'Fossil Oil shale',
    'B08': 'Fossil Peat',
    'B09': 'Geothermal',
    'B10': 'Hydro Pumped Storage',
    'B11': 'Hydro Run-of-river and poundage',
    'B12': 'Hydro Water Reservoir',
    'B13': 'Marine',
    'B14': 'Nuclear',
    'B15': 'Other renewable',
    'B16': 'Solar',
    'B17': 'Waste',
    'B18': 'Wind Offshore',
    'B19': 'Wind Onshore',
    'B20': 'Other',
}


def get_entsoe_client():
    """Initialize ENTSO-E API client."""
    api_key = os.getenv('ENTSOE_API_KEY')
    if not api_key:
        raise ValueError(
            "ENTSOE_API_KEY not found. Please set it in your .env file.\n"
            "Get your free API key from https://transparency.entsoe.eu/"
        )
    return EntsoePandasClient(api_key=api_key)


def fetch_generation_by_type(client, country_code, start_date, end_date):
    """
    Fetch actual generation by production type.

    Args:
        client: EntsoePandasClient instance
        country_code: Two-letter country code
        start_date: Start date
        end_date: End date

    Returns:
        pd.DataFrame: Generation by type
    """
    country_domain = COUNTRIES_MAPPING.get(country_code)
    if not country_domain:
        raise ValueError(f"Unsupported country: {country_code}")

    if isinstance(start_date, str):
        start_date = pd.Timestamp(start_date, tz='Europe/Paris')
    if isinstance(end_date, str):
        end_date = pd.Timestamp(end_date, tz='Europe/Paris')

    logger.info(f"Fetching generation by type for {country_code}")

    try:
        # Fetch generation data
        generation = client.query_generation(
            country_domain,
            start=start_date,
            end=end_date,
            psr_type=None  # Get all types
        )

        if isinstance(generation, pd.DataFrame):
            logger.info(f"Fetched generation data: {generation.shape}")
            return generation
        else:
            logger.warning("No generation data returned")
            return None

    except Exception as e:
        logger.error(f"Error fetching generation data: {str(e)}")
        return None


def fetch_load_data(client, country_code, start_date, end_date):
    """
    Fetch actual load and day-ahead load forecast.

    Args:
        client: EntsoePandasClient instance
        country_code: Two-letter country code
        start_date: Start date
        end_date: End date

    Returns:
        pd.DataFrame: Actual and forecasted load
    """
    country_domain = COUNTRIES_MAPPING.get(country_code)
    if not country_domain:
        raise ValueError(f"Unsupported country: {country_code}")

    if isinstance(start_date, str):
        start_date = pd.Timestamp(start_date, tz='Europe/Paris')
    if isinstance(end_date, str):
        end_date = pd.Timestamp(end_date, tz='Europe/Paris')

    logger.info(f"Fetching load data for {country_code}")

    results = {}

    try:
        # Fetch actual load
        actual_load = client.query_load(country_domain, start=start_date, end=end_date)
        results['actual_load'] = actual_load
        logger.info(f"Fetched actual load: {len(actual_load)} records")
    except Exception as e:
        logger.error(f"Error fetching actual load: {str(e)}")

    try:
        # Fetch day-ahead load forecast
        forecast_load = client.query_load_forecast(country_domain, start=start_date, end=end_date)
        results['forecast_load'] = forecast_load
        logger.info(f"Fetched forecast load: {len(forecast_load)} records")
    except Exception as e:
        logger.error(f"Error fetching forecast load: {str(e)}")

    if not results:
        return None

    # Combine into single DataFrame
    df = pd.DataFrame(results)
    return df


def fetch_cross_border_flows(client, country_from, country_to, start_date, end_date):
    """
    Fetch cross-border electricity flows between two countries.

    Args:
        client: EntsoePandasClient instance
        country_from: Source country code
        country_to: Destination country code
        start_date: Start date
        end_date: End date

    Returns:
        pd.DataFrame: Cross-border flows
    """
    domain_from = COUNTRIES_MAPPING.get(country_from)
    domain_to = COUNTRIES_MAPPING.get(country_to)

    if not domain_from or not domain_to:
        raise ValueError(f"Unsupported countries: {country_from} or {country_to}")

    if isinstance(start_date, str):
        start_date = pd.Timestamp(start_date, tz='Europe/Paris')
    if isinstance(end_date, str):
        end_date = pd.Timestamp(end_date, tz='Europe/Paris')

    logger.info(f"Fetching cross-border flows: {country_from} -> {country_to}")

    try:
        flows = client.query_crossborder_flows(
            domain_from,
            domain_to,
            start=start_date,
            end=end_date
        )

        if isinstance(flows, pd.Series):
            df = pd.DataFrame(flows, columns=[f'flow_{country_from}_{country_to}'])
            df['country_from'] = country_from
            df['country_to'] = country_to
            logger.info(f"Fetched {len(df)} flow records")
            return df
        else:
            return flows

    except Exception as e:
        logger.error(f"Error fetching cross-border flows: {str(e)}")
        return None


def fetch_installed_capacity(client, country_code, start_date, end_date):
    """
    Fetch installed generation capacity by production type.

    Args:
        client: EntsoePandasClient instance
        country_code: Two-letter country code
        start_date: Start date
        end_date: End date

    Returns:
        pd.DataFrame: Installed capacity by type
    """
    country_domain = COUNTRIES_MAPPING.get(country_code)
    if not country_domain:
        raise ValueError(f"Unsupported country: {country_code}")

    if isinstance(start_date, str):
        start_date = pd.Timestamp(start_date, tz='Europe/Paris')
    if isinstance(end_date, str):
        end_date = pd.Timestamp(end_date, tz='Europe/Paris')

    logger.info(f"Fetching installed capacity for {country_code}")

    try:
        capacity = client.query_installed_generation_capacity(
            country_domain,
            start=start_date,
            end=end_date
        )

        logger.info(f"Fetched installed capacity data")
        return capacity

    except Exception as e:
        logger.error(f"Error fetching installed capacity: {str(e)}")
        return None


def calculate_renewable_share(generation_df):
    """
    Calculate renewable energy share from generation data.

    Args:
        generation_df: DataFrame with generation by type

    Returns:
        pd.DataFrame: Renewable share statistics
    """
    if generation_df is None or generation_df.empty:
        return None

    # Identify renewable columns
    renewable_types = ['Solar', 'Wind Onshore', 'Wind Offshore', 'Hydro Run-of-river and poundage',
                       'Hydro Water Reservoir', 'Biomass', 'Geothermal']

    renewable_cols = [col for col in generation_df.columns
                     if any(r_type.lower() in col.lower() for r_type in renewable_types)]

    if not renewable_cols:
        logger.warning("No renewable generation columns found")
        return None

    # Calculate totals
    df = pd.DataFrame()
    df['total_generation'] = generation_df.sum(axis=1)
    df['renewable_generation'] = generation_df[renewable_cols].sum(axis=1)
    df['renewable_share'] = df['renewable_generation'] / df['total_generation'] * 100

    logger.info(f"Average renewable share: {df['renewable_share'].mean():.2f}%")

    return df


def save_data(df, filename, output_dir):
    """Save DataFrame to CSV."""
    if df is None or df.empty:
        logger.warning(f"No data to save for {filename}")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / filename
    df.to_csv(output_file)
    logger.info(f"Saved data to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Fetch fundamental energy data from ENTSO-E',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch generation data for France
  python data_fundamentals.py --country FR --data_type generation

  # Fetch load data
  python data_fundamentals.py --country FR --data_type load

  # Fetch cross-border flows
  python data_fundamentals.py --data_type cross_border --country_from FR --country_to DE

Data types: generation, load, cross_border, capacity, all
        """
    )

    parser.add_argument('--start_date', type=str, default='2020-01-01')
    parser.add_argument('--end_date', type=str, default=datetime.now().strftime('%Y-%m-%d'))
    parser.add_argument('--country', type=str, default='FR')
    parser.add_argument('--data_type', type=str, default='all',
                       choices=['generation', 'load', 'cross_border', 'capacity', 'all'])
    parser.add_argument('--country_from', type=str, help='For cross-border flows')
    parser.add_argument('--country_to', type=str, help='For cross-border flows')
    parser.add_argument('--output_dir', type=str, default='data/raw_data/fundamentals')

    args = parser.parse_args()

    client = get_entsoe_client()

    logger.info("="*60)
    logger.info("ENTSO-E Fundamental Data Collection")
    logger.info("="*60)

    if args.data_type in ['generation', 'all']:
        df = fetch_generation_by_type(client, args.country, args.start_date, args.end_date)
        save_data(df, f'generation_{args.country}.csv', args.output_dir)

        # Calculate renewable share
        renewable_df = calculate_renewable_share(df)
        save_data(renewable_df, f'renewable_share_{args.country}.csv', args.output_dir)

    if args.data_type in ['load', 'all']:
        df = fetch_load_data(client, args.country, args.start_date, args.end_date)
        save_data(df, f'load_{args.country}.csv', args.output_dir)

    if args.data_type in ['capacity', 'all']:
        df = fetch_installed_capacity(client, args.country, args.start_date, args.end_date)
        save_data(df, f'capacity_{args.country}.csv', args.output_dir)

    if args.data_type == 'cross_border':
        if not args.country_from or not args.country_to:
            logger.error("For cross_border, specify --country_from and --country_to")
            sys.exit(1)

        df = fetch_cross_border_flows(
            client, args.country_from, args.country_to,
            args.start_date, args.end_date
        )
        save_data(df, f'flow_{args.country_from}_{args.country_to}.csv', args.output_dir)

    logger.info("="*60)
    logger.info("Data collection completed!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
