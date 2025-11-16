"""
Test ENTSO-E API Connection

This script tests the connection to ENTSO-E API and validates your API key.

Usage:
    python test_api_connection.py

Requirements:
    - Set ENTSOE_API_KEY in your .env file
    - Run: cp .env.example .env (if not already done)
    - Add your API key to .env
"""

import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


def check_env_setup():
    """Check if .env file exists and API key is configured."""
    env_path = Path(".env")
    env_example_path = Path(".env.example")

    if not env_path.exists():
        logger.error("❌ .env file not found")
        logger.info("\n📋 Setup instructions:")
        logger.info("1. Copy the example file:")
        logger.info("   cp .env.example .env")
        logger.info("\n2. Get your free API key:")
        logger.info("   - Visit: https://transparency.entsoe.eu/")
        logger.info("   - Register an account")
        logger.info("   - Go to 'My Account Settings'")
        logger.info("   - Generate API key (takes ~24h for approval)")
        logger.info("\n3. Edit .env and add your key:")
        logger.info("   ENTSOE_API_KEY=your_actual_key_here")
        return False

    api_key = os.getenv("ENTSOE_API_KEY")

    if not api_key or api_key == "your_entsoe_api_key_here":
        logger.error("❌ ENTSOE_API_KEY not configured in .env")
        logger.info("\n📋 Please edit .env and replace:")
        logger.info("   ENTSOE_API_KEY=your_entsoe_api_key_here")
        logger.info("with your actual API key from https://transparency.entsoe.eu/")
        return False

    logger.info("✅ .env file found")
    logger.info(f"✅ API key configured (length: {len(api_key)} chars)")
    return True


def test_connector_import():
    """Test if EntsoeConnector can be imported."""
    try:
        from data_collection.entsoe_connector import EntsoeConnector
        logger.info("✅ EntsoeConnector imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ Failed to import EntsoeConnector: {e}")
        logger.info("\n📋 Make sure you're in the project root directory")
        return False


def test_api_connection():
    """Test actual API connection with a small data request."""
    from data_collection.entsoe_connector import EntsoeConnector

    try:
        # Initialize connector
        logger.info("\n🔄 Initializing ENTSO-E connector...")
        connector = EntsoeConnector()
        logger.info("✅ Connector initialized")

        # Test with a small date range (2 days)
        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(days=2)

        logger.info(f"\n🔄 Testing API with small request:")
        logger.info(f"   Country: FR (France)")
        logger.info(f"   Period: {start_date.date()} to {end_date.date()}")

        # Fetch day-ahead prices
        prices = connector.get_day_ahead_prices(
            country="FR",
            start=start_date,
            end=end_date
        )

        if prices is not None and len(prices) > 0:
            logger.info(f"\n✅ SUCCESS! Fetched {len(prices)} hourly price records")
            logger.info("\n📊 Sample data (first 5 records):")
            print(prices.head())

            logger.info("\n📊 Basic statistics:")
            logger.info(f"   Mean price: {prices['price_eur_mwh'].mean():.2f} EUR/MWh")
            logger.info(f"   Min price:  {prices['price_eur_mwh'].min():.2f} EUR/MWh")
            logger.info(f"   Max price:  {prices['price_eur_mwh'].max():.2f} EUR/MWh")

            logger.info("\n🎉 Your API connection is working correctly!")
            logger.info("\n📋 Next steps:")
            logger.info("1. Collect historical data:")
            logger.info("   python data_recuperation/data_market_prices.py --start_date 2023-01-01 --end_date 2024-12-31 --countries FR")
            logger.info("\n2. Check collected data:")
            logger.info("   ls -lh data/raw_data/market_prices/")

            return True
        else:
            logger.error("❌ API returned empty data")
            return False

    except ValueError as e:
        logger.error(f"❌ Configuration error: {e}")
        return False
    except Exception as e:
        logger.error(f"❌ API request failed: {e}")
        logger.info("\n📋 Common issues:")
        logger.info("1. Invalid API key - check your key on https://transparency.entsoe.eu/")
        logger.info("2. API key not yet activated (takes ~24h after registration)")
        logger.info("3. Network connectivity issues")
        logger.info("4. ENTSO-E service temporarily unavailable")
        return False


def main():
    """Run all connection tests."""
    print("=" * 70)
    print("ENTSO-E API CONNECTION TEST")
    print("=" * 70)

    # Test 1: Environment setup
    logger.info("\n[1/3] Checking environment setup...")
    if not check_env_setup():
        sys.exit(1)

    # Test 2: Import connector
    logger.info("\n[2/3] Testing connector import...")
    if not test_connector_import():
        sys.exit(1)

    # Test 3: API connection
    logger.info("\n[3/3] Testing API connection...")
    if not test_api_connection():
        sys.exit(1)

    print("\n" + "=" * 70)
    print("✅ ALL TESTS PASSED - API IS READY TO USE")
    print("=" * 70)


if __name__ == "__main__":
    main()
