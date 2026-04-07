"""
Download market data for French Power Spot trading.

This script downloads:
1. French Day-Ahead spot prices (ENTSO-E)
2. German Day-Ahead spot prices (ENTSO-E)
3. Gas prices (TTF proxy via Yahoo Finance)
4. CO2 prices (EUA proxy via Yahoo Finance)

Data Sources:
- ENTSO-E Transparency Platform (free, requires API key)
- Yahoo Finance (commodity proxies)
"""

import sys
from pathlib import Path
from typing import Optional, Dict
import warnings

import pandas as pd

warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ==============================================================================
# CONFIGURATION
# ==============================================================================

START_DATE = "2023-01-01"
END_DATE = "2026-04-06"  # Update daily via cron

OUTPUT_DIR = project_root / "data" / "market_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

import os
ENTSOE_API_KEY = os.getenv("ENTSOE_API_KEY", None)


# ==============================================================================
# 1. ENTSO-E DAY-AHEAD SPOT PRICES
# ==============================================================================

def download_entsoe_prices(
    country_code: str,
    start_date: str,
    end_date: str,
    api_key: Optional[str] = None
) -> Optional[pd.DataFrame]:
    """
    Download Day-Ahead prices from ENTSO-E Transparency Platform.
    Free but requires API key: https://transparency.entsoe.eu/
    """
    print(f"\nDownloading ENTSO-E Day-Ahead prices for {country_code}...")

    if api_key is None:
        print("  WARNING: No ENTSO-E API key provided")
        print("  Get one at: https://transparency.entsoe.eu/")
        print("  Then set environment variable: ENTSOE_API_KEY")
        return None

    try:
        from entsoe import EntsoePandasClient

        client = EntsoePandasClient(api_key=api_key)

        start = pd.Timestamp(start_date, tz='Europe/Paris')
        end = pd.Timestamp(end_date, tz='Europe/Paris')

        print(f"  Fetching {start.date()} to {end.date()}...")
        prices = client.query_day_ahead_prices(country_code, start=start, end=end)

        df = pd.DataFrame({
            'datetime': prices.index,
            'price': prices.values,
            'country': country_code
        })

        df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)

        print(f"  SUCCESS: Downloaded {len(df)} hours of data")
        return df

    except ImportError:
        print("  ERROR: entsoe-py not installed")
        print("  Install with: pip install entsoe-py")
        return None

    except Exception as e:
        print(f"  ERROR: {e}")
        return None


# ==============================================================================
# 2. YAHOO FINANCE COMMODITY PRICES (PROXY)
# ==============================================================================

def download_yahoo_commodity(
    ticker: str,
    start_date: str,
    end_date: str,
    name: str
) -> Optional[pd.DataFrame]:
    """
    Download commodity prices from Yahoo Finance.
    """
    print(f"\nDownloading {name} from Yahoo Finance (ticker: {ticker})...")

    try:
        import yfinance as yf

        data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if len(data) == 0:
            print(f"  WARNING: No data found for {ticker}")
            return None

        df = pd.DataFrame({
            'date': data.index,
            'price': data['Close'].values,
            'commodity': name
        })

        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)

        print(f"  SUCCESS: Downloaded {len(df)} days of data")
        return df

    except ImportError:
        print("  ERROR: yfinance not installed")
        print("  Install with: pip install yfinance")
        return None

    except Exception as e:
        print(f"  ERROR: {e}")
        return None


# ==============================================================================
# 3. MAIN DOWNLOAD ORCHESTRATOR
# ==============================================================================

def download_all_market_data(
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    entsoe_key: Optional[str] = ENTSOE_API_KEY
) -> Dict[str, pd.DataFrame]:
    """
    Download all required market data.
    """
    print("="*80)
    print("DOWNLOADING MARKET DATA FOR POWER SPOT TRADING")
    print("="*80)
    print(f"Date range: {start_date} to {end_date}")
    print(f"Output directory: {OUTPUT_DIR}")

    results = {}

    # -------------------------------------------------------------------------
    # 1. French Day-Ahead Spot Prices (ENTSO-E)
    # -------------------------------------------------------------------------

    fr_spot = download_entsoe_prices('FR', start_date, end_date, entsoe_key)

    if fr_spot is not None:
        fr_spot['date'] = fr_spot['datetime'].dt.date
        fr_spot_daily = fr_spot.groupby('date')['price'].mean().reset_index()
        fr_spot_daily['date'] = pd.to_datetime(fr_spot_daily['date'])

        results['fr_spot_hourly'] = fr_spot
        results['fr_spot_daily'] = fr_spot_daily

        fr_spot.to_csv(OUTPUT_DIR / "fr_spot_hourly.csv", index=False)
        fr_spot_daily.to_csv(OUTPUT_DIR / "fr_spot_daily.csv", index=False)

        print(f"  Saved: fr_spot_hourly.csv ({len(fr_spot)} hours)")
        print(f"  Saved: fr_spot_daily.csv ({len(fr_spot_daily)} days)")
    else:
        print("  WARNING: Using existing data from processed folder")
        processed_file = project_root / "data/processed/price_forecasting_dataset_with_forecasts.csv"
        if processed_file.exists():
            df_processed = pd.read_csv(processed_file)
            df_processed['datetime_hour'] = pd.to_datetime(df_processed['datetime_hour'])
            df_processed['date'] = df_processed['datetime_hour'].dt.date

            fr_spot_daily = df_processed.groupby('date')['price'].mean().reset_index()
            fr_spot_daily['date'] = pd.to_datetime(fr_spot_daily['date'])

            results['fr_spot_daily'] = fr_spot_daily

    # -------------------------------------------------------------------------
    # 2. German Day-Ahead Spot Prices (ENTSO-E) - Optional
    # -------------------------------------------------------------------------

    de_spot = download_entsoe_prices('DE_LU', start_date, end_date, entsoe_key)

    if de_spot is not None:
        de_spot['date'] = de_spot['datetime'].dt.date
        de_spot_daily = de_spot.groupby('date')['price'].mean().reset_index()
        de_spot_daily['date'] = pd.to_datetime(de_spot_daily['date'])

        results['de_spot_hourly'] = de_spot
        results['de_spot_daily'] = de_spot_daily

        de_spot.to_csv(OUTPUT_DIR / "de_spot_hourly.csv", index=False)
        de_spot_daily.to_csv(OUTPUT_DIR / "de_spot_daily.csv", index=False)

        print(f"  Saved: de_spot_hourly.csv ({len(de_spot)} hours)")
        print(f"  Saved: de_spot_daily.csv ({len(de_spot_daily)} days)")

    # -------------------------------------------------------------------------
    # 3. Natural Gas Prices (TTF proxy via Yahoo)
    # -------------------------------------------------------------------------

    gas = download_yahoo_commodity("NG=F", start_date, end_date, "Natural Gas")

    if gas is not None:
        results['gas'] = gas
        gas.to_csv(OUTPUT_DIR / "gas_prices.csv", index=False)
        print(f"  Saved: gas_prices.csv ({len(gas)} days)")
    else:
        print("  WARNING: Gas prices unavailable. TTF data requires a paid feed (e.g. Databento).")

    # -------------------------------------------------------------------------
    # 4. CO2 Prices (EUA proxy via Yahoo)
    # -------------------------------------------------------------------------

    co2 = download_yahoo_commodity("KRBN", start_date, end_date, "CO2 EUA")

    if co2 is not None:
        results['co2'] = co2
        co2.to_csv(OUTPUT_DIR / "co2_prices.csv", index=False)
        print(f"  Saved: co2_prices.csv ({len(co2)} days)")
    else:
        print("  WARNING: CO2 prices unavailable. EUA data requires a paid feed (e.g. EEX).")

    return results


# ==============================================================================
# 4. SUMMARY
# ==============================================================================

def print_data_summary(results: Dict[str, pd.DataFrame]):
    print("\n" + "="*80)
    print("DATA DOWNLOAD SUMMARY")
    print("="*80)

    for name, df in results.items():
        if df is not None and len(df) > 0:
            date_col = 'date' if 'date' in df.columns else 'datetime'
            price_col = 'price' if 'price' in df.columns else df.columns[1]

            print(f"\n{name.upper()}:")
            print(f"  Rows: {len(df):,}")
            print(f"  Date range: {df[date_col].min()} to {df[date_col].max()}")

            if price_col in df.columns:
                print(f"  Price mean: {df[price_col].mean():.2f}")
                print(f"  Price std:  {df[price_col].std():.2f}")
                print(f"  Price min:  {df[price_col].min():.2f}")
                print(f"  Price max:  {df[price_col].max():.2f}")

    print("\n" + "="*80)
    print("FILES SAVED TO:", OUTPUT_DIR)
    print("="*80)


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("MARKET DATA DOWNLOAD SCRIPT")
    print("="*80)

    try:
        import yfinance
        print("yfinance: OK")
    except ImportError:
        print("yfinance: NOT INSTALLED (pip install yfinance)")

    try:
        import entsoe
        print("entsoe-py: OK")
    except ImportError:
        print("entsoe-py: NOT INSTALLED (pip install entsoe-py)")

    if ENTSOE_API_KEY:
        print(f"ENTSO-E API Key: SET (length={len(ENTSOE_API_KEY)})")
    else:
        print("ENTSO-E API Key: NOT SET")
        print("  Get one at: https://transparency.entsoe.eu/")
        print("  Set as: export ENTSOE_API_KEY='your-key-here'")

    results = download_all_market_data(START_DATE, END_DATE, ENTSOE_API_KEY)
    print_data_summary(results)

    print("\n" + "="*80)
    print("DOWNLOAD COMPLETE!")
    print("="*80)
