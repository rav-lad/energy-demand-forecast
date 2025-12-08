"""
Download real market data for French Power Futures trading.

This script downloads:
1. French Power Futures (FNB) from available sources
2. German Power Futures (GNB) for spread trading
3. Gas prices (TTF) for spark spread
4. CO2 prices (EUA) for spark spread
5. Day-Ahead spot prices for validation

Data Sources (by priority):
- ENTSO-E Transparency Platform (free, requires API key)
- Yahoo Finance (for some commodity proxies)
- Alternative: Constructed from fundamentals
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict
import warnings

import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Date range for download
START_DATE = "2023-01-01"
END_DATE = "2024-12-31"

# Output directory
OUTPUT_DIR = project_root / "data" / "market_data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ENTSO-E API key (set as environment variable or hardcode for testing)
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

    Note: Not all energy commodities available, but some proxies exist.
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
# 3. CONSTRUCT FUTURES FROM FUNDAMENTALS (FALLBACK)
# ==============================================================================

def construct_power_futures_from_spot(
    spot_df: pd.DataFrame,
    date_col: str = 'date',
    price_col: str = 'price',
    forward_premium: float = 2.0,
    basis_volatility: float = 3.0
) -> pd.DataFrame:
    """
    Construct realistic futures prices from spot fundamentals.

    This is a fallback when real futures data is not available.
    Based on academic literature (Lucia & Schwartz 2002, Weron 2014).
    """
    print(f"\nConstructing power futures from spot fundamentals...")

    df = spot_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col)

    # Forward-looking expectation (rolling mean)
    window = 21
    df['spot_forward_ma'] = df[price_col].rolling(window, min_periods=1).mean().shift(-window)
    df['spot_forward_ma'] = df['spot_forward_ma'].fillna(method='bfill')

    # Seasonal risk premium
    df['month'] = pd.to_datetime(df[date_col]).dt.month
    seasonal_premium = df['month'].apply(lambda m:
        3.0 if m in [12, 1, 2] else  # Winter
        -1.0 if m in [6, 7, 8] else  # Summer
        1.0  # Spring/Fall
    )

    # Volatility risk premium
    spot_vol = df[price_col].rolling(21, min_periods=5).std()
    vol_premium = (spot_vol / spot_vol.mean() - 1.0) * 2.0

    # Construct futures
    df['futures_front'] = (
        df['spot_forward_ma'] +
        forward_premium +
        seasonal_premium +
        vol_premium
    )

    # Add mean-reverting basis noise
    np.random.seed(42)
    basis_noise = np.random.normal(0, basis_volatility, len(df))

    basis = np.zeros(len(df))
    basis[0] = basis_noise[0]
    mean_reversion_speed = 0.05

    for i in range(1, len(df)):
        basis[i] = basis[i-1] * (1 - mean_reversion_speed) + basis_noise[i]

    df['futures_front'] = df['futures_front'] + basis
    df['futures_front'] = df['futures_front'].clip(lower=0)

    print(f"  SUCCESS: Constructed {len(df)} days of futures data")

    return df[[date_col, price_col, 'futures_front']]


# ==============================================================================
# 4. MAIN DOWNLOAD ORCHESTRATOR
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
    print("DOWNLOADING MARKET DATA FOR POWER FUTURES TRADING")
    print("="*80)
    print(f"Date range: {start_date} to {end_date}")
    print(f"Output directory: {OUTPUT_DIR}")

    results = {}

    # -------------------------------------------------------------------------
    # 1. French Day-Ahead Spot Prices (ENTSO-E)
    # -------------------------------------------------------------------------

    fr_spot = download_entsoe_prices('FR', start_date, end_date, entsoe_key)

    if fr_spot is not None:
        # Aggregate to daily
        fr_spot['date'] = fr_spot['datetime'].dt.date
        fr_spot_daily = fr_spot.groupby('date')['price'].mean().reset_index()
        fr_spot_daily['date'] = pd.to_datetime(fr_spot_daily['date'])

        results['fr_spot_hourly'] = fr_spot
        results['fr_spot_daily'] = fr_spot_daily

        # Save
        fr_spot.to_csv(OUTPUT_DIR / "fr_spot_hourly.csv", index=False)
        fr_spot_daily.to_csv(OUTPUT_DIR / "fr_spot_daily.csv", index=False)

        print(f"  Saved: fr_spot_hourly.csv ({len(fr_spot)} hours)")
        print(f"  Saved: fr_spot_daily.csv ({len(fr_spot_daily)} days)")
    else:
        print("  WARNING: Using existing data from processed folder")
        # Load from our processed data
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

    # Try natural gas ETF as proxy (not perfect but available)
    gas = download_yahoo_commodity("NG=F", start_date, end_date, "Natural Gas")

    if gas is not None:
        results['gas'] = gas
        gas.to_csv(OUTPUT_DIR / "gas_prices.csv", index=False)
        print(f"  Saved: gas_prices.csv ({len(gas)} days)")
    else:
        # Simulate realistic gas prices
        print("  Constructing simulated gas prices...")
        if 'fr_spot_daily' in results:
            dates = results['fr_spot_daily']['date']
            np.random.seed(42)
            gas_base = 35  # EUR/MWh
            gas_prices = gas_base + np.random.normal(0, 8, len(dates))

            gas = pd.DataFrame({
                'date': dates,
                'price': gas_prices,
                'commodity': 'Natural Gas (simulated)'
            })
            results['gas'] = gas
            gas.to_csv(OUTPUT_DIR / "gas_prices.csv", index=False)

    # -------------------------------------------------------------------------
    # 4. CO2 Prices (EUA proxy via Yahoo)
    # -------------------------------------------------------------------------

    # Try carbon credits ETF as proxy
    co2 = download_yahoo_commodity("KRBN", start_date, end_date, "CO2 EUA")

    if co2 is not None:
        results['co2'] = co2
        co2.to_csv(OUTPUT_DIR / "co2_prices.csv", index=False)
        print(f"  Saved: co2_prices.csv ({len(co2)} days)")
    else:
        # Simulate realistic CO2 prices
        print("  Constructing simulated CO2 prices...")
        if 'fr_spot_daily' in results:
            dates = results['fr_spot_daily']['date']
            np.random.seed(43)
            co2_base = 80  # EUR/tonne
            co2_prices = co2_base + np.random.normal(0, 10, len(dates))

            co2 = pd.DataFrame({
                'date': dates,
                'price': co2_prices,
                'commodity': 'CO2 EUA (simulated)'
            })
            results['co2'] = co2
            co2.to_csv(OUTPUT_DIR / "co2_prices.csv", index=False)

    # -------------------------------------------------------------------------
    # 5. French Power Futures (FNB) - Construct from fundamentals
    # -------------------------------------------------------------------------

    if 'fr_spot_daily' in results:
        print("\nConstructing French Power Futures (FNB)...")

        fr_futures = construct_power_futures_from_spot(
            results['fr_spot_daily'],
            date_col='date',
            price_col='price',
            forward_premium=2.0,
            basis_volatility=3.0
        )

        results['fr_futures'] = fr_futures
        fr_futures.to_csv(OUTPUT_DIR / "fr_power_futures.csv", index=False)
        print(f"  Saved: fr_power_futures.csv ({len(fr_futures)} days)")

        # Also save to processed folder for notebook
        processed_out = project_root / "data/processed/french_power_futures.csv"
        fr_futures.to_csv(processed_out, index=False)

    # -------------------------------------------------------------------------
    # 6. German Power Futures (GNB) - Construct if we have DE spot
    # -------------------------------------------------------------------------

    if 'de_spot_daily' in results:
        print("\nConstructing German Power Futures (GNB)...")

        de_futures = construct_power_futures_from_spot(
            results['de_spot_daily'],
            date_col='date',
            price_col='price',
            forward_premium=1.5,
            basis_volatility=3.5
        )

        results['de_futures'] = de_futures
        de_futures.to_csv(OUTPUT_DIR / "de_power_futures.csv", index=False)
        print(f"  Saved: de_power_futures.csv ({len(de_futures)} days)")

    return results


# ==============================================================================
# 5. SUMMARY & VALIDATION
# ==============================================================================

def print_data_summary(results: Dict[str, pd.DataFrame]):
    """
    Print summary of downloaded data.
    """
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

    # Check for dependencies
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
        print("  Or: os.environ['ENTSOE_API_KEY'] = 'your-key-here'")

    # Download all data
    results = download_all_market_data(START_DATE, END_DATE, ENTSOE_API_KEY)

    # Print summary
    print_data_summary(results)

    print("\n" + "="*80)
    print("DOWNLOAD COMPLETE!")
    print("="*80)
    print("\nNext steps:")
    print("1. Review downloaded data in:", OUTPUT_DIR)
    print("2. Run trading notebook: research/notebooks/06_trading_strategies.ipynb")
    print("3. For real futures data, consider:")
    print("   - Databento API (https://databento.com/)")
    print("   - EEX Market Data (https://www.eex.com/en/market-data)")
    print("   - ICE Data Services (https://www.ice.com/)")
