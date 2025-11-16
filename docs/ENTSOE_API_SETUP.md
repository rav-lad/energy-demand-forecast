# ENTSO-E API Setup Guide

Complete guide to set up and use real market data from the ENTSO-E Transparency Platform.

## Table of Contents

1. [Quick Start](#quick-start)
2. [API Key Setup](#api-key-setup)
3. [Data Collection](#data-collection)
4. [Data Validation](#data-validation)
5. [Using Real Data in Models](#using-real-data-in-models)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Features](#advanced-features)

---

## Quick Start

**5-minute setup to test the API connection:**

```bash
# 1. Copy environment template
cp .env.example .env

# 2. Edit .env and add your API key
# ENTSOE_API_KEY=your_key_here

# 3. Test connection
python test_api_connection.py
```

Expected output:
```
✅ SUCCESS! Fetched 48 hourly price records
```

---

## API Key Setup

### Step 1: Register on ENTSO-E

1. Visit [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/)
2. Click **"Register"** (top-right corner)
3. Fill in registration form:
   - Email address
   - Organization (can be "Individual" or "Research")
   - Country
4. Verify your email

### Step 2: Request API Key

1. Log in to [ENTSO-E Portal](https://transparency.entsoe.eu/)
2. Go to **"My Account Settings"**
3. Click **"Generate API Key"**
4. Copy the generated key (long alphanumeric string)

**Note:** API key activation takes **24-48 hours** after generation.

### Step 3: Configure Environment

```bash
# Copy template
cp .env.example .env

# Edit .env file
nano .env  # or use your preferred editor
```

Add your API key:
```bash
ENTSOE_API_KEY=your_actual_api_key_here
```

### Step 4: Test Connection

```bash
python test_api_connection.py
```

If successful, you'll see:
```
✅ ALL TESTS PASSED - API IS READY TO USE
```

---

## Data Collection

### Collect Historical Price Data

**Basic usage (France, 1 year):**

```bash
python data_recuperation/data_market_prices.py \
  --start_date 2023-01-01 \
  --end_date 2024-12-31 \
  --countries FR
```

**Multiple countries:**

```bash
python data_recuperation/data_market_prices.py \
  --start_date 2023-01-01 \
  --end_date 2024-12-31 \
  --countries FR DE ES IT
```

**Output:**
```
data/raw_data/market_prices/
├── day_ahead_prices_FR.csv
├── day_ahead_prices_DE.csv
├── day_ahead_prices_ES.csv
├── day_ahead_prices_IT.csv
└── day_ahead_prices_all_countries.csv
```

### Data Collection Best Practices

1. **Rate Limiting**: The API has a limit of 400 requests/minute
   - Our connector automatically handles this
   - For large date ranges, collection may take several minutes

2. **Incremental Updates**: Collect small chunks first to test
   ```bash
   # Test with 1 month first
   python data_recuperation/data_market_prices.py \
     --start_date 2024-01-01 \
     --end_date 2024-02-01 \
     --countries FR
   ```

3. **Caching**: Data is automatically cached for 7 days
   - Reduces API calls during development
   - Cache location: `data/cache/entsoe/`
   - Clear cache: `python data_collection/api_cache.py --clear`

---

## Data Validation

### Validate Downloaded Data

```bash
python data_collection/data_validator.py \
  data/raw_data/market_prices/day_ahead_prices_FR.csv \
  --type prices
```

**Example output:**

```
📊 Statistics:
  mean: 67.85 EUR/MWh
  min: -10.50 EUR/MWh
  max: 485.30 EUR/MWh
  negative_prices: 45 (0.52%)

⚠️ Warnings:
  1. Found 45 (0.52%) very low prices (-50.0 to -10.0 EUR/MWh).
     This may be normal during high renewable production.

✅ VALID
```

### What to Look For

**✅ Good signs:**
- Mean price: 40-80 EUR/MWh (typical range)
- No missing values or < 1%
- Negative prices < 5% (normal for renewables)
- Continuous hourly timestamps

**❌ Red flags:**
- Mean price < 20 or > 150 EUR/MWh
- Missing values > 5%
- Large gaps in timestamps
- Prices outside -500 to 3000 EUR/MWh range

---

## Using Real Data in Models

### Update Model Training Scripts

**Before (simulated data):**
```python
from model.price_forecasting.data_loader import prepare_price_forecasting_dataset

# Uses simulated prices by default
df, features = prepare_price_forecasting_dataset()
```

**After (real data):**
```python
from model.price_forecasting.data_loader import load_price_and_load_data

# Load real ENTSO-E prices
df = load_price_and_load_data(
    simulate_prices=False,  # Use real data!
    price_file="data/raw_data/market_prices/day_ahead_prices_FR.csv"
)
```

### Full Training Example

```python
# train_with_real_data.py

import pandas as pd
from model.price_forecasting.data_loader import (
    load_price_and_load_data,
    add_calendar_features,
    add_lag_features
)
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

# 1. Load real price data
df = load_price_and_load_data(
    simulate_prices=False,
    price_file="data/raw_data/market_prices/day_ahead_prices_FR.csv"
)

# 2. Feature engineering
df = add_calendar_features(df)
df = add_lag_features(df, target_col="price")
df = df.dropna()

# 3. Train model
feature_cols = [col for col in df.columns if col not in ["datetime_hour", "price"]]
X = df[feature_cols]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = GradientBoostingRegressor(n_estimators=100)
model.fit(X_train, y_train)

# 4. Evaluate
score = model.score(X_test, y_test)
print(f"R² Score: {score:.3f}")
```

### Expected Performance Change

**Important:** Real data will have **lower performance** than simulated data. This is normal!

| Metric | Simulated Data | Real Data (Expected) |
|--------|----------------|---------------------|
| R² Score | 0.85-0.95 | 0.40-0.70 |
| Sharpe Ratio | 1.5-1.8 | 0.4-1.0 |
| MAE | 5-10 EUR/MWh | 10-20 EUR/MWh |

**Why?** Real markets have:
- Unexpected events (outages, policy changes)
- Non-stationary patterns
- Regime shifts (energy crisis, COVID-19)
- Complex cross-market dependencies

This is documented in the project README and is **expected behavior**.

---

## Troubleshooting

### API Key Issues

**Error:** `ValueError: ENTSO-E API key required`

**Solution:**
1. Check `.env` file exists: `ls -la .env`
2. Check key is set: `cat .env | grep ENTSOE_API_KEY`
3. Make sure no spaces: `ENTSOE_API_KEY=key` not `ENTSOE_API_KEY = key`

---

**Error:** `Invalid API key. Check your ENTSOE_API_KEY.`

**Solution:**
1. Verify key on ENTSO-E website
2. Check if key is activated (wait 24-48h after generation)
3. Try regenerating the key

---

### Rate Limiting

**Warning:** `Rate limit reached (400 req/min). Waiting 45.2s...`

**This is normal!** The connector automatically handles rate limiting.

For large data requests:
- Be patient (may take 10-30 minutes for multi-year data)
- Use `--sleep_time 3` to be more conservative
- Consider using cache for development

---

### Missing Data

**Error:** `Price data file not found: data/raw_data/market_prices/day_ahead_prices_FR.csv`

**Solution:**
```bash
# Collect the data first
python data_recuperation/data_market_prices.py \
  --start_date 2023-01-01 \
  --end_date 2024-12-31 \
  --countries FR

# Verify it was created
ls -lh data/raw_data/market_prices/
```

---

### Data Quality Issues

**Warning:** `Found 150 time gaps in data (missing hours)`

**Possible causes:**
1. DST transitions (expected, 2x per year)
2. API outages during collection
3. Data not available for that period

**Solution:**
- Re-fetch the affected date range
- Use data validation to identify gaps:
  ```bash
  python data_collection/data_validator.py \
    data/raw_data/market_prices/day_ahead_prices_FR.csv
  ```

---

## Advanced Features

### 1. Cache Management

**View cache statistics:**
```bash
python data_collection/api_cache.py --stats
```

**Clear old cache (>30 days):**
```bash
python data_collection/api_cache.py --clear 30
```

**Clear all cache:**
```bash
python data_collection/api_cache.py --clear
```

### 2. Custom Rate Limiting

```python
from data_collection.entsoe_connector import EntsoeConnector

# Slower, more conservative rate limiting
connector = EntsoeConnector(rate_limit_rpm=200)

# Check usage
stats = connector.get_usage_stats()
print(f"Requests: {stats['total_requests']}")
print(f"Utilization: {stats['utilization_percent']:.1f}%")
```

### 3. Load Data (Consumption)

```python
# Fetch actual load data (consumption)
connector = EntsoeConnector()
load_data = connector.get_actual_load("FR", "2024-01-01", "2024-02-01")

# Or get both prices and load together
market_data = connector.get_market_data("FR", "2024-01-01", "2024-02-01")
# Returns: datetime, price_eur_mwh, load_mw, country
```

### 4. Incremental Updates

For daily updates, only fetch new data:

```python
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

def incremental_update():
    """Update price data with only new records."""
    price_file = "data/raw_data/market_prices/day_ahead_prices_FR.csv"

    # Load existing data
    if Path(price_file).exists():
        df_existing = pd.read_csv(price_file, parse_dates=["datetime"])
        last_date = df_existing["datetime"].max()
        start_date = last_date + timedelta(days=1)
    else:
        start_date = datetime(2023, 1, 1)

    end_date = datetime.now()

    # Fetch only new data
    connector = EntsoeConnector()
    df_new = connector.get_day_ahead_prices("FR", start_date, end_date)

    # Append and save
    if Path(price_file).exists():
        df_combined = pd.concat([df_existing, df_new]).drop_duplicates()
    else:
        df_combined = df_new

    df_combined.to_csv(price_file, index=False)
    print(f"Updated with {len(df_new)} new records")
```

### 5. Multi-Country Analysis

```python
# Fetch prices for multiple countries
from data_collection.entsoe_connector import EntsoeConnector

connector = EntsoeConnector()
countries = ["FR", "DE", "ES", "IT"]

all_prices = {}
for country in countries:
    prices = connector.get_day_ahead_prices(country, "2024-01-01", "2024-02-01")
    all_prices[country] = prices

# Analyze price spreads
import pandas as pd
merged = all_prices["FR"][["datetime", "price_eur_mwh"]].rename(
    columns={"price_eur_mwh": "FR"}
)
for country in ["DE", "ES", "IT"]:
    country_prices = all_prices[country][["datetime", "price_eur_mwh"]].rename(
        columns={"price_eur_mwh": country}
    )
    merged = merged.merge(country_prices, on="datetime")

# Calculate spreads
merged["FR_DE_spread"] = merged["FR"] - merged["DE"]
print(merged[["datetime", "FR", "DE", "FR_DE_spread"]].head())
```

---

## Testing

Run the integration tests:

```bash
# Run all tests (requires API key)
pytest tests/test_entsoe_integration.py -v

# Run only unit tests (no API key needed)
pytest tests/test_entsoe_integration.py -v -k "not real_api"

# Run specific test
pytest tests/test_entsoe_integration.py::TestEntsoeConnector::test_rate_limiter_initialization -v
```

---

## Summary Checklist

**Setup:**
- [ ] Register on ENTSO-E Transparency Platform
- [ ] Generate API key (wait 24-48h for activation)
- [ ] Copy `.env.example` to `.env`
- [ ] Add API key to `.env`
- [ ] Test connection: `python test_api_connection.py`

**Data Collection:**
- [ ] Collect historical data: `python data_recuperation/data_market_prices.py`
- [ ] Validate data quality: `python data_collection/data_validator.py`
- [ ] Verify files exist: `ls data/raw_data/market_prices/`

**Model Training:**
- [ ] Update data loader to use `simulate_prices=False`
- [ ] Retrain models with real data
- [ ] Expect lower performance (this is normal!)
- [ ] Document performance changes

---

## Resources

- **ENTSO-E API Documentation**: https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html
- **ENTSO-E Portal**: https://transparency.entsoe.eu/
- **Project README**: ../README.md
- **API Connector Code**: `data_collection/entsoe_connector.py`

---

**Need Help?**

Check the [Troubleshooting](#troubleshooting) section or open an issue on GitHub.
