# Migration Guide: Simulated Data → Real ENTSO-E Data

This guide helps you transition from using simulated electricity prices to real market data from ENTSO-E.

## Why Migrate?

**Simulated data is great for:**
- ✅ Testing and development
- ✅ Understanding the codebase
- ✅ Experimenting with features
- ✅ Quick prototyping

**Real data is necessary for:**
- ✅ Production deployment
- ✅ Actual trading decisions
- ✅ Research publications
- ✅ Real-world performance validation

## Migration Checklist

### Phase 1: Setup (30 minutes)

- [ ] **Step 1.1**: Get ENTSO-E API key
  ```bash
  # Visit https://transparency.entsoe.eu/
  # Register → My Account → Generate API Key
  # Wait 24-48h for activation
  ```

- [ ] **Step 1.2**: Configure environment
  ```bash
  cp .env.example .env
  # Edit .env and add: ENTSOE_API_KEY=your_key_here
  ```

- [ ] **Step 1.3**: Test connection
  ```bash
  python test_api_connection.py
  # Expected: ✅ ALL TESTS PASSED
  ```

### Phase 2: Data Collection (1-2 hours)

- [ ] **Step 2.1**: Collect small test dataset
  ```bash
  # Start with 1 month to test
  python data_recuperation/data_market_prices.py \
    --start_date 2024-01-01 \
    --end_date 2024-02-01 \
    --countries FR
  ```

- [ ] **Step 2.2**: Validate test data
  ```bash
  python data_collection/data_validator.py \
    data/raw_data/market_prices/day_ahead_prices_FR.csv \
    --type prices

  # Expected: ✅ VALID (or minor warnings only)
  ```

- [ ] **Step 2.3**: Collect full historical dataset
  ```bash
  # Collect 2+ years for robust training
  python data_recuperation/data_market_prices.py \
    --start_date 2022-01-01 \
    --end_date 2024-12-31 \
    --countries FR

  # This will take 30-60 minutes due to rate limiting
  ```

- [ ] **Step 2.4**: Verify data completeness
  ```bash
  # Check file size (should be ~1-2 MB for 1 year)
  ls -lh data/raw_data/market_prices/day_ahead_prices_FR.csv

  # Check record count
  wc -l data/raw_data/market_prices/day_ahead_prices_FR.csv
  # Expected: ~26,300 lines for 3 years (24h * 365d * 3y)
  ```

### Phase 3: Code Migration (30 minutes)

**Before (using simulated data):**

```python
# model/price_forecasting/train_model.py

from model.price_forecasting.data_loader import prepare_price_forecasting_dataset

# Loads simulated prices by default
df, features = prepare_price_forecasting_dataset()
```

**After (using real data):**

```python
# model/price_forecasting/train_model.py

from model.price_forecasting.data_loader import load_price_and_load_data, add_calendar_features, add_lag_features

# Load real ENTSO-E prices
df = load_price_and_load_data(
    simulate_prices=False,  # ← KEY CHANGE!
    price_file="data/raw_data/market_prices/day_ahead_prices_FR.csv"
)

# Add features as before
df = add_calendar_features(df)
df = add_lag_features(df, target_col="price")
df = df.dropna()

# Continue with model training...
```

**Files to update:**

- [ ] `model/price_forecasting/train_model.py` (or your training script)
- [ ] `model/xgboost/train_xgboost.py`
- [ ] `trading_system/backtests/run_backtest.py`
- [ ] Any custom training scripts

### Phase 4: Model Retraining (2-4 hours)

- [ ] **Step 4.1**: Retrain models with real data
  ```bash
  # Example: retrain XGBoost model
  python model/xgboost/train_xgboost.py --frequency daily
  ```

- [ ] **Step 4.2**: Run backtests
  ```bash
  make benchmark  # or your backtest command
  ```

- [ ] **Step 4.3**: Compare performance

  **Expected changes:**

  | Metric | Simulated | Real (Expected) | Status |
  |--------|-----------|-----------------|--------|
  | R² Score | 0.85-0.95 | 0.40-0.70 | ⚠️ Lower |
  | MAE | 5-10 EUR/MWh | 10-20 EUR/MWh | ⚠️ Higher |
  | Sharpe Ratio | 1.5-1.8 | 0.4-1.0 | ⚠️ Lower |

  **This is NORMAL and expected!** Real markets are harder to predict.

- [ ] **Step 4.4**: Document performance
  ```bash
  # Save backtest results
  cp trading_system/backtests/results.csv \
     trading_system/backtests/results_real_data_$(date +%Y%m%d).csv
  ```

### Phase 5: Testing & Validation (1 hour)

- [ ] **Step 5.1**: Run unit tests
  ```bash
  pytest tests/test_entsoe_integration.py -v -k "not real_api"
  ```

- [ ] **Step 5.2**: Run integration tests (with API key)
  ```bash
  pytest tests/test_entsoe_integration.py -v
  ```

- [ ] **Step 5.3**: Validate model outputs
  ```bash
  # Check predictions are in reasonable range
  python -c "
  import pandas as pd
  import joblib

  # Load model
  model = joblib.load('models/price_forecast/xgboost_daily.pkl')

  # Make predictions on test set
  # ... (your prediction code)

  # Check range
  print(f'Predictions: {predictions.min():.2f} to {predictions.max():.2f} EUR/MWh')
  # Should be roughly 0-500 EUR/MWh for normal conditions
  "
  ```

- [ ] **Step 5.4**: Sanity checks
  - [ ] Model predictions are non-negative (or only slightly negative)
  - [ ] Predictions follow daily patterns (peaks during day, lows at night)
  - [ ] Predictions respond to load changes
  - [ ] No extreme outliers (> 1000 EUR/MWh) unless justified

### Phase 6: Production Deployment

- [ ] **Step 6.1**: Update README/docs with real data usage
- [ ] **Step 6.2**: Set up automated data updates
  ```bash
  # Add to crontab for daily updates
  0 2 * * * cd /path/to/project && python data_recuperation/data_market_prices.py --incremental
  ```

- [ ] **Step 6.3**: Monitor data quality
  ```bash
  # Add validation to daily workflow
  python data_collection/data_validator.py \
    data/raw_data/market_prices/day_ahead_prices_FR.csv
  ```

- [ ] **Step 6.4**: Set up alerts for data issues
  - Missing data
  - Stale data (no updates in 48h)
  - Validation failures

---

## Common Migration Issues

### Issue 1: Performance Drop

**Symptom:**
```
Simulated data R² = 0.92
Real data R² = 0.55  ← "Model got worse!"
```

**This is EXPECTED and NORMAL.**

**Why?**
- Simulated data has clean, predictable patterns
- Real markets have:
  - Unexpected events (plant outages, grid failures)
  - Policy changes (carbon taxes, subsidies)
  - Weather surprises
  - Geopolitical events (energy crisis, trade disruptions)

**Action:**
✅ Document the new baseline
✅ Focus on improving from this baseline
❌ Don't try to match simulated performance

---

### Issue 2: Missing Data Points

**Symptom:**
```
⚠️ Found 150 time gaps in data
```

**Causes:**
1. **Daylight Saving Time** (2x per year): Normal, expected
2. **API outages during collection**: Re-fetch that period
3. **Data not available**: Check ENTSO-E website for that date

**Solution:**
```bash
# Re-fetch specific problematic period
python data_recuperation/data_market_prices.py \
  --start_date 2024-03-15 \
  --end_date 2024-03-20 \
  --countries FR
```

---

### Issue 3: Extreme Price Values

**Symptom:**
```
⚠️ Found prices above 500 EUR/MWh
Max price: 1,850 EUR/MWh on 2022-08-25
```

**This can be NORMAL.**

Real electricity prices can spike to very high values during:
- Heat waves (high AC demand)
- Cold snaps (heating demand)
- Renewable droughts (low wind/solar)
- Plant outages
- Energy crises (2022 gas shortage)

**Action:**
1. **Verify it's real:** Check [ENTSO-E website](https://transparency.entsoe.eu/) for that date
2. **Keep the data:** These events are important for robust models
3. **Consider outlier handling:** Cap predictions at reasonable max (e.g., 500 EUR/MWh) for trading strategies

---

### Issue 4: Negative Prices

**Symptom:**
```
Found 450 (5.2%) negative prices
Min price: -45.30 EUR/MWh
```

**This is NORMAL.**

Negative prices occur when:
- High renewable production (wind/solar)
- Low demand (weekend nights, holidays)
- Must-run generation (nuclear can't ramp down quickly)
- Grid constraints

**Action:**
✅ Keep negative prices in training data
✅ Ensure models can predict negative values
✅ Trading strategies should exploit negative price periods

---

### Issue 5: Model Divergence

**Symptom:**
```
Training on simulated data: Great
Training on real data: Model diverges / predictions are constant
```

**Causes:**
- Different data scale (real prices more volatile)
- Different feature distributions
- Outliers affecting gradient descent

**Solution:**
```python
from sklearn.preprocessing import RobustScaler

# Use robust scaling to handle outliers
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Or: Remove extreme outliers for training (but keep for evaluation)
q99 = df['price'].quantile(0.99)
q01 = df['price'].quantile(0.01)
df_train = df[(df['price'] >= q01) & (df['price'] <= q99)]
```

---

## Rollback Plan

If real data causes critical issues:

```bash
# 1. Revert code changes
git checkout HEAD -- model/price_forecasting/data_loader.py

# 2. Return to simulated data
# In your training scripts, set:
# simulate_prices=True

# 3. Retrain models
python model/xgboost/train_xgboost.py
```

You can always retry real data migration later after debugging.

---

## Success Criteria

You've successfully migrated when:

- ✅ Models train without errors on real data
- ✅ Predictions are in reasonable range (0-500 EUR/MWh typically)
- ✅ Backtests complete successfully
- ✅ Performance metrics are documented
- ✅ Data quality validation passes
- ✅ Daily data updates are automated

**Expected timeline:** 1-2 days for full migration and validation.

---

## Next Steps After Migration

1. **Feature Engineering**: Add real-world features
   - Weather forecasts (temperature, wind, solar radiation)
   - Fuel prices (gas, coal, carbon)
   - Grid fundamentals (generation by type, cross-border flows)

2. **Model Improvements**:
   - Ensemble methods
   - Time-varying parameters
   - Regime-switching models (normal vs. crisis periods)

3. **Production Monitoring**:
   - Track prediction accuracy over time
   - Detect model drift
   - Automate retraining

4. **Multi-Market Analysis**:
   - Cross-border price spreads (FR-DE, FR-ES)
   - Arbitrage opportunities

---

## Resources

- **Setup Guide**: [ENTSOE_API_SETUP.md](./ENTSOE_API_SETUP.md)
- **API Documentation**: https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html
- **Troubleshooting**: See ENTSOE_API_SETUP.md § Troubleshooting

---

**Questions?** Check the troubleshooting section or open an issue on GitHub.
