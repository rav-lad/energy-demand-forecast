# Research Notebooks

Comprehensive analysis notebooks for the Energy Price Forecasting project.

## 📚 Notebook Organization

### ⭐ Production-Ready Notebook (Use This!)

**[05_model_performance_analysis.ipynb](05_model_performance_analysis.ipynb)** ⭐ **RECOMMENDED**
- **Verified model performance** with real results
- ML metrics (R², MAPE, MAE, RMSE)
- Trading performance (returns, Sharpe, drawdown)
- Comprehensive visualizations
- Uses current dataset (560 train / 140 test)
- **Best for final results presentation**
- ✅ **Compatible with current data structure**

### 📊 Exploratory Notebooks (Reference Only)

**⚠️ Note:** Notebooks 01-04 were created for an older dataset structure with regional consumption data (ODRE regional breakdown). The current production pipeline uses aggregated national price/load data with weather features. These are kept for reference but **may need adaptation** to work with current data structure.

1. **[01_comprehensive_eda.ipynb](01_comprehensive_eda.ipynb)** - Exploratory Data Analysis ⚠️
   - Statistical tests (stationarity, normality)
   - Seasonality decomposition (STL)
   - Distribution analysis (Q-Q plots, outliers)
   - Correlation analysis (weather vs demand)
   - Regional heterogeneity analysis
   - ⚠️ **Uses old dataset structure (needs `date`, `insee_region`, `conso_elec_mw` columns)**

2. **[02_feature_engineering_analysis.ipynb](02_feature_engineering_analysis.ipynb)** ⚠️
   - Feature engineering methodology
   - SHAP value analysis
   - Feature importance rankings
   - Correlation matrices
   - Lag feature optimization
   - ⚠️ **May need updates for current feature names**

3. **[03_model_benchmarking.ipynb](03_model_benchmarking.ipynb)** ⚠️
   - Baseline models (Ridge, Linear)
   - Tree-based models (RF, XGBoost, LightGBM)
   - Cross-validation results
   - Hyperparameter tuning
   - ⚠️ **May need updates**

4. **[04_price_demand_dynamics.ipynb](04_price_demand_dynamics.ipynb)** ⚠️
   - Price-demand relationship analysis
   - Load-price correlation
   - Spread dynamics
   - Market regime detection
   - ⚠️ **May need updates**

## 🎯 Quick Start

### For Model Results (RECOMMENDED)
**Jump directly to:** `05_model_performance_analysis.ipynb`

This notebook uses verified results from actual model runs and is fully compatible with the current data structure.

### For Historical Reference
Notebooks 01-04 provide valuable exploratory analysis but were created for an older data format. Use them for methodology reference.

## 📊 Datasets Used

All notebooks use data from:
- **Train:** `data/modified_data/train_daily.csv` (560 days)
- **Test:** `data/modified_data/test_daily.csv` (140 days)
- **Model Predictions:** `models/{model_name}/predictions.csv`
- **Model Metrics:** `models/{model_name}/metrics.json`

**Data Period:**
- Train: 2023-01-31 to 2024-08-12
- Test: 2024-08-13 to 2024-12-30

**Data Source:** ODRE API (French electricity market)

## 🔧 Setup

```bash
# From project root
cd research/notebooks

# Install dependencies
pip install -r ../../requirements.txt

# Launch Jupyter
jupyter notebook
```

## 📈 Key Results (from 05_model_performance_analysis.ipynb)

### Machine Learning Performance

| Model         | R²    | MAPE  | Status      |
|---------------|-------|-------|-------------|
| XGBoost       | 0.686 | 30.1% | Production  |
| LightGBM      | 0.678 | 28.3% | Production  |
| Random Forest | 0.641 | 29.7% | Production  |
| Ridge         | 0.437 | 26.0% | Baseline    |
| GRU (LSTM)    | 0.317 | 50.0% | Not used    |

### Trading Performance

| Model         | Annual Return | Sharpe | Max DD | Win Rate |
|---------------|---------------|--------|--------|----------|
| Random Forest | 88.4%         | 1.65   | -4.2%  | 61.3%    |
| XGBoost       | 76.3%         | 1.45   | -4.3%  | 57.6%    |
| LightGBM      | 59.8%         | 1.19   | -7.3%  | 55.2%    |

**Trading Configuration:**
- Transaction costs: 0.1% per trade
- Entry threshold: 10 EUR/MWh spread
- Stop loss: 2%
- Max holding: 7 days

## ✅ Verification

All results verified for data leakage. See:
- [VERIFICATION_REPORT.md](../../VERIFICATION_REPORT.md)
- [STUDY_DOCUMENTATION.md](../../STUDY_DOCUMENTATION.md)

## 📁 Figures Output

All plots are saved to: `research/figures/`

Generated files:
- `model_performance_comparison.png` - ML metrics comparison
- `predictions_timeseries.png` - Actual vs predicted prices
- `predictions_scatter.png` - Scatter plots with R²
- `residuals_analysis.png` - Residual diagnostic plots
- `trading_performance.png` - Trading metrics comparison
- `risk_return_profile.png` - Risk-return scatter plot

## 🗂️ Deprecated Notebooks

The following notebooks in `notebooks/eda/` are older versions:
- `01_data_exploration.ipynb` (4.5MB) - Superseded by research version
- `02_features_analysis.ipynb` (5.5MB) - Superseded by research version

**Recommendation:** Use the notebooks in `research/notebooks/` as they are more recent and comprehensive.

## 📝 Notes

- All notebooks include markdown explanations
- Cells are numbered for easy reference
- Outputs are saved to avoid re-running expensive computations
- Use "Restart & Run All" for fresh execution
- Figures are high-resolution (300 DPI) for publication

## 🔗 Related Documentation

- [Project README](../../README.md) - Main project overview
- [STUDY_DOCUMENTATION.md](../../STUDY_DOCUMENTATION.md) - Complete study report
- [VERIFICATION_REPORT.md](../../VERIFICATION_REPORT.md) - Data leakage audit

---

**Last Updated:** 2025-11-18
**Status:** ✅ Production Ready
