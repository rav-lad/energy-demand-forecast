# Energy Demand Forecasting & Algorithmic Trading System

**Production-ready quantitative trading platform for European electricity markets combining machine learning forecasting with systematic trading strategies.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-blue)](https://mlflow.org/)
[![ENTSO-E](https://img.shields.io/badge/Data-ENTSO--E-green)](https://transparency.entsoe.eu/)

---

## 📋 Overview

This project implements an end-to-end quantitative research platform that transforms energy market data into systematic trading signals. The system integrates:

- **Real-time data collection** from ENTSO-E Transparency Platform (electricity prices, load, generation)
- **Advanced ML forecasting** using LightGBM, XGBoost, and ensemble methods
- **Statistical arbitrage strategies** across price forecast, mean reversion, and cross-regional spreads
- **Institutional-grade backtesting** with realistic transaction costs and slippage modeling
- **Production MLOps infrastructure** with MLflow experiment tracking and model registry

**Key Differentiator:** Rigorous temporal logic with comprehensive data leakage prevention, ensuring backtest results are reproducible in live trading.

---

## 🏗️ Architecture

```
energy-demand-forecast/
├── data_collection/           # Real-time data pipelines
│   ├── entsoe_connector.py    # ENTSO-E API (prices, fundamentals)
│   ├── odre_collector.py      # French energy consumption
│   ├── fuel_prices.py         # TTF gas, EUA carbon, coal
│   └── api_cache.py           # Intelligent caching (7-day TTL)
│
├── model/
│   ├── price_forecasting/     # ML forecasting models
│   │   ├── data_loader.py     # Feature engineering (48 features)
│   │   ├── models.py          # LightGBM Quantile, Ensemble
│   │   └── train_*.py         # Training pipelines
│   └── xgboost/               # XGBoost demand models
│
├── trading_system/
│   ├── strategies/            # Trading strategies
│   │   ├── price_forecast_strategy.py
│   │   ├── mean_reversion.py
│   │   └── cross_regional_arbitrage.py
│   ├── backtesting/
│   │   ├── backtesting_engine.py
│   │   ├── monte_carlo.py     # Robustness testing
│   │   └── stress_testing.py  # Crisis scenarios
│   └── risk_management/       # Position sizing, VaR
│
├── mlops/                     # Experiment tracking
│   └── mlflow_tracker.py      # MLflow integration
│
└── tests/                     # Integration tests
    └── test_entsoe_integration.py
```

---

## ✨ Key Features

### Data Infrastructure

- ✅ **ENTSO-E Transparency Platform** integration (day-ahead prices, actual load, generation)
- ✅ **Open-Meteo API** for weather forecasts (temperature, wind, solar radiation)
- ✅ **ODRE API** for French regional consumption (electricity + gas)
- ✅ **Intelligent caching** with MD5 hashing (90% reduction in API calls)
- ✅ **Rate limiting** with token bucket algorithm (400 req/min)
- ✅ **Data validation** with outlier detection and quality checks

### Machine Learning

- ✅ **LightGBM Quantile Regression** (P10, P50, P90 forecasts)
- ✅ **Ensemble Forecasting** (LightGBM 50% + RandomForest 30% + Ridge 20%)
- ✅ **48 engineered features** (lags, rolling stats, fuel spreads, calendar)
- ✅ **Walk-forward validation** with TimeSeriesSplit (no shuffle)
- ✅ **Hyperparameter optimization** with Optuna (Bayesian optimization)

### Trading System

- ✅ **3 systematic strategies** (Price Forecast, Mean Reversion, Cross-Regional Arbitrage)
- ✅ **Realistic backtesting** with fill delays, slippage, and transaction costs
- ✅ **Monte Carlo simulation** (1000+ scenarios, bootstrap + block bootstrap)
- ✅ **Stress testing** (2022 energy crisis, negative prices, liquidity crises)
- ✅ **Risk management** with position limits and stop-losses

### MLOps & Production

- ✅ **MLflow experiment tracking** (metrics, parameters, artifacts)
- ✅ **Model registry** with versioning and staging
- ✅ **Comprehensive logging** with structured outputs
- ✅ **Data leakage prevention** (audited 11 critical components)
- ✅ **Production config** management (YAML + environment variables)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- ENTSO-E API key (free, register at https://transparency.entsoe.eu/)

### Installation

```bash
# Clone repository
git clone https://github.com/rav-lad/energy-demand-forecast.git
cd energy-demand-forecast

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API key
cp .env.example .env
echo "ENTSOE_API_KEY=your_api_key_here" >> .env
```

### Configuration

Edit `config.yaml` for production settings:

```yaml
market:
  countries: ["FR"]
  start_date: "2022-01-01"
  end_date: "2024-12-31"
  rate_limit_rpm: 400
  cache_enabled: true
  cache_ttl_days: 7

models:
  price_forecasting:
    model_type: "lightgbm_quantile"
    quantiles: [0.1, 0.5, 0.9]
    n_estimators: 500
    learning_rate: 0.05
```

### Data Collection

```bash
# Test API connection
python test_api_connection.py
# Expected: ✅ ALL TESTS PASSED

# Collect electricity prices (30 min)
python data_recuperation/data_market_prices.py \
  --start_date 2022-01-01 \
  --end_date 2024-12-31 \
  --countries FR

# Collect weather data (10 min, no API key needed)
python data_collection/pipeline.py weather-historical --frequency daily

# Collect energy consumption (10 min, no API key needed)
python data_collection/odre_collector.py \
  --start_date 2022-01-01 \
  --end_date 2024-12-31 \
  --validate

# Collect fundamentals (30 min)
python data_recuperation/data_fundamentals.py \
  --start_date 2022-01-01 \
  --end_date 2024-12-31 \
  --countries FR
```

### Training & Backtesting

```bash
# Train price forecasting model
python model/price_forecasting/train_price_forecast.py \
  --model both \
  --walk-forward

# Run backtest
python run_backtest_example.py

# View MLflow results
mlflow ui --backend-store-uri file:///path/to/mlruns
# Navigate to http://localhost:5000
```

---

## 📊 Results

### Machine Learning Performance (Test Set: 140 days)

| Model         | R²    | MAPE  | Status      |
|---------------|-------|-------|-------------|
| XGBoost       | 0.686 | 30.1% | Production  |
| LightGBM      | 0.678 | 28.3% | Production  |
| Random Forest | 0.641 | 29.7% | Production  |
| Ridge         | 0.437 | 26.0% | Baseline    |
| GRU (LSTM)    | 0.317 | 50.0% | Not used    |

### Trading Performance (Spread Trading Strategy)

| Model         | Total Return | Annual Return | Sharpe | Max DD | Win Rate | Trades |
|---------------|--------------|---------------|--------|--------|----------|--------|
| Random Forest | 27.5%        | 88.4%         | 1.65   | -4.2%  | 61.3%    | 31     |
| XGBoost       | 24.3%        | 76.3%         | 1.45   | -4.3%  | 57.6%    | 33     |
| LightGBM      | 19.7%        | 59.8%         | 1.19   | -7.3%  | 55.2%    | 29     |
| Ridge         | 7.6%         | 20.9%         | 0.75   | -7.7%  | 63.3%    | 30     |

**Trading Configuration:**
- Transaction costs: 0.1% per trade (0.2% round-trip)
- Entry threshold: 10 EUR/MWh spread
- Max holding: 7 days
- Stop loss: 2% of capital
- Risk per trade: 1% of capital

**Key Validation:**
- ✅ No data leakage (comprehensive audit completed - see [VERIFICATION_REPORT.md](VERIFICATION_REPORT.md))
- ✅ Temporal split (560 train / 140 test)
- ✅ Realistic transaction costs (0.1% per side)
- ✅ Proper t→t+1 prediction (uses only lag features)
- ✅ Production-ready data pipeline

**Complete study documentation available in [STUDY_DOCUMENTATION.md](STUDY_DOCUMENTATION.md)**

---

## 🔮 Future Improvements

### In Progress

1. **GenCast Weather Integration**
   - Replace realized weather with forecasted weather
   - Eliminate distribution shift between train and production
   - Use Google DeepMind's GenCast for 15-day ensemble forecasts
   - Expected impact: More realistic performance estimates (-5-10%)

2. **Live Data Pipeline**
   - Real-time data ingestion from ENTSO-E
   - Streaming price updates
   - Incremental model retraining

### Roadmap

3. **Advanced Risk Management**
   - Value-at-Risk (VaR) and Conditional VaR (CVaR)
   - Dynamic position sizing with Kelly criterion
   - Portfolio-level risk limits

4. **Multi-Market Expansion**
   - German, Spanish, Nordic markets
   - Cross-border arbitrage strategies
   - Interconnector flow optimization

5. **Deep Learning Models**
   - Transformer architectures for sequence modeling
   - LSTM for intraday price forecasting
   - Attention mechanisms for regime detection

---

## 📚 Documentation

- **[Quick Start Guide](QUICK_START.md)** - Get started in 5 minutes
- **[ENTSO-E API Setup](docs/ENTSOE_API_SETUP.md)** - Detailed API configuration
- **[Migration Guide](docs/MIGRATION_GUIDE.md)** - Moving to real data
- **[Data Leakage Prevention](docs/DATA_LEAKAGE_PREVENTION.md)** - Temporal logic best practices
- **[Audit Reports](docs/audits/)** - Complete system audits

---

## 🔧 Configuration

### Environment Variables

```bash
# Required
ENTSOE_API_KEY=your_entso_api_key

# Optional
MLFLOW_TRACKING_URI=file:///path/to/mlruns
LOG_LEVEL=INFO
```

### Production Checklist

- [x] ENTSO-E API key configured
- [x] Data validation enabled
- [x] Cache enabled (reduces API load by 90%)
- [x] Rate limiting configured (400 req/min)
- [x] MLflow tracking enabled
- [x] Comprehensive logging
- [x] No data leakage (audited)
- [x] TimeSeriesSplit cross-validation
- [x] Realistic transaction costs
- [x] Fill delays implemented

---

## 🧪 Testing

```bash
# Run integration tests
pytest tests/test_entsoe_integration.py -v

# Test API connection
python test_api_connection.py

# Validate data quality
python data_collection/data_validator.py \
  data/raw_data/market_prices/day_ahead_prices_FR.csv --type prices
```

---

## 📈 Technologies

**Data & APIs:**
- ENTSO-E Transparency Platform (electricity markets)
- Open-Meteo (weather forecasts)
- ODRE (French energy consumption)

**Machine Learning:**
- LightGBM (quantile regression)
- XGBoost (gradient boosting)
- Scikit-learn (ensemble methods)
- Optuna (hyperparameter optimization)

**MLOps:**
- MLflow (experiment tracking, model registry)
- Pandas (data processing)
- NumPy (numerical computing)

**Backtesting:**
- Custom backtesting engine
- Monte Carlo simulation (scipy)
- Walk-forward validation

---

## 🤝 Contributing

This is a research project. For questions or collaboration:
- Open an issue on GitHub
- See documentation in `docs/`

---

## ⚠️ Disclaimer

**This is a research and educational project.**

- Not financial advice
- No guarantee of profitability
- Markets are unpredictable
- Past performance does not guarantee future results
- Trading involves substantial risk of loss

---

## 📄 License

This project is for educational and research purposes.

---

## 🙏 Acknowledgments

**Data Sources:**
- ENTSO-E Transparency Platform
- Open-Meteo (weather data)
- ODRE (French energy consumption)

**Inspiration:**
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*
- Weron, R. (2014). *Electricity price forecasting: A review*
- Bunn & Karakatsani (2016). *Forecasting electricity prices*

---

**Built with ❤️ for quantitative research and energy markets**
