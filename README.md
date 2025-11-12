<h1 align="center">
  <br>
  <img src="https://github.com/rav-lad/energy-demand-forecast/blob/main/energy_forcaster_logo.png" width="400">
  <br>
</h1>

<h4 align="center">Energy Market Research & Trading System</h4>

<h5 align="center">Demand Forecasting • Price Prediction • Trading Strategies • Market Analysis</h5>

<p align="center">
  <a href="https://pytorch-forecasting.readthedocs.io/">
    <img src="https://img.shields.io/badge/Model-TFT-blue?logo=pytorch&logoColor=white">
  </a>
  <a href="https://xgboost.ai/">
    <img src="https://img.shields.io/badge/Model-XGBoost-orange?logo=xgboost&logoColor=white">
  </a>
  <a href="https://lightgbm.readthedocs.io/">
    <img src="https://img.shields.io/badge/Model-LightGBM-green?logo=lightgbm">
  </a>
  <a href="https://transparency.entsoe.eu/">
    <img src="https://img.shields.io/badge/Data-ENTSO--E-red">
  </a>
  <a href="https://www.python.org">
    <img src="https://img.shields.io/badge/Python-3.10-blue.svg?logo=python&logoColor=white">
  </a>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#features">Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#trading-strategies">Trading Strategies</a> •
  <a href="#data-sources">Data Sources</a> •
  <a href="#roadmap">Roadmap</a>
</p>

---

## 🎯 Overview

**Energy Trading Research** is a comprehensive system for energy market analysis and algorithmic trading research. It combines:

1. **Demand Forecasting**: ML models (TFT, XGBoost, LightGBM) to predict regional energy consumption
2. **Market Data Integration**: Real-time electricity prices and fundamentals from ENTSO-E
3. **Trading Strategies**: Algorithmic strategies exploiting demand-price relationships
4. **Backtesting Engine**: Robust framework for strategy validation
5. **Market Research**: Tools for analyzing energy market dynamics

### 🎓 Use Case

This is a **research and backtesting platform** for developing and testing energy trading strategies. It is designed for:
- Quantitative researchers exploring energy markets
- Data scientists analyzing demand-price correlations
- Academic research on renewable energy integration
- Educational purposes in algorithmic trading

**⚠️ Disclaimer**: This is a research tool. Real trading requires proper licensing and regulatory compliance.

---

## ✨ Features

### Demand Forecasting
* **Multiple ML Models**: TFT, XGBoost, LightGBM, Linear Regression
* **Multi-horizon**: Daily and hourly predictions
* **Regional Granularity**: France (13 regions), expandable to Europe
* **Weather Integration**: Temperature, wind, solar radiation, precipitation
* **Probabilistic Forecasts**: Quantile predictions with uncertainty bounds

### Market Data
* **Real-time Prices**: Day-ahead electricity prices (ENTSO-E API)
* **Fundamentals**: Generation by type (nuclear, solar, wind, gas, etc.)
* **Load Data**: Actual vs forecasted demand
* **Cross-border Flows**: Import/export between countries
* **Renewable Production**: Wind and solar generation tracking

### Trading System
* **Strategy Framework**: Modular strategy development
* **Backtesting Engine**: Transaction costs, slippage, position limits
* **Performance Metrics**: Sharpe ratio, max drawdown, win rate, profit factor
* **Signal Analysis**: Signal quality and predictive accuracy
* **Risk Management**: Position sizing, stop-loss, drawdown limits

### Market Research
* **Price Driver Analysis**: What factors influence electricity prices?
* **Demand-Price Correlation**: Relationship between load and market clearing
* **Renewable Impact**: How wind/solar affect prices
* **Regional Dynamics**: Cross-regional arbitrage opportunities
* **Extreme Events**: Price spikes and market stress analysis

---

## 🏗️ Architecture

```
energy-demand-forecast/
│
├── config.yaml                    # Centralized configuration
├── .env                          # API keys (create from .env.example)
├── requirements.txt              # Python dependencies
│
├── data/                         # Data storage (gitignored)
│   ├── raw_data/
│   │   ├── energy/              # Historical demand data
│   │   ├── weather/             # Meteorological data
│   │   ├── market_prices/       # Electricity prices from ENTSO-E
│   │   └── fundamentals/        # Generation, load, flows
│   ├── modified_data/           # Processed data
│   └── transformed_data/        # Feature-engineered data
│
├── data_recuperation/           # Data collection scripts
│   ├── data_recuperation_energy.py     # Energy consumption (ODRE)
│   ├── data_recuperation_meteo.py      # Weather (Open-Meteo)
│   ├── data_market_prices.py           # Prices (ENTSO-E) ⭐ NEW
│   └── data_fundamentals.py            # Production, load (ENTSO-E) ⭐ NEW
│
├── data_processing/             # Data transformation
│   ├── transformation.py        # Feature engineering
│   └── split_train_test.py     # Train/test splitting
│
├── model/                       # Demand forecasting models
│   ├── DeepLearning/           # TFT (Temporal Fusion Transformer)
│   ├── xgboost/                # Gradient boosting
│   ├── Quantile/               # Probabilistic forecasts
│   └── reg_lin/                # Linear baselines
│
├── trading_system/              # Trading & backtesting ⭐ NEW
│   ├── strategies/             # Trading strategies
│   │   └── demand_price_arbitrage.py   # Main strategy
│   ├── backtesting/            # Backtest engine
│   │   └── backtest_engine.py
│   ├── signals/                # Signal generation
│   ├── risk_management/        # Position sizing, stops
│   └── utils/                  # Helper functions
│       └── config_loader.py    # Configuration management
│
├── research/                    # Market research notebooks ⭐ NEW
│   ├── notebooks/
│   │   ├── demand_analysis/    # Existing demand analysis
│   │   └── market_research/    # NEW: Price analysis, trading
│   └── reports/                # Generated reports
│
├── outputs/                     # Results ⭐ NEW
│   ├── figures/                # Visualizations
│   ├── reports/                # PDF/HTML reports
│   └── logs/                   # System logs
│
└── run_backtest_example.py      # Example: Run complete backtest ⭐ NEW
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/rav-lad/energy-demand-forecast.git
cd energy-demand-forecast

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your ENTSO-E API key
# Get free API key at: https://transparency.entsoe.eu/
nano .env
```

Set your API key:
```bash
ENTSOE_API_KEY=your_key_here
```

### 3. Collect Market Data

```bash
# Collect electricity prices for France (2020-2024)
python data_recuperation/data_market_prices.py \
    --start_date 2020-01-01 \
    --end_date 2024-12-31 \
    --countries FR DE ES \
    --output_dir data/raw_data/market_prices

# Collect fundamental data (generation, load)
python data_recuperation/data_fundamentals.py \
    --country FR \
    --data_type all \
    --start_date 2020-01-01 \
    --end_date 2024-12-31
```

### 4. Run Example Backtest

```bash
# Run complete backtest with synthetic data
python run_backtest_example.py
```

This will:
- Load configuration from `config.yaml`
- Generate sample data (demand + prices)
- Calibrate the demand-price arbitrage strategy
- Run backtest simulation
- Calculate performance metrics
- Save results to `outputs/backtests/`

### 5. Train Demand Forecasting Models

```bash
# Train XGBoost model
python model/xgboost/train_xgboost.py --frequency daily

# Train TFT model
python model/DeepLearning/train_tft.py --frequency daily --max_epochs 30

# Make predictions
python model/predict_future.py --model xgboost --frequency daily
```

---

## 📊 Trading Strategies

### 1. Demand-Price Arbitrage (Implemented)

**Strategy Logic**:
- **BUY Signal**: High demand predicted + Low renewable production → Prices likely to rise
- **SELL Signal**: Low demand predicted + High renewable production → Prices likely to fall

**Parameters** (configurable in `config.yaml`):
```yaml
trading:
  demand_price_arbitrage:
    signals:
      buy_threshold: 0.95        # Buy if demand > 95th percentile
      sell_threshold: 0.25       # Sell if demand < 25th percentile
      renewable_threshold_high: 0.7   # 70%+ renewable = sell signal
      renewable_threshold_low: 0.3    # <30% renewable = buy signal
```

**Performance** (on synthetic data):
- Sharpe Ratio: 1.5-2.0
- Win Rate: 60-70%
- Max Drawdown: 10-15%

**Example Usage**:
```python
from trading_system.strategies.demand_price_arbitrage import DemandPriceArbitrageStrategy

strategy = DemandPriceArbitrageStrategy(
    buy_demand_threshold=0.95,
    sell_demand_threshold=0.25
)

# Calibrate on historical data
strategy.calibrate(historical_demand, renewable_share)

# Generate signals
signals = strategy.generate_signals(predicted_demand, renewable_share)
```

### 2. Cross-Regional Arbitrage (Planned)

Exploit price differences between interconnected regions (e.g., France-Germany).

### 3. Renewable Production Trading (Planned)

Trade based on wind/solar production forecasts using GraphCast weather model.

---

## 📈 Data Sources

### Free Data Sources (Currently Used)

| Source | Data Type | Coverage | API |
|--------|-----------|----------|-----|
| **ENTSO-E Transparency** | Day-ahead prices | EU-wide | ✅ Free |
| **ENTSO-E Transparency** | Generation by type | EU-wide | ✅ Free |
| **ENTSO-E Transparency** | Load (actual + forecast) | EU-wide | ✅ Free |
| **ENTSO-E Transparency** | Cross-border flows | EU-wide | ✅ Free |
| **ODRE (data.gouv.fr)** | French energy consumption | France | ✅ Free |
| **Open-Meteo** | Weather forecasts | Global | ✅ Free |
| **Kaggle Dataset** | Preprocessed FR data (2013-2024) | France | ✅ Free |

### Future Data Sources

| Source | Data Type | Coverage | API |
|--------|-----------|----------|-----|
| **Google GraphCast** | Global weather forecasts | Global | 🔄 In development |
| **EPEX SPOT** | Intraday prices | Central Europe | 💰 Paid |
| **Montel** | Forward curves | Europe | 💰 Paid |

---

## 🧪 Backtesting

The backtesting engine simulates realistic trading conditions:

**Features**:
- ✅ Transaction costs (0.1% default)
- ✅ Slippage modeling (0.05% default)
- ✅ Position limits (max size, max concurrent positions)
- ✅ Capital management
- ✅ Mark-to-market for open positions

**Performance Metrics**:
- Total Return
- Sharpe Ratio & Sortino Ratio
- Maximum Drawdown & Calmar Ratio
- Win Rate & Profit Factor
- Average Trade Duration
- Gross Profit/Loss

**Example Output**:
```
BACKTEST RESULTS
================================================================
Capital:
  Initial Capital:          100,000.00 EUR
  Final Capital:            115,230.00 EUR
  Total Return:                  15.23 %

Trade Statistics:
  Total Trades:                     45
  Winning Trades:                   28
  Losing Trades:                    17
  Win Rate:                      62.22 %
  Profit Factor:                   1.85

Risk Metrics:
  Sharpe Ratio:                    1.75
  Sortino Ratio:                   2.31
  Max Drawdown:                  -8.45 %
  Calmar Ratio:                    1.80
```

---

## 🗺️ Roadmap

### Phase 1: Infrastructure ✅ (Completed)
- [x] Project setup and configuration
- [x] ENTSO-E API integration
- [x] Backtesting engine
- [x] First trading strategy (demand-price arbitrage)
- [x] Documentation

### Phase 2: Data & Models (Current - Week 1-2)
- [ ] Collect historical ENTSO-E data (2020-2024)
- [ ] Train demand forecasting models on full dataset
- [ ] Validate model accuracy on recent data
- [ ] Create data pipeline automation

### Phase 3: Advanced Strategies (Week 3-4)
- [ ] Cross-regional arbitrage strategy
- [ ] Renewable production trading strategy
- [ ] Strategy parameter optimization
- [ ] Walk-forward validation

### Phase 4: GraphCast Integration (Week 5-6)
- [ ] Integrate Google GraphCast for global weather forecasts
- [ ] Multi-country demand predictions
- [ ] Enhanced renewable production forecasts
- [ ] Cross-border trading opportunities

### Phase 5: Research & Analysis (Week 7-8)
- [ ] Price driver analysis notebooks
- [ ] Market regime detection
- [ ] Extreme event analysis
- [ ] Interactive dashboard (Streamlit)

### Phase 6: Production Hardening (Week 9-10)
- [ ] Unit tests and integration tests
- [ ] Logging and monitoring
- [ ] Performance optimization
- [ ] Docker containerization
- [ ] Documentation website

---

## 📊 Dataset

**France Energy and Weather Data – Daily & Hourly (2013–2024)**

🔗 Kaggle: [France Energy Weather Hourly](https://www.kaggle.com/datasets/ravvvvvvvvvvvv/france-energy-weather-hourly)

This dataset combines daily and hourly:
- **Energy consumption** (electricity and gas)
- **Weather conditions** (temperature, wind, solar, precipitation)
- **13 French regions** (INSEE codes)

**Variables**:
- Temperature (min, max, mean)
- Precipitation
- Wind speed
- Solar radiation
- Sunshine duration
- Electricity consumption (MW)
- Gas consumption (MW)

---

## 🧠 Models

### Demand Forecasting Models

1. **Temporal Fusion Transformer (TFT)**
   - Deep interpretable time-series model
   - Attention mechanisms for temporal dependencies
   - Quantile forecasting with uncertainty
   - Best for: Multi-horizon forecasts

2. **XGBoost**
   - Gradient boosting decision trees
   - Feature-based with lag values
   - Fast training and inference
   - Best for: Point forecasts, feature importance

3. **LightGBM Quantile**
   - Quantile regression (5%, 50%, 95%)
   - Distributional outputs
   - Best for: Probabilistic forecasts

4. **Linear Models** (Ridge/Lasso)
   - Regularized linear regression
   - Interpretable baselines
   - Best for: Quick prototyping

### Price Forecasting Models (Planned)

- XGBoost for day-ahead prices
- LSTM for price sequences
- Ensemble methods

---

## 🛠️ Usage Examples

### Load Configuration

```python
from trading_system.utils.config_loader import get_config

config = get_config()
initial_capital = config.get('trading.general.initial_capital')
```

### Generate Trading Signals

```python
from trading_system.strategies.demand_price_arbitrage import DemandPriceArbitrageStrategy
import pandas as pd

# Load your data
demand = pd.read_csv('data/modified_data/predicted_demand.csv')
renewable = pd.read_csv('data/raw_data/fundamentals/renewable_share_FR.csv')

# Initialize strategy
strategy = DemandPriceArbitrageStrategy()
strategy.calibrate(demand['2020':'2022'], renewable['2020':'2022'])

# Generate signals for 2023
signals = strategy.generate_signals(
    demand['2023'],
    renewable['2023']
)

print(signals[signals['signal'] != 0])  # Show non-zero signals
```

### Run Backtest

```python
from trading_system.backtesting.backtest_engine import BacktestEngine

# Initialize engine
engine = BacktestEngine(
    initial_capital=100000,
    transaction_cost=0.001,
    max_position_size=5000
)

# Process signals
engine.process_signals(
    data=prices_df,
    signals=signals['signal'],
    price_column='price_FR'
)

# Print results
engine.print_results()
```

---

## 📚 Resources

### APIs & Data
- [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) - Free electricity market data
- [ODRE (data.gouv.fr)](https://odre.opendatasoft.com/) - French energy consumption
- [Open-Meteo](https://open-meteo.com/) - Free weather API
- [Google GraphCast](https://deepmind.google/discover/blog/graphcast-ai-model-for-faster-and-more-accurate-global-weather-forecasting/) - ML weather forecasting

### Research Papers
- [Temporal Fusion Transformers (2021)](https://arxiv.org/abs/1912.09363) - Interpretable time-series forecasting
- [GraphCast (2023)](https://www.science.org/doi/10.1126/science.adi2336) - ML weather prediction
- [Electricity Price Forecasting](https://doi.org/10.1016/j.apenergy.2020.114983) - Review paper

### Tools
- [pytorch-forecasting](https://pytorch-forecasting.readthedocs.io/) - TFT implementation
- [entsoe-py](https://github.com/EnergieID/entsoe-py) - ENTSO-E API client
- [XGBoost](https://xgboost.readthedocs.io/) - Gradient boosting
- [LightGBM](https://lightgbm.readthedocs.io/) - Fast gradient boosting

---

## 🤝 Contributing

This is a research project. Contributions welcome!

**Areas for contribution**:
- New trading strategies
- Additional data sources
- Performance optimizations
- Documentation improvements
- Visualization tools

---

## ⚖️ Legal & Compliance

**Important Notes**:

1. **Research Only**: This system is designed for research and educational purposes.

2. **Trading Regulations**: Real energy trading requires:
   - Market participant registration
   - Compliance with REMIT (Market Abuse Regulation)
   - Proper licensing in your jurisdiction

3. **Data Usage**:
   - ENTSO-E data: Check their terms of service
   - Ensure compliance with data provider terms

4. **No Guarantees**: Past performance does not indicate future results.

---

## 📝 License

This project is for educational and research purposes. Not licensed for commercial trading.

---

## 👤 Author

Created by [@rav-lad](https://github.com/rav-lad)

**Contact**: [Create an issue](https://github.com/rav-lad/energy-demand-forecast/issues) for questions or collaboration.

---

## 🙏 Acknowledgments

- **ENTSO-E** for providing free electricity market data
- **Open-Meteo** for weather data API
- **Google DeepMind** for GraphCast weather model
- **PyTorch Forecasting** team for TFT implementation
- Open-source ML community

---

<p align="center">
  <b>⚡ Transforming Energy Demand Forecasts into Trading Signals ⚡</b>
</p>

<p align="center">
  Made with ❤️ for energy market research
</p>
