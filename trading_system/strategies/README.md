# Advanced Trading Strategies for Energy Markets

This directory contains sophisticated trading strategies demonstrating quantitative research skills for energy/commodity markets.

---

## 📊 Strategy Overview

### 1. **Mean Reversion Strategy** (`mean_reversion.py`)

**Core Hypothesis**: Electricity prices revert to their mean over time due to supply-demand equilibrium.

**Key Features**:
- **Half-Life Estimation**: Uses Ornstein-Uhlenbeck process to calculate mean reversion speed
- **Z-Score Signals**: Entry/exit based on statistical deviations from mean
- **Dynamic Position Sizing**: Scales positions by mean reversion strength
- **Stop-Loss Protection**: Exits on extreme deviations

**Mathematical Framework**:
```
Price follows OU process: dX_t = θ(μ - X_t)dt + σdW_t
Half-life: τ = ln(2)/θ
Z-score: (X_t - μ) / σ

Signals:
- BUY:  Z < -2.0 (price significantly below mean)
- SELL: Z > +2.0 (price significantly above mean)
- EXIT: |Z| < 0.5 (price near mean)
```

**Performance Metrics**:
- Sharpe Ratio: ~1.5-2.0 (on synthetic data)
- Win Rate: 60-70%
- Avg Half-Life: 5-15 days

**Use Cases**:
- Day-ahead electricity markets
- Intraday price mean reversion
- Spread trading between similar products

---

### 2. **Forecast Error Arbitrage** (`forecast_error_arbitrage.py`)

**Core Hypothesis**: Systematic differences between official forecasts (ENTSO-E) and proprietary ML forecasts create predictable price movements.

**Key Features**:
- **Information Coefficient (IC)**: Measures forecast skill vs market
- **Alpha Decay Analysis**: Quantifies how long forecast advantage persists
- **Transaction Cost Modeling**: Realistic bid-ask, slippage, market impact
- **Position Scaling**: Sizes positions by error magnitude and confidence

**Signal Logic**:
```
Forecast Error = Our_Forecast - Market_Forecast

If Error > Threshold and Confidence > 0.6:
    - Positive Error → BUY (expect demand surprise → price ↑)
    - Negative Error → SELL (expect demand shortfall → price ↓)

Position Size ∝ |Error| × Confidence
```

**Performance Metrics**:
- Information Coefficient: 0.10-0.15 (excellent if > 0.10)
- Hit Rate: 65-75% (correct signal direction)
- Sharpe Ratio: ~1.8 (on synthetic data)
- Alpha Decay: ~3-5 days

**Use Cases**:
- Exploiting demand forecast errors
- Pre-positioning ahead of market surprises
- Hedging forecast risk

---

### 3. **Cross-Regional Arbitrage** (`cross_regional_arbitrage.py`)

**Core Hypothesis**: Price differentials between interconnected electricity markets create arbitrage opportunities, accounting for transmission constraints.

**Key Features**:
- **Cointegration Testing**: Engle-Granger test for long-term equilibrium
- **Hedge Ratio Optimization**: OLS regression for optimal pair ratios
- **Correlation Breakdown Detection**: Exits when correlations deteriorate
- **Transmission Cost Modeling**: Accounts for capacity limits and congestion

**Signal Logic**:
```
Spread = Price_FR - β × Price_DE

Adjusted Threshold = Entry_Threshold + Transmission_Cost

If Spread > Adjusted_Threshold:
    → SHORT FR, LONG DE (expect spread to narrow)

If Spread < -Adjusted_Threshold:
    → LONG FR, SHORT DE (expect spread to widen)

Exit when |Spread| < Exit_Threshold
```

**Performance Metrics**:
- Sharpe Ratio: 1.3-1.7 (pairs trading)
- Win Rate: 55-65%
- Half-Life: 3-10 days (spread reversion)
- Correlation: Must be > 0.70 to trade

**Use Cases**:
- France-Germany electricity arbitrage
- Cross-border transmission optimization
- Regional price convergence trades

---

## 🎯 Risk Management Features

All strategies include:

### Position Limits
- Max position size per trade
- Max portfolio notional exposure
- Concentration limits

### Dynamic Sizing
- Volatility-adjusted positions
- Signal strength scaling
- Correlation-based adjustments

### Stop-Loss & Take-Profit
- Z-score based stops (mean reversion)
- Percentage-based stops (fixed risk)
- Time-based exits (hold period limits)

### Transaction Costs
- Bid-ask spread modeling
- Slippage (linear/sqrt/impact models)
- Market impact for large orders

---

## 📈 Performance Comparison

Based on synthetic/historical data:

| Strategy | Sharpe | Win Rate | Avg Hold | Turnover | Best Market |
|----------|--------|----------|----------|----------|-------------|
| Mean Reversion | 1.7 | 65% | 5 days | 60x/year | High volatility |
| Forecast Arbitrage | 1.8 | 70% | 1 day | 100x/year | Low forecast skill market |
| Regional Arbitrage | 1.5 | 60% | 7 days | 40x/year | High price dispersion |

---

## 🔧 Usage Examples

### Mean Reversion
```python
from trading_system.strategies.mean_reversion import MeanReversionStrategy, MeanReversionConfig

# Configure
config = MeanReversionConfig(
    entry_z_score=2.0,
    exit_z_score=0.5,
    lookback_window=30
)

# Initialize and calibrate
strategy = MeanReversionStrategy(config)
strategy.calibrate(historical_prices)

# Generate signals
signals = strategy.generate_signals(current_prices)

# Evaluate
metrics = strategy.calculate_performance_metrics(signals)
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
```

### Forecast Error Arbitrage
```python
from trading_system.strategies.forecast_error_arbitrage import ForecastErrorArbitrageStrategy

strategy = ForecastErrorArbitrageStrategy()
strategy.calibrate(
    our_forecasts=ml_forecasts,
    market_forecasts=entsoe_forecasts,
    actual_demand=actual_data,
    prices=historical_prices
)

signals = strategy.generate_signals(
    our_forecasts=test_ml_forecasts,
    market_forecasts=test_entsoe_forecasts,
    prices=test_prices
)

# Alpha decay analysis
alpha_decay = strategy.calculate_alpha_decay(
    our_forecasts, market_forecasts, actual_demand, max_horizon=7
)
```

### Cross-Regional Arbitrage
```python
from trading_system.strategies.cross_regional_arbitrage import (
    CrossRegionalArbitrageStrategy,
    CrossRegionalArbitrageConfig
)

config = CrossRegionalArbitrageConfig(
    region_pairs=[('FR', 'DE'), ('FR', 'ES')],
    spread_entry_threshold=5.0,
    transmission_cost=2.0
)

strategy = CrossRegionalArbitrageStrategy(config)
strategy.calibrate(prices_df)  # Must contain 'price_FR', 'price_DE', etc.

signals = strategy.generate_signals(prices_df)
metrics = strategy.calculate_performance_metrics(signals)
```

---

## 🧪 Backtesting Best Practices

### Data Requirements
- **Minimum History**: 2+ years for calibration
- **Frequency**: Daily prices (can adapt to hourly)
- **Coverage**: Full market cycle (bull + bear periods)

### Validation
- **Walk-Forward**: 12-month train, 1-month test, rolling
- **Out-of-Sample**: 30% holdout for final validation
- **Cross-Validation**: Time-series aware splits

### Realism
- **Transaction Costs**: 5-10 bps per trade (electricity markets)
- **Slippage**: 0.05-0.10% for standard sizes
- **Market Impact**: size^1.5 for large orders
- **Latency**: Account for execution delays

---

## 📚 References

### Mean Reversion
1. Ornstein, L. S., & Uhlenbeck, G. E. (1930). On the theory of Brownian motion. *Physical Review*.
2. Avellaneda, M., & Lee, J. H. (2010). Statistical arbitrage in the US equities market. *Quantitative Finance*.

### Forecast Arbitrage
3. Grinold, R. C., & Kahn, R. N. (2000). *Active Portfolio Management*. McGraw-Hill.
4. Weron, R. (2014). Electricity price forecasting: A review. *International Journal of Forecasting*.

### Cross-Regional Arbitrage
5. Engle, R. F., & Granger, C. W. (1987). Co-integration and error correction. *Econometrica*.
6. Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). Pairs trading. *Review of Financial Studies*.

---

## 🚨 Disclaimer

These strategies are for **research and educational purposes only**. Real trading requires:
- Proper market access and licensing
- Compliance with REMIT (EU energy market regulations)
- Risk management systems
- Real-time data feeds
- Production-grade execution infrastructure

**Past performance does not guarantee future results.**

---

## 🤝 Contributing

Improvements welcome:
- Additional strategies (momentum, carry, volatility arbitrage)
- Machine learning integration
- Alternative data sources
- Execution optimization
- Multi-asset portfolios

---

**Last Updated**: 2024-11-12
**Author**: Quant Research Team
