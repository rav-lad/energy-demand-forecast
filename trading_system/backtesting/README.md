# Backtesting Framework

A comprehensive, production-grade backtesting infrastructure for energy trading strategies with rigorous validation methodologies.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Components](#components)
- [Usage Examples](#usage-examples)
- [Methodologies](#methodologies)
- [Best Practices](#best-practices)
- [References](#references)

## 🎯 Overview

This backtesting framework implements institutional-grade validation methodologies to ensure trading strategies perform robustly in production. It addresses common pitfalls such as overfitting, lookahead bias, and unrealistic assumptions.

### Key Principles

1. **Realism**: Accurate modeling of transaction costs, slippage, and market impact
2. **Robustness**: Walk-forward validation and Monte Carlo simulation prevent overfitting
3. **Transparency**: Detailed performance attribution and risk analysis
4. **Reproducibility**: Deterministic results with proper random seed management

## ✨ Features

### Core Backtesting Engine
- Event-driven architecture (no lookahead bias)
- Realistic transaction cost modeling
  - Percentage-based commissions
  - Fixed costs per trade
  - Market impact (√volume scaling)
  - Volatility-adjusted slippage
- Position tracking with mark-to-market
- Configurable execution delays
- Multiple order types (market, limit, stop)

### Walk-Forward Validation
- Prevents overfitting through out-of-sample testing
- Anchored and rolling window modes
- Parameter optimization on training data
- Performance validation on unseen data
- Efficiency Ratio calculation (OOS/IS performance)

### Monte Carlo Simulation
- Statistical confidence intervals
- Multiple simulation methods:
  - Bootstrap resampling (empirical distribution)
  - Block bootstrap (preserves autocorrelation)
  - Parametric simulation (t-distribution)
  - Parameter perturbation (sensitivity analysis)
- Probability-based risk metrics

### Performance Attribution
- Alpha/Beta decomposition (CAPM)
- Information Ratio calculation
- Multi-factor attribution
- Regime analysis (bull/bear markets)

### Results Analysis
- Comprehensive trade analysis
- Returns distribution analysis
- Rolling performance metrics
- Professional tear sheets
- Visualization suite

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Integrated Backtester                     │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  Backtest   │  │     Risk     │  │   Performance    │  │
│  │   Engine    │──│  Management  │──│   Attribution    │  │
│  └─────────────┘  └──────────────┘  └──────────────────┘  │
└─────────────────────────────────────────────────────────────┘
           │                    │                    │
           ▼                    ▼                    ▼
  ┌─────────────────┐  ┌─────────────────┐  ┌────────────────┐
  │  Walk-Forward   │  │  Monte Carlo    │  │    Results     │
  │   Validation    │  │   Simulation    │  │    Analyzer    │
  └─────────────────┘  └─────────────────┘  └────────────────┘
```

## 🧩 Components

### 1. BacktestingEngine

Core backtesting engine with realistic execution modeling.

```python
from trading_system.backtesting import BacktestingEngine, BacktestConfig, TransactionCostModel

# Configure transaction costs
cost_model = TransactionCostModel(
    commission_pct=0.001,      # 0.1% commission
    slippage_pct=0.0005,       # 0.05% slippage
    market_impact_coef=0.01    # Market impact coefficient
)

# Configure backtest
config = BacktestConfig(
    initial_capital=100000.0,
    cost_model=cost_model,
    max_position_size=1000.0,
    max_leverage=3.0,
    fill_delay_bars=1          # Fill at next bar (realistic)
)

# Run backtest
engine = BacktestingEngine(config)
results = engine.run(price_data, signals, symbol='ENERGY')
metrics = engine.get_performance_metrics()

print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
```

### 2. Walk-Forward Validation

Prevent overfitting with out-of-sample validation.

```python
from trading_system.backtesting import WalkForwardAnalyzer, WalkForwardConfig

# Configure walk-forward
config = WalkForwardConfig(
    train_period_days=180,     # 6 months training
    test_period_days=60,       # 2 months testing
    step_days=30,              # Monthly reoptimization
    window_type='rolling',     # or 'anchored'
    optimization_metric='sharpe_ratio'
)

# Define parameter grid
param_grid = {
    'lookback': [20, 30, 50],
    'threshold': [1.5, 2.0, 2.5]
}

# Run walk-forward analysis
wfa = WalkForwardAnalyzer(config)
results = wfa.run(price_data, strategy_func, param_grid)

print(f"Efficiency Ratio: {results['efficiency_ratio']:.3f}")
print(f"Avg OOS Sharpe: {results['aggregate_metrics']['avg_test_sharpe']:.3f}")

# Plot results
wfa.plot_results(results, save_path='walk_forward_results.png')
```

**Interpreting Efficiency Ratio:**
- ER > 0.7: Robust strategy ✅
- ER 0.5-0.7: Acceptable performance
- ER < 0.5: Severe overfitting ⚠️

### 3. Monte Carlo Simulation

Assess statistical confidence and robustness.

```python
from trading_system.backtesting import MonteCarloSimulator, MonteCarloConfig

# Configure Monte Carlo
config = MonteCarloConfig(
    n_simulations=1000,
    method='bootstrap',        # or 'parametric', 'block_bootstrap', 'parameter_perturbation'
    confidence_level=0.95,
    random_seed=42
)

# Run simulation
simulator = MonteCarloSimulator(config)
mc_results = simulator.run(
    original_returns=strategy_returns,
    strategy_func=strategy_func,
    price_data=price_data,
    params=optimal_params,
    original_backtest_results=backtest_metrics
)

print(mc_results)  # Comprehensive summary

# Key metrics
print(f"Mean Sharpe: {mc_results.mean_sharpe:.3f}")
print(f"95% CI: [{mc_results.ci_sharpe_lower:.3f}, {mc_results.ci_sharpe_upper:.3f}]")
print(f"P(Sharpe > 1.0): {mc_results.prob_sharpe_above_1:.1%}")

# Visualization
simulator.plot_results(mc_results, save_path='monte_carlo_results.png')
```

### 4. Integrated Backtesting

Combine backtesting with attribution and risk management.

```python
from trading_system.backtesting import IntegratedBacktester, IntegratedBacktestConfig
from trading_system.backtesting import BacktestConfig
from trading_system.risk_management import RiskLimits

# Configure
config = IntegratedBacktestConfig(
    backtest_config=BacktestConfig(initial_capital=100000),
    enable_risk_management=True,
    risk_limits=RiskLimits(max_var_daily=10000.0)
)

# Run integrated backtest
backtester = IntegratedBacktester(config)
results = backtester.run(
    price_data=prices,
    signals=signals,
    benchmark_returns=benchmark
)

# Results include:
# - backtest_results: Core metrics
# - attribution: Alpha/beta decomposition
# - risk_metrics: VaR, CVaR, Sortino
# - report: Professional summary

print(results['report'])
```

### 5. Results Analysis

Comprehensive analysis and visualization.

```python
from trading_system.backtesting import ResultsAnalyzer

analyzer = ResultsAnalyzer()

# Trade analysis
trade_analysis = analyzer.analyze_trades(engine.trades)
print(trade_analysis)  # Win rate, profit factor, etc.

# Returns distribution
dist_stats = analyzer.analyze_returns_distribution(returns)
print(f"Skewness: {dist_stats['skewness']:.3f}")
print(f"Kurtosis: {dist_stats['kurtosis']:.3f}")

# Regime analysis
regime_stats = analyzer.regime_analysis(equity_curve)

# Visualizations
analyzer.plot_equity_curve(equity_curve, save_path='equity_curve.png')
analyzer.plot_returns_distribution(returns, save_path='returns_dist.png')
analyzer.plot_rolling_metrics(equity_curve, window=60, save_path='rolling_metrics.png')

# Generate tear sheet
tearsheet = analyzer.generate_tearsheet(results, save_path='tearsheet.txt')
```

## 📚 Methodologies

### Walk-Forward Analysis

Walk-forward analysis simulates real trading by repeatedly:
1. Optimizing parameters on in-sample data
2. Testing on out-of-sample data
3. Moving forward in time

**Mathematical Framework:**

For total period T, divide into windows:
- Training: t to t+n_train
- Testing: t+n_train to t+n_train+n_test
- Step forward by n_step

**Efficiency Ratio:**
```
ER = Sharpe_OOS / Sharpe_IS

where:
- Sharpe_OOS = Average out-of-sample Sharpe ratio
- Sharpe_IS = Average in-sample Sharpe ratio
```

**Advantages:**
- Prevents overfitting
- Realistic simulation of parameter reoptimization
- Detects parameter stability

**Limitations:**
- Requires sufficient data
- Computationally intensive
- May be conservative (OOS < IS expected)

### Monte Carlo Simulation

Monte Carlo methods provide statistical confidence through simulation.

#### 1. Bootstrap Resampling

Randomly resample historical returns with replacement.

```python
# Pseudo-code
resampled_returns = np.random.choice(original_returns, size=n, replace=True)
synthetic_equity = initial_capital * np.cumprod(1 + resampled_returns)
```

**Pros:** Preserves empirical distribution
**Cons:** Loses temporal structure

#### 2. Block Bootstrap

Resample blocks of consecutive returns.

```python
# Preserve autocorrelation
n_blocks = n / block_size
for _ in range(n_blocks):
    start = random.randint(0, n - block_size)
    block = returns[start:start+block_size]
    resampled_returns.extend(block)
```

**Pros:** Preserves autocorrelation
**Cons:** More complex

#### 3. Parametric Simulation

Fit returns to statistical distribution and simulate.

```python
# Fit Student's t-distribution (captures fat tails)
df, loc, scale = stats.t.fit(returns)
synthetic_returns = stats.t.rvs(df, loc=loc, scale=scale, size=n)
```

**Pros:** Smooth, captures fat tails
**Cons:** Assumes distributional form

#### 4. Parameter Perturbation

Add noise to strategy parameters.

```python
# Test parameter stability
perturbed_param = param * (1 + random.uniform(-0.1, 0.1))
```

**Pros:** Tests robustness to parameter uncertainty
**Cons:** Requires carefully chosen noise levels

### Performance Attribution

Decompose returns into components.

#### Alpha-Beta (CAPM)

```
R_strategy = α + β * R_benchmark + ε

where:
- α = Excess return (skill)
- β = Market exposure (systematic risk)
- ε = Residual (idiosyncratic risk)
```

**Information Ratio:**
```
IR = α / σ(ε)

where σ(ε) is tracking error
```

**Interpretation:**
- IR > 0.5: Excellent
- IR > 1.0: Outstanding (very rare)

#### Multi-Factor Attribution

```
R = α + β₁*F₁ + β₂*F₂ + ... + βₙ*Fₙ + ε

Common factors:
- Market (equity/commodity index)
- Size (large vs small cap)
- Value (value vs growth)
- Momentum (recent performance)
- Volatility (low vs high vol)
```

## 📖 Best Practices

### 1. Data Quality

- **No survivorship bias**: Include delisted/failed instruments
- **Point-in-time data**: Only use information available at decision time
- **Corporate actions**: Adjust for splits, dividends
- **Time zones**: Ensure consistent timestamps

### 2. Realistic Assumptions

- **Transaction costs**: Model all costs (commission + slippage + market impact)
- **Execution delay**: Fill at next bar minimum
- **Liquidity constraints**: Limit position size relative to volume
- **Overnight risk**: Account for gaps and after-hours moves

### 3. Risk Management

- **Position limits**: Maximum size per instrument
- **Portfolio limits**: Maximum total exposure
- **Stop-loss**: Automatic exit at drawdown threshold
- **VaR limits**: Daily Value at Risk constraints

### 4. Validation

- **Walk-forward**: Mandatory for any strategy claiming predictive power
- **Monte Carlo**: Assess statistical significance
- **Regime analysis**: Test across different market conditions
- **Stress testing**: Simulate extreme scenarios

### 5. Overfitting Prevention

- **Simplicity**: Prefer simple strategies over complex ones (Occam's Razor)
- **Parsimony**: Minimize number of parameters
- **Economic rationale**: Strategy must have logical explanation
- **Out-of-sample testing**: Always reserve holdout data

### 6. Reporting

- **Transparency**: Report all metrics (including unfavorable ones)
- **Context**: Compare to relevant benchmark
- **Attribution**: Explain sources of returns
- **Costs**: Clearly state impact of transaction costs

## 📊 Performance Metrics Reference

### Return Metrics

| Metric | Formula | Good Value |
|--------|---------|------------|
| Total Return | (Final - Initial) / Initial | > 0% |
| CAGR | (Final/Initial)^(1/years) - 1 | > 10% |
| Excess Return | Return - Risk-Free Rate | > 5% |

### Risk-Adjusted Metrics

| Metric | Formula | Good Value |
|--------|---------|------------|
| Sharpe Ratio | (Return - RFR) / Volatility | > 1.0 |
| Sortino Ratio | Excess Return / Downside Dev | > 1.5 |
| Calmar Ratio | CAGR / Max Drawdown | > 3.0 |
| Information Ratio | Alpha / Tracking Error | > 0.5 |

### Risk Metrics

| Metric | Description | Acceptable |
|--------|-------------|------------|
| Max Drawdown | Largest peak-to-trough decline | < 20% |
| VaR (95%) | Maximum loss (95% confidence) | < 5% daily |
| CVaR (95%) | Expected loss beyond VaR | < 8% daily |
| Volatility | Standard deviation (annualized) | < 30% |

### Trading Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| Win Rate | % of profitable trades | > 50% |
| Profit Factor | Gross Profit / Gross Loss | > 1.5 |
| Avg Win/Avg Loss | Risk-reward ratio | > 1.5 |

## 🔬 Statistical Tests

### Normality Tests

```python
from scipy import stats

# Jarque-Bera test
jb_stat, jb_pvalue = stats.jarque_bera(returns)
is_normal = jb_pvalue > 0.05  # H0: Normal distribution
```

### Stationarity Tests

```python
from statsmodels.tsa.stattools import adfuller

# Augmented Dickey-Fuller test
adf_stat, adf_pvalue = adfuller(price_series)
is_stationary = adf_pvalue < 0.05  # H0: Non-stationary (has unit root)
```

### Autocorrelation

```python
from statsmodels.stats.diagnostic import acorr_ljungbox

# Ljung-Box test
lb_stat, lb_pvalue = acorr_ljungbox(returns, lags=10)
has_autocorr = (lb_pvalue < 0.05).any()  # H0: No autocorrelation
```

## 📚 References

### Academic Papers

1. **Prado, M. L. (2018)**. *Advances in Financial Machine Learning*. Wiley.
   - Chapter 7: Cross-Validation in Finance
   - Chapter 11: The Dangers of Backtesting

2. **Bailey, D. H., Borwein, J., Lopez de Prado, M., & Zhu, Q. J. (2014)**.
   *Pseudomathematics and Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance*.
   Notices of the AMS, 61(5), 458-471.

3. **Harvey, C. R., & Liu, Y. (2015)**.
   *Backtesting*.
   The Journal of Portfolio Management, 41(1), 13-28.

4. **Pardo, R. (2011)**.
   *The Evaluation and Optimization of Trading Strategies* (2nd ed.).
   Wiley Trading.

### Books

- **Jansen, S. (2020)**. *Machine Learning for Algorithmic Trading* (2nd ed.). Packt.
- **Chan, E. P. (2013)**. *Algorithmic Trading: Winning Strategies and Their Rationale*. Wiley.
- **Narang, R. K. (2013)**. *Inside the Black Box: A Simple Guide to Quantitative and High Frequency Trading* (2nd ed.). Wiley.

### Industry Standards

- **CFA Institute**: *Global Investment Performance Standards (GIPS)*
- **AIMR**: *Performance Presentation Standards*

## 🚀 Quick Start Example

Complete example combining all components:

```python
from trading_system.backtesting import (
    IntegratedBacktester, IntegratedBacktestConfig,
    WalkForwardAnalyzer, WalkForwardConfig,
    MonteCarloSimulator, MonteCarloConfig,
    ResultsAnalyzer, BacktestConfig
)

# 1. Run integrated backtest
backtest_config = BacktestConfig(initial_capital=100000)
integrated_config = IntegratedBacktestConfig(
    backtest_config=backtest_config,
    enable_risk_management=True
)

backtester = IntegratedBacktester(integrated_config)
results = backtester.run(price_data, signals, benchmark_returns=benchmark)

# 2. Walk-forward validation
wf_config = WalkForwardConfig(train_period_days=180, test_period_days=60)
wfa = WalkForwardAnalyzer(wf_config)
wf_results = wfa.run(price_data, strategy_func, param_grid)

# 3. Monte Carlo simulation
mc_config = MonteCarloConfig(n_simulations=1000)
simulator = MonteCarloSimulator(mc_config)
mc_results = simulator.run(
    results['backtest_results']['returns'],
    strategy_func,
    price_data,
    optimal_params,
    results['performance_metrics']
)

# 4. Analysis
analyzer = ResultsAnalyzer()
trade_analysis = analyzer.analyze_trades(results['trades'])
tearsheet = analyzer.generate_tearsheet(results)

# 5. Report
print("="*80)
print("COMPREHENSIVE STRATEGY VALIDATION")
print("="*80)
print(f"\nBacktest Sharpe: {results['performance_metrics']['sharpe_ratio']:.3f}")
print(f"Walk-Forward Efficiency: {wf_results['efficiency_ratio']:.3f}")
print(f"Monte Carlo 95% CI: [{mc_results.ci_sharpe_lower:.3f}, {mc_results.ci_sharpe_upper:.3f}]")
print(f"Win Rate: {trade_analysis.win_rate:.1%}")
print("\n" + results['report'])
```

## 📞 Support

For questions or issues with the backtesting framework:
- Review this documentation
- Check example notebooks in `/notebooks/`
- Consult academic references above

---

**Version**: 1.0.0
**Last Updated**: 2025-11-12
**Maintainer**: Energy Trading Quant Team
