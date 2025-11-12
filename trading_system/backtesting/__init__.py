"""
Backtesting Framework for Energy Trading Strategies

This package provides a comprehensive backtesting infrastructure including:
- Event-driven backtesting engine with realistic transaction costs
- Walk-forward validation for preventing overfitting
- Monte Carlo simulation for robustness testing
- Integrated performance attribution analysis
- Professional results analysis and reporting

Author: Energy Trading Quant Team
Date: 2025-11-12
"""

from .backtesting_engine import (
    BacktestingEngine,
    BacktestConfig,
    TransactionCostModel,
    Order,
    Position,
    OrderType,
    OrderSide
)

from .walk_forward import (
    WalkForwardAnalyzer,
    WalkForwardConfig,
    WalkForwardPeriod,
    WindowType,
    create_simple_strategy
)

from .monte_carlo import (
    MonteCarloSimulator,
    MonteCarloConfig,
    MonteCarloResults
)

from .integrated_backtesting import (
    IntegratedBacktester,
    IntegratedBacktestConfig
)

from .results_analyzer import (
    ResultsAnalyzer,
    TradeAnalysis
)

__all__ = [
    # Core backtesting
    'BacktestingEngine',
    'BacktestConfig',
    'TransactionCostModel',
    'Order',
    'Position',
    'OrderType',
    'OrderSide',

    # Walk-forward
    'WalkForwardAnalyzer',
    'WalkForwardConfig',
    'WalkForwardPeriod',
    'WindowType',
    'create_simple_strategy',

    # Monte Carlo
    'MonteCarloSimulator',
    'MonteCarloConfig',
    'MonteCarloResults',

    # Integrated
    'IntegratedBacktester',
    'IntegratedBacktestConfig',

    # Analysis
    'ResultsAnalyzer',
    'TradeAnalysis',
]

__version__ = '1.0.0'
