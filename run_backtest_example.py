"""
Example: End-to-End Backtesting Pipeline

This script demonstrates the complete workflow:
1. Load historical data (demand predictions + market prices)
2. Initialize trading strategy
3. Run backtest
4. Analyze results

Usage:
    python run_backtest_example.py
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from trading_system.strategies.demand_price_arbitrage import (
    DemandPriceArbitrageStrategy,
)
from trading_system.backtesting.backtest_engine import BacktestEngine
from trading_system.utils.config_loader import get_config

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_sample_data():
    """
    Load or generate sample data for demonstration.

    In production, this would load:
    - Predicted demand from your trained models
    - Market prices from ENTSO-E data
    - Renewable production data
    """
    logger.info("Loading sample data...")

    # Generate synthetic data for demonstration
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    n = len(dates)

    # Synthetic demand with weekly and annual seasonality
    t = np.arange(n)
    demand = (
        5000  # Base load
        + 1000 * np.sin(2 * np.pi * t / 365)  # Annual seasonality
        + 300 * np.sin(2 * np.pi * t / 7)  # Weekly seasonality
        + np.random.randn(n) * 200  # Noise
    )

    # Synthetic renewable share (higher in summer)
    renewable_share = (
        0.35
        + 0.15 * np.sin(2 * np.pi * t / 365 + np.pi / 2)
        + np.random.randn(n) * 0.05
    )
    renewable_share = np.clip(renewable_share, 0, 1)

    # Synthetic prices (correlated with residual load)
    residual_load = demand * (1 - renewable_share)
    prices = (
        40  # Base price
        + (residual_load - residual_load.mean()) / residual_load.std() * 10
        + np.random.randn(n) * 5
    )
    prices = np.clip(prices, 10, 150)  # Reasonable price bounds

    # Create DataFrame
    data = pd.DataFrame(
        {
            "predicted_demand": demand,
            "renewable_share": renewable_share,
            "residual_load": residual_load,
            "price_FR": prices,
        },
        index=dates,
    )

    logger.info(f"Loaded {len(data)} days of data")
    logger.info(f"  Demand range: {demand.min():.0f} - {demand.max():.0f} MW")
    logger.info(f"  Price range: {prices.min():.2f} - {prices.max():.2f} EUR/MWh")
    logger.info(f"  Avg renewable share: {renewable_share.mean():.2%}")

    return data


def run_backtest(data, config):
    """
    Run backtest with demand-price arbitrage strategy.

    Args:
        data: DataFrame with historical data
        config: Configuration object
    """
    logger.info("=" * 70)
    logger.info("STARTING BACKTEST")
    logger.info("=" * 70)

    # Split data into calibration and test sets
    split_date = "2023-06-30"
    calibration_data = data[:split_date]
    test_data = data[split_date:]

    logger.info(f"\nData split:")
    logger.info(
        f"  Calibration: {calibration_data.index[0].date()} to {calibration_data.index[-1].date()}"
    )
    logger.info(
        f"  Test:        {test_data.index[0].date()} to {test_data.index[-1].date()}"
    )

    # Initialize strategy
    strategy_config = config.get("trading.demand_price_arbitrage", {})

    strategy = DemandPriceArbitrageStrategy(
        buy_demand_threshold=strategy_config.get("signals", {}).get(
            "buy_threshold", 0.95
        ),
        sell_demand_threshold=strategy_config.get("signals", {}).get(
            "sell_threshold", 0.25
        ),
        renewable_high_threshold=strategy_config.get("signals", {}).get(
            "renewable_threshold_high", 0.7
        ),
        renewable_low_threshold=strategy_config.get("signals", {}).get(
            "renewable_threshold_low", 0.3
        ),
        base_position_size=strategy_config.get("position_sizing", {}).get(
            "base_size", 1000
        ),
        use_confidence_scaling=strategy_config.get("position_sizing", {}).get(
            "scale_with_confidence", True
        ),
    )

    # Calibrate strategy on historical data
    logger.info("\n" + "-" * 70)
    logger.info("STEP 1: Calibrating Strategy")
    logger.info("-" * 70)

    strategy.calibrate(
        calibration_data["predicted_demand"], calibration_data["renewable_share"]
    )

    # Generate signals
    logger.info("\n" + "-" * 70)
    logger.info("STEP 2: Generating Trading Signals")
    logger.info("-" * 70)

    signals_df = strategy.generate_signals(
        test_data["predicted_demand"], test_data["renewable_share"]
    )

    # Analyze signal quality (how well signals predict price movements)
    logger.info("\n" + "-" * 70)
    logger.info("STEP 3: Analyzing Signal Quality")
    logger.info("-" * 70)

    signal_quality = strategy.analyze_signal_quality(signals_df, test_data["price_FR"])

    # Initialize backtest engine
    logger.info("\n" + "-" * 70)
    logger.info("STEP 4: Running Backtest")
    logger.info("-" * 70)

    trading_config = config.get("trading.general", {})

    engine = BacktestEngine(
        initial_capital=trading_config.get("initial_capital", 100000),
        transaction_cost=trading_config.get("transaction_cost", 0.001),
        slippage=trading_config.get("slippage_value", 0.0005),
        max_position_size=trading_config.get("max_position_size", 10000),
        max_positions=trading_config.get("max_positions", 5),
    )

    # Run backtest
    engine.process_signals(
        test_data,
        signals_df["signal"],
        price_column="price_FR",
        quantity_column="position_size",
    )

    # Print results
    logger.info("\n" + "-" * 70)
    logger.info("STEP 5: Results Analysis")
    logger.info("-" * 70)

    engine.print_results()

    # Return results for further analysis
    return {
        "engine": engine,
        "signals_df": signals_df,
        "signal_quality": signal_quality,
        "test_data": test_data,
    }


def save_results(results, output_dir="outputs/backtests"):
    """
    Save backtest results to files.

    Args:
        results: Dictionary with backtest results
        output_dir: Output directory path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

    # Save equity curve
    equity_file = output_dir / f"equity_curve_{timestamp}.csv"
    results["engine"].equity_curve.to_csv(equity_file)
    logger.info(f"Saved equity curve to {equity_file}")

    # Save trades
    trades_data = []
    for trade in results["engine"].trades:
        trades_data.append(
            {
                "timestamp": trade.timestamp,
                "action": trade.action,
                "quantity": trade.quantity,
                "price": trade.price,
                "value": trade.value,
                "region": trade.region,
            }
        )

    trades_df = pd.DataFrame(trades_data)
    trades_file = output_dir / f"trades_{timestamp}.csv"
    trades_df.to_csv(trades_file, index=False)
    logger.info(f"Saved trades to {trades_file}")

    # Save metrics
    metrics = results["engine"].calculate_metrics()
    metrics_df = pd.DataFrame([metrics])
    metrics_file = output_dir / f"metrics_{timestamp}.csv"
    metrics_df.to_csv(metrics_file, index=False)
    logger.info(f"Saved metrics to {metrics_file}")

    # Save signals
    signals_file = output_dir / f"signals_{timestamp}.csv"
    results["signals_df"].to_csv(signals_file)
    logger.info(f"Saved signals to {signals_file}")

    logger.info(f"\nAll results saved to {output_dir}")


def main():
    """Main execution function."""
    print("\n" + "=" * 70)
    print("ENERGY TRADING BACKTEST - DEMAND-PRICE ARBITRAGE STRATEGY")
    print("=" * 70 + "\n")

    try:
        # Load configuration
        config = get_config()
        logger.info(
            f"Configuration loaded: {config.get('project.name')} v{config.get('project.version')}"
        )

        # Load data
        data = load_sample_data()

        # Run backtest
        results = run_backtest(data, config)

        # Save results
        save_results(results)

        print("\n" + "=" * 70)
        print("BACKTEST COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nNext steps:")
        print("1. Review the results in outputs/backtests/")
        print("2. Adjust strategy parameters in config.yaml")
        print("3. Test with real data from ENTSO-E")
        print("4. Develop additional strategies")
        print("\n" + "=" * 70 + "\n")

    except Exception as e:
        logger.error(f"Backtest failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
