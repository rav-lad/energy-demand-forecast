"""
Trading Inference and Backtesting

This script:
1. Loads trained ML models (XGBoost, LightGBM, Random Forest)
2. Makes price forecasts on test data
3. Runs trading strategy backtests
4. Generates performance reports and visualizations

Usage:
    python run_trading_inference.py --model xgboost
    python run_trading_inference.py --model all
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent  # Project root
DATA_DIR = BASE_DIR / "data" / "modified_data"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def load_model(model_name, target='load'):
    """Load trained model and metadata.

    Args:
        model_name: Model name (xgboost, lightgbm, etc.)
        target: 'load' or 'price' - which target the model predicts
    """
    logger.info(f"Loading {model_name} model (target={target})...")

    # Adjust model directory based on target
    if target == 'price':
        model_dir = MODELS_DIR / f"{model_name}_price"
    else:
        model_dir = MODELS_DIR / model_name

    model_file = model_dir / "model.pkl"
    model = joblib.load(model_file)

    metrics_file = model_dir / "metrics.json"
    with open(metrics_file) as f:
        metrics = json.load(f)

    logger.info(f"  Model metrics: R²={metrics['r2']:.4f}, MAPE={metrics['mape']:.2f}%")

    return model, metrics


def load_test_data():
    """Load test data for inference."""
    logger.info("Loading test data...")

    df_test = pd.read_csv(DATA_DIR / "test_daily.csv")
    df_test['datetime'] = pd.to_datetime(df_test['datetime'])

    logger.info(f"  Test period: {df_test['datetime'].min()} to {df_test['datetime'].max()}")
    logger.info(f"  Test samples: {len(df_test)}")

    return df_test


def prepare_features(df_test, target='load'):
    """Prepare features for inference.

    Args:
        target: 'load' or 'price' - what the model predicts
    """
    if target == 'load':
        drop_cols = ['datetime', 'country', 'price_eur_mwh']
        target_col = 'load_mw'
    else:  # price
        drop_cols = ['datetime', 'country', 'load_mw', 'price_eur_mwh']
        target_col = 'price_eur_mwh'

    X_test = df_test.drop(columns=drop_cols + [target_col], errors='ignore')
    y_test = df_test[target_col]

    # For trading, we need actual price data
    prices = df_test['price_eur_mwh']
    dates = df_test['datetime']

    return X_test, y_test, prices, dates


def make_predictions(model, X_test):
    """Make predictions with the model."""
    logger.info("Generating predictions...")
    predictions = model.predict(X_test)
    return predictions


def calculate_price_forecasts(load_predictions, historical_data, train_data):
    """
    Convert load predictions to price forecasts using historical relationship.

    IMPORTANT: Uses ONLY training data to avoid look-ahead bias.

    In electricity markets, there's a strong relationship between demand (load)
    and price. We use historical data to model this relationship.
    """
    logger.info("Converting load predictions to price forecasts...")

    # CRITICAL FIX: Use ONLY training data to fit the relationship
    # Using test data would be look-ahead bias!
    load_train = train_data['load_mw'].values
    price_train = train_data['price_eur_mwh'].values

    # Fit simple linear model: price = a * load + b
    coef = np.polyfit(load_train, price_train, 1)

    # Predict prices from load predictions
    price_predictions = np.polyval(coef, load_predictions)

    logger.info(f"  Load-price correlation (train): {np.corrcoef(load_train, price_train)[0,1]:.3f}")
    logger.info(f"  Model: price = {coef[0]:.4f} * load + {coef[1]:.2f}")

    return price_predictions


def run_backtest(market_prices, price_forecasts, dates, model_name, initial_capital=100000):
    """
    Run realistic electricity trading backtest.

    Model: Simplified spread trading
    - When forecast > market + threshold: Buy forward at market price
    - Hold position for up to N days
    - Exit when forecast converges or max holding period
    - P&L = volume_MWh × (exit_price - entry_price)
    """
    logger.info(f"\nRunning backtest for {model_name}...")

    # Create price series
    market_prices_series = pd.Series(market_prices.values, index=dates)
    forecasts_series = pd.Series(price_forecasts, index=dates)

    # Params
    entry_threshold = 10.0  # EUR/MWh - higher threshold for more selective trades
    exit_threshold = 5.0    # EUR/MWh
    max_holding_period = 7  # days
    risk_per_trade = 0.01   # 1% of capital at risk per trade
    transaction_cost_pct = 0.001  # 0.1%
    max_position_value_pct = 0.05  # Max 5% of capital in one position
    stop_loss_pct = 0.02  # Stop loss at 2% of capital

    # Trading simulation
    capital = initial_capital
    equity_curve = []
    position = None  # {direction, entry_price, entry_date, volume_mwh}
    trades = []

    for i in range(len(market_prices_series)):
        date = dates.iloc[i]
        market_price = market_prices_series.iloc[i]
        forecast = forecasts_series.iloc[i]
        forecast_error = forecast - market_price

        # Check if we have a position
        if position is not None:
            days_held = (date - position['entry_date']).days

            # Calculate current P&L
            if position['direction'] == 'long':
                unrealized_pnl = position['volume_mwh'] * (market_price - position['entry_price'])
            else:
                unrealized_pnl = position['volume_mwh'] * (position['entry_price'] - market_price)

            # Exit conditions
            should_exit = False
            exit_reason = None

            # Stop loss: exit if losing more than stop_loss_pct of entry capital
            if unrealized_pnl < -(position['entry_capital'] * stop_loss_pct):
                should_exit = True
                exit_reason = "stop_loss"
            elif abs(forecast_error) < exit_threshold:
                should_exit = True
                exit_reason = "forecast_converged"
            elif days_held >= max_holding_period:
                should_exit = True
                exit_reason = "max_holding"

            if should_exit:
                # Calculate P&L
                if position['direction'] == 'long':
                    pnl_per_mwh = market_price - position['entry_price']
                else:  # short
                    pnl_per_mwh = position['entry_price'] - market_price

                gross_pnl = position['volume_mwh'] * pnl_per_mwh

                # Transaction costs (entry + exit)
                entry_cost = position['volume_mwh'] * position['entry_price'] * transaction_cost_pct
                exit_cost = position['volume_mwh'] * market_price * transaction_cost_pct
                net_pnl = gross_pnl - entry_cost - exit_cost

                # Update capital
                capital += net_pnl

                # Record trade
                trades.append({
                    'entry_date': position['entry_date'],
                    'exit_date': date,
                    'direction': position['direction'],
                    'entry_price': position['entry_price'],
                    'exit_price': market_price,
                    'volume_mwh': position['volume_mwh'],
                    'pnl': net_pnl,
                    'days_held': days_held,
                    'exit_reason': exit_reason
                })

                position = None

        # Entry logic (only if no position)
        if position is None:
            if forecast_error > entry_threshold:
                # Long signal: forecast significantly above market
                direction = 'long'
                # Calculate position size
                # Risk amount per trade (EUR)
                risk_amount = capital * risk_per_trade
                # Volume in MWh: risk per EUR/MWh of movement
                # Assume we risk entry_threshold EUR/MWh movement
                volume_mwh = risk_amount / entry_threshold

                # CRITICAL: Cap position value at max_position_value_pct of capital
                max_volume = (capital * max_position_value_pct) / market_price
                volume_mwh = min(volume_mwh, max_volume)

                position = {
                    'direction': direction,
                    'entry_price': market_price,
                    'entry_date': date,
                    'volume_mwh': volume_mwh,
                    'entry_capital': capital  # Store capital at entry for stop loss
                }

            elif forecast_error < -entry_threshold:
                # Short signal: forecast significantly below market
                direction = 'short'
                risk_amount = capital * risk_per_trade
                volume_mwh = risk_amount / entry_threshold

                # CRITICAL: Cap position value at max_position_value_pct of capital
                max_volume = (capital * max_position_value_pct) / market_price
                volume_mwh = min(volume_mwh, max_volume)

                position = {
                    'direction': direction,
                    'entry_price': market_price,
                    'entry_date': date,
                    'volume_mwh': volume_mwh,
                    'entry_capital': capital  # Store capital at entry for stop loss
                }

        equity_curve.append(capital)

    # Close any remaining position at end
    if position is not None:
        market_price = market_prices_series.iloc[-1]
        if position['direction'] == 'long':
            pnl_per_mwh = market_price - position['entry_price']
        else:
            pnl_per_mwh = position['entry_price'] - market_price

        gross_pnl = position['volume_mwh'] * pnl_per_mwh
        entry_cost = position['volume_mwh'] * position['entry_price'] * transaction_cost_pct
        exit_cost = position['volume_mwh'] * market_price * transaction_cost_pct
        net_pnl = gross_pnl - entry_cost - exit_cost
        capital += net_pnl

        trades.append({
            'entry_date': position['entry_date'],
            'exit_date': dates.iloc[-1],
            'direction': position['direction'],
            'entry_price': position['entry_price'],
            'exit_price': market_price,
            'volume_mwh': position['volume_mwh'],
            'pnl': net_pnl,
            'days_held': (dates.iloc[-1] - position['entry_date']).days,
            'exit_reason': 'end_of_period'
        })

    # Calculate metrics
    equity_series = pd.Series(equity_curve, index=dates)
    returns = equity_series.pct_change().fillna(0)

    total_return = (capital - initial_capital) / initial_capital
    annual_return = (1 + total_return) ** (252 / len(returns)) - 1

    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0

    # Drawdown
    cumulative = equity_series / initial_capital
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    # Trade statistics
    trades_df = pd.DataFrame(trades)
    num_trades = len(trades_df)

    if num_trades > 0:
        winning_trades = (trades_df['pnl'] > 0).sum()
        win_rate = winning_trades / num_trades
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if (num_trades - winning_trades) > 0 else 0
        profit_factor = abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum() / trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if trades_df[trades_df['pnl'] < 0]['pnl'].sum() != 0 else 0
    else:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
        profit_factor = 0

    metrics = {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "num_trades": int(num_trades),
        "win_rate": float(win_rate),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "profit_factor": float(profit_factor),
        "final_capital": float(capital),
    }

    # Create signals dataframe for compatibility
    signals = pd.DataFrame({
        'forecast_error': forecasts_series - market_prices_series,
        'equity': equity_series
    }, index=dates)

    return signals, metrics, trades_df


def visualize_trading_results(dates, market_prices, price_forecasts, signals, metrics, model_name):
    """Visualize trading strategy results."""
    logger.info("Generating trading visualizations...")

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Plot 1: Price forecasts vs market
    ax1 = axes[0]
    ax1.plot(dates, market_prices, label='Market Price', color='blue', linewidth=1.5, alpha=0.7)
    ax1.plot(dates, price_forecasts, label='Forecast', color='red', linewidth=1.5, alpha=0.7)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price (EUR/MWh)')
    ax1.set_title(f'{model_name.upper()} - Price Forecasts vs Market')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Forecast errors
    ax2 = axes[1]
    ax2.plot(dates, signals['forecast_error'], color='purple', linewidth=1, alpha=0.7, label='Forecast error')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    ax2.set_xlabel('Date')
    ax2.set_ylabel('Forecast Error (EUR/MWh)')
    ax2.set_title('Forecast Error (Forecast - Market)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Equity curve
    ax3 = axes[2]
    ax3.plot(dates, signals['equity'], color='darkgreen', linewidth=2, label='Equity')
    ax3.axhline(y=100000, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Initial capital')
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Capital (EUR)')
    ax3.set_title('Equity Curve')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Add performance metrics
    metrics_text = (
        f"Annual Return: {metrics['annual_return']:.2%}\n"
        f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
        f"Max Drawdown: {metrics['max_drawdown']:.2%}\n"
        f"Win Rate: {metrics['win_rate']:.2%}\n"
        f"Num Trades: {metrics['num_trades']}"
    )
    ax3.text(0.02, 0.98, metrics_text, transform=ax3.transAxes,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save figure
    output_file = OUTPUTS_DIR / f"trading_{model_name}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"  Saved: {output_file}")

    plt.close()


def print_backtest_summary(model_name, metrics):
    """Print backtest summary."""
    print("\n" + "="*70)
    print(f"BACKTEST RESULTS: {model_name.upper()}")
    print("="*70)
    print(f"Total Return:          {metrics['total_return']:>10.2%}")
    print(f"Annual Return:         {metrics['annual_return']:>10.2%}")
    print(f"Sharpe Ratio:          {metrics['sharpe_ratio']:>10.2f}")
    print(f"Max Drawdown:          {metrics['max_drawdown']:>10.2%}")
    print(f"Number of Trades:      {metrics['num_trades']:>10.0f}")
    print(f"Win Rate:              {metrics['win_rate']:>10.2%}")
    print(f"Avg Win:               {metrics['avg_win']:>10.2f} EUR")
    print(f"Avg Loss:              {metrics['avg_loss']:>10.2f} EUR")
    print(f"Profit Factor:         {metrics['profit_factor']:>10.2f}")
    print(f"Final Capital:         {metrics['final_capital']:>10.2f} EUR")
    print("="*70)


def save_backtest_results(model_name, metrics, signals):
    """Save backtest results to JSON."""
    results_file = OUTPUTS_DIR / f"backtest_{model_name}.json"

    # Convert metrics to JSON-serializable format
    metrics_serializable = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v
                           for k, v in metrics.items()}

    with open(results_file, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)

    logger.info(f"  Saved backtest results: {results_file}")

    # Save signals to CSV
    signals_file = OUTPUTS_DIR / f"signals_{model_name}.csv"
    signals.to_csv(signals_file)
    logger.info(f"  Saved trading signals: {signals_file}")


def main():
    """Main inference and backtesting pipeline."""
    parser = argparse.ArgumentParser(description="Trading inference and backtesting")
    parser.add_argument(
        "--model",
        choices=["xgboost", "lightgbm", "ridge", "random_forest", "all"],
        default="all",
        help="Model to use for inference"
    )
    parser.add_argument(
        "--target",
        choices=["load", "price"],
        default="price",
        help="Target variable: 'load' (predict load then convert) or 'price' (predict price directly)"
    )

    args = parser.parse_args()

    logger.info("="*70)
    logger.info("TRADING INFERENCE AND BACKTESTING")
    logger.info("="*70)

    # Determine models to test
    if args.model == "all":
        models_to_test = ["xgboost", "lightgbm", "random_forest", "ridge"]
    else:
        models_to_test = [args.model]

    # Load test data once
    df_test = load_test_data()
    X_test, y_test, market_prices, dates = prepare_features(df_test, target=args.target)

    # Load training data (needed for load→price conversion if using load models)
    if args.target == 'load':
        logger.info("Loading training data for load→price conversion...")
        df_train = pd.read_csv(DATA_DIR / "train_daily.csv")
        df_train['datetime'] = pd.to_datetime(df_train['datetime'])
        logger.info(f"  Train samples: {len(df_train)}")
    else:
        df_train = None

    # Store all results for comparison
    all_results = []

    for model_name in models_to_test:
        try:
            logger.info("\n" + "="*70)
            logger.info(f"Processing model: {model_name.upper()}")
            logger.info("="*70)

            # Load model
            model, model_metrics = load_model(model_name, target=args.target)

            # Make predictions
            predictions = make_predictions(model, X_test)

            # Get price forecasts
            if args.target == 'load':
                # Convert load predictions to price forecasts
                price_forecasts = calculate_price_forecasts(predictions, df_test, df_train)
            else:
                # Direct price predictions
                price_forecasts = predictions

            # Run backtest
            signals, backtest_metrics, trades_df = run_backtest(market_prices, price_forecasts, dates, model_name)

            # Print summary
            print_backtest_summary(model_name, backtest_metrics)

            # Save results
            save_backtest_results(model_name, backtest_metrics, signals)

            # Save trades
            trades_file = OUTPUTS_DIR / f"trades_{model_name}.csv"
            trades_df.to_csv(trades_file, index=False)
            logger.info(f"  Saved trades: {trades_file}")

            # Visualize
            visualize_trading_results(dates, market_prices, price_forecasts,
                                     signals, backtest_metrics, model_name)

            # Store for comparison
            all_results.append({
                'model': model_name,
                'ml_r2': model_metrics['r2'],
                'ml_mape': model_metrics['mape'],
                'total_return': backtest_metrics['total_return'],
                'annual_return': backtest_metrics['annual_return'],
                'sharpe_ratio': backtest_metrics['sharpe_ratio'],
                'max_drawdown': backtest_metrics['max_drawdown'],
                'win_rate': backtest_metrics['win_rate'],
                'num_trades': backtest_metrics['num_trades'],
                'profit_factor': backtest_metrics['profit_factor']
            })

        except Exception as e:
            logger.error(f"Failed to process {model_name}: {e}")
            import traceback
            traceback.print_exc()

    # Print comparison if multiple models
    if len(all_results) > 1:
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)

        df_comparison = pd.DataFrame(all_results)
        df_comparison = df_comparison.sort_values('sharpe_ratio', ascending=False)

        print(df_comparison.to_string(index=False))

        # Save comparison
        comparison_file = OUTPUTS_DIR / "model_comparison.csv"
        df_comparison.to_csv(comparison_file, index=False)
        logger.info(f"\nSaved comparison: {comparison_file}")

    logger.info("\n" + "="*70)
    logger.info("TRADING INFERENCE COMPLETED")
    logger.info("="*70)
    logger.info(f"\nResults saved in: {OUTPUTS_DIR}")


if __name__ == "__main__":
    main()
