"""
Daily production pipeline for energy price forecasting and trading.

This script runs the complete pipeline:
1. Load latest data
2. Train ensemble model (walk-forward)
3. Generate predictions
4. Execute trading strategy
5. Report performance
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from production.config import (
    UNIFIED_DATASET_PATH,
    FEATURE_FILE,
    MODELS_TO_TRAIN,
    ENSEMBLE_WEIGHTS,
    INITIAL_TRAIN_DAYS,
    FORECAST_HORIZON_HOURS,
    TRADING_STRATEGY,
    INITIAL_CAPITAL,
    TRADE_VOLUME,
    REPORTS_DIR,
    TUNE_FREQUENCY_DAYS,
    TUNING_ITERATIONS,
    ADAPTIVE_MODEL_TYPE
)
from model.price_forecasting.bayesian_tuner import AdaptiveWalkForwardTuner
from production.trading_strategy import backtest_strategy


def load_data() -> tuple[pd.DataFrame, list[str]]:
    """
    Load unified dataset and feature list.

    Returns:
        df: DataFrame with all data
        feature_cols: List of feature column names
    """
    print("="*80)
    print("1. LOADING DATA")
    print("="*80)

    # Load dataset
    df = pd.read_csv(UNIFIED_DATASET_PATH)
    df['datetime_hour'] = pd.to_datetime(df['datetime_hour'])

    print(f"[OK] Dataset loaded: {len(df):,} rows")
    print(f"   Date range: {df['datetime_hour'].min()} to {df['datetime_hour'].max()}")

    # Load features
    with open(FEATURE_FILE, 'r') as f:
        feature_cols = [line.strip() for line in f.readlines()]

    print(f"[OK] Features loaded: {len(feature_cols)} columns")

    return df, feature_cols


def train_ensemble(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    """
    Train ensemble model with walk-forward validation.

    Args:
        df: DataFrame with data
        feature_cols: List of feature columns

    Returns:
        predictions_df: DataFrame with predictions
        metrics: Overall metrics
    """
    print("\n" + "="*80)
    print("2. TRAINING ENSEMBLE MODEL (Walk-Forward)")
    print("="*80)

    print(f"Model: {ADAPTIVE_MODEL_TYPE} (Adaptive Bayesian Tuning)")
    print(f"Initial train: {INITIAL_TRAIN_DAYS} days")
    print(f"Forecast horizon: {FORECAST_HORIZON_HOURS}h")
    print(f"Tuning frequency: {TUNE_FREQUENCY_DAYS} days")
    print(f"Tuning iterations: {TUNING_ITERATIONS}")

    tuner = AdaptiveWalkForwardTuner(
        model_type=ADAPTIVE_MODEL_TYPE,
        tune_frequency_days=TUNE_FREQUENCY_DAYS,
        tuning_iterations=TUNING_ITERATIONS,
        random_state=42
    )

    predictions_df, metrics, _ = tuner.walk_forward_with_adaptive_tuning(
        df=df,
        feature_cols=feature_cols,
        target_col='price',
        initial_train_days=INITIAL_TRAIN_DAYS,
        forecast_horizon_hours=FORECAST_HORIZON_HOURS,
        retrain_frequency_days=1,
        verbose=True
    )
    
    # Calculate direction accuracy (since tuner might not return it directly in standard format)
    if 'direction_accuracy' not in metrics:
        from model.price_forecasting.metrics import calculate_direction_accuracy
        metrics['direction_accuracy'] = calculate_direction_accuracy(
            predictions_df['actual'].values, 
            predictions_df['predicted'].values
        )

    print(f"\n[OK] Ensemble trained")
    print(f"   Predictions: {len(predictions_df):,} hours")
    print(f"   MAE: {metrics['mae']:.2f} EUR/MWh")
    print(f"   RMSE: {metrics['rmse']:.2f} EUR/MWh")
    print(f"   R²: {metrics['r2']:.3f}")
    print(f"   Direction accuracy: {metrics['direction_accuracy']:.2f}%")

    return predictions_df, metrics


def execute_trading(predictions_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Execute trading strategy.

    Args:
        predictions_df: DataFrame with predictions

    Returns:
        trades_df: DataFrame with executed trades
        trading_metrics: Trading performance metrics
    """
    print("\n" + "="*80)
    print("3. EXECUTING TRADING STRATEGY")
    print("="*80)

    print(f"Strategy: {TRADING_STRATEGY}")
    print(f"Initial capital: {INITIAL_CAPITAL} EUR")
    print(f"Trade volume: {TRADE_VOLUME} MWh")

    trades_df, trading_metrics = backtest_strategy(
        predictions_df=predictions_df,
        strategy=TRADING_STRATEGY,
        initial_capital=INITIAL_CAPITAL,
        trade_volume=TRADE_VOLUME
    )

    print(f"\n[OK] Trading executed")
    print(f"   Trades: {trading_metrics['n_trades']} days")
    print(f"   Total PnL: {trading_metrics['total_pnl']:.2f} EUR")
    print(f"   Return: {trading_metrics['return_pct']:.2f}%")
    print(f"   Win rate: {trading_metrics['win_rate']:.2f}%")
    print(f"   Avg profit/day: {trading_metrics['avg_profit']:.2f} EUR")
    print(f"   Sharpe ratio: {trading_metrics['sharpe_ratio']:.3f}")
    print(f"   Max drawdown: {trading_metrics['max_drawdown']:.2f} EUR")

    return trades_df, trading_metrics


def save_results(predictions_df: pd.DataFrame, metrics: dict, trades_df: pd.DataFrame, trading_metrics: dict):
    """
    Save all results to disk.

    Args:
        predictions_df: Price predictions
        metrics: Prediction metrics
        trades_df: Executed trades
        trading_metrics: Trading metrics
    """
    print("\n" + "="*80)
    print("4. SAVING RESULTS")
    print("="*80)

    # Create output directory
    output_dir = REPORTS_DIR / "ensemble"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save predictions
    predictions_path = output_dir / "ensemble_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"[OK] Predictions saved: {predictions_path}")

    # Save prediction metrics
    metrics_path = output_dir / "ensemble_metrics.txt"
    with open(metrics_path, 'w') as f:
        f.write("Ensemble Model - Prediction Metrics\n")
        f.write("="*80 + "\n\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
    print(f"[OK] Metrics saved: {metrics_path}")

    # Save trades
    trades_path = output_dir / "trading_results.csv"
    trades_df.to_csv(trades_path, index=False)
    print(f"[OK] Trades saved: {trades_path}")

    # Save trading metrics
    trading_metrics_path = output_dir / "trading_metrics.txt"
    with open(trading_metrics_path, 'w') as f:
        f.write("Trading Performance Metrics\n")
        f.write("="*80 + "\n\n")
        for key, value in trading_metrics.items():
            f.write(f"{key}: {value}\n")
    print(f"[OK] Trading metrics saved: {trading_metrics_path}")


def generate_report(metrics: dict, trading_metrics: dict):
    """
    Generate final summary report.

    Args:
        metrics: Prediction metrics
        trading_metrics: Trading metrics
    """
    print("\n" + "="*80)
    print("FINAL REPORT")
    print("="*80)

    print("\n[PREDICTION PERFORMANCE]")
    print(f"  MAE:                  {metrics['mae']:>10.2f} EUR/MWh")
    print(f"  RMSE:                 {metrics['rmse']:>10.2f} EUR/MWh")
    print(f"  R²:                   {metrics['r2']:>10.3f}")
    print(f"  Direction Accuracy:   {metrics['direction_accuracy']:>10.2f}%")

    print("\n[TRADING PERFORMANCE]")
    print(f"  Total PnL:            {trading_metrics['total_pnl']:>10.2f} EUR")
    print(f"  Return:               {trading_metrics['return_pct']:>10.2f}%")
    print(f"  Win Rate:             {trading_metrics['win_rate']:>10.2f}%")
    print(f"  Avg Profit/Day:       {trading_metrics['avg_profit']:>10.2f} EUR")
    print(f"  Sharpe Ratio:         {trading_metrics['sharpe_ratio']:>10.3f}")
    print(f"  Max Drawdown:         {trading_metrics['max_drawdown']:>10.2f} EUR")

    print("\n" + "="*80)
    print(f"Pipeline completed successfully at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)


def main():
    """Run the complete pipeline."""
    try:
        # 1. Load data
        df, feature_cols = load_data()

        # 2. Train ensemble model
        predictions_df, metrics = train_ensemble(df, feature_cols)

        # 3. Execute trading
        trades_df, trading_metrics = execute_trading(predictions_df)

        # 4. Save results
        save_results(predictions_df, metrics, trades_df, trading_metrics)

        # 5. Generate report
        generate_report(metrics, trading_metrics)

        return 0

    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
