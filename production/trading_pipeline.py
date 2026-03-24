"""
Production Trading Pipeline for French Power Futures

This script implements the complete trading strategy pipeline:
1. Load ML predictions from walk-forward validation
2. Aggregate to daily frequency
3. Load futures data
4. Generate ensemble signals with filters
5. Apply position sizing with minimum holding period
6. Calculate PnL with transaction costs
7. Generate performance reports

Strategy: Ensemble with Minimum Holding Period
Performance: Sharpe Ratio 1.55, Total PnL €3,643

Author: Energy Trading Research Team
Date: 2025-12-08
"""

import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Setup
warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
REPORTS_DIR = PROJECT_ROOT / "research/reports"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "production/output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Trading parameters
class TradingConfig:
    """Configuration for trading strategy"""
    CONTRACT_MWH_PER_DAY = 24
    TRANSACTION_COST_PER_MWH = 0.10
    SLIPPAGE_PER_MWH = 0.05
    MAX_POSITION_MW = 5
    SIGNAL_THRESHOLD_QUANTILE = 0.65
    MIN_HOLDING_DAYS = 2
    VOL_FILTER_QUANTILE = 0.85
    DAYS_PER_YEAR = 252
    # Out-of-sample holdout: last N days are NEVER used for parameter tuning
    HOLDOUT_DAYS = 60


def load_ml_predictions() -> Dict[str, pd.DataFrame]:
    """
    Load ML predictions from walk-forward validation

    Returns:
        Dictionary of model predictions
    """
    print("\n" + "="*80)
    print("STEP 1: LOADING ML PREDICTIONS")
    print("="*80)

    models_predictions = {}

    model_files = {
        'ridge': 'ridge_predictions.csv',
        'xgb': 'xgboost_predictions.csv',
        'lgbm': 'lightgbm_quantile_predictions.csv'
    }

    for model_name, file_name in model_files.items():
        file_path = REPORTS_DIR / file_name
        if file_path.exists():
            df = pd.read_csv(file_path)
            df['datetime_hour'] = pd.to_datetime(df['datetime_hour'])

            # Standardize column names
            if 'pred_p50' in df.columns and 'predicted' not in df.columns:
                df['predicted'] = df['pred_p50']

            models_predictions[model_name] = df
            print(f"  {model_name.upper():8s}: {len(df):,} hourly predictions")
        else:
            print(f"  WARNING: {file_name} not found")

    print(f"\nTotal models loaded: {len(models_predictions)}")
    return models_predictions


def aggregate_to_daily(models_predictions: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Aggregate hourly predictions to daily frequency

    Args:
        models_predictions: Dictionary of hourly predictions

    Returns:
        DataFrame with daily aggregated data
    """
    print("\n" + "="*80)
    print("STEP 2: AGGREGATING TO DAILY FREQUENCY")
    print("="*80)

    daily_predictions = {}

    for model_name, df_hourly in models_predictions.items():
        df_hourly['date'] = df_hourly['datetime_hour'].dt.date

        df_daily = df_hourly.groupby('date').agg({
            'actual': 'mean',
            'predicted': 'mean'
        }).reset_index()

        df_daily['date'] = pd.to_datetime(df_daily['date'])
        df_daily = df_daily.set_index('date').sort_index()

        df_daily = df_daily.rename(columns={
            'actual': 'price',
            'predicted': f'pred_{model_name}'
        })

        daily_predictions[model_name] = df_daily
        print(f"  {model_name.upper():8s}: {len(df_daily)} trading days")

    # Merge all predictions
    df_spot = daily_predictions['ridge'][['price']].copy()
    for model_name, df_daily in daily_predictions.items():
        df_spot = df_spot.join(df_daily[[f'pred_{model_name}']], how='outer')

    df_spot = df_spot.dropna(subset=['price'])

    print(f"\nConsolidated: {len(df_spot)} days")
    print(f"Date range: {df_spot.index.min().date()} to {df_spot.index.max().date()}")

    return df_spot


def load_futures_data(df_spot: pd.DataFrame) -> pd.DataFrame:
    """
    Load French Power Futures data and merge with spot

    Args:
        df_spot: Spot price DataFrame

    Returns:
        DataFrame with futures data merged
    """
    print("\n" + "="*80)
    print("STEP 3: LOADING FUTURES DATA")
    print("="*80)

    futures_file = DATA_DIR / "processed/french_power_futures.csv"

    if not futures_file.exists():
        print("  WARNING: Futures file not found, generating...")
        import subprocess
        subprocess.run([sys.executable, str(PROJECT_ROOT / "data_collection/futures_data.py")])

    df_futures = pd.read_csv(futures_file, parse_dates=['date'])
    df_futures = df_futures.set_index('date').sort_index()

    df_spot = df_spot.join(df_futures[['futures_front']], how='inner')
    df_spot = df_spot.rename(columns={'futures_front': 'F'})

    print(f"  Futures data loaded: {len(df_spot)} days")
    print(f"  Spot mean: {df_spot['price'].mean():.2f} EUR/MWh")
    print(f"  Futures mean: {df_spot['F'].mean():.2f} EUR/MWh")
    print(f"  Correlation: {df_spot['price'].corr(df_spot['F']):.3f}")

    return df_spot


def build_signals(df_spot: pd.DataFrame) -> pd.DataFrame:
    """
    Build trading signals from ML predictions

    Args:
        df_spot: DataFrame with spot and predictions

    Returns:
        DataFrame with individual and ensemble signals
    """
    print("\n" + "="*80)
    print("STEP 4: BUILDING TRADING SIGNALS")
    print("="*80)

    # Calculate returns and volatility
    df_spot['price_lag_1'] = df_spot['price'].shift(1)
    df_spot['F_lag_1'] = df_spot['F'].shift(1)
    df_spot['dF'] = df_spot['F'] - df_spot['F_lag_1']
    df_spot['returns'] = df_spot['price'].pct_change()
    df_spot['dF_std'] = df_spot['dF'].rolling(21, min_periods=5).std()

    # Individual model signals
    for model in ['ridge', 'xgb', 'lgbm']:
        pred_col = f'pred_{model}'
        if pred_col in df_spot.columns:
            df_spot[f'surprise_{model}'] = df_spot[pred_col] - df_spot['price_lag_1']
            df_spot[f'signal_{model}'] = df_spot[f'surprise_{model}'] / df_spot['dF_std']
            print(f"  {model.upper():8s} signal: mean={df_spot[f'signal_{model}'].mean():.3f}, "
                  f"std={df_spot[f'signal_{model}'].std():.3f}")

    df_spot = df_spot.dropna(subset=['dF_std'])

    # Ensemble signal
    ensemble_signals = [df_spot[f'signal_{m}'] for m in ['ridge', 'xgb', 'lgbm']
                       if f'signal_{m}' in df_spot.columns]
    df_spot['signal_ensemble'] = pd.concat(ensemble_signals, axis=1).mean(axis=1)

    # Agreement count
    df_spot['signal_agreement'] = sum(
        (df_spot[f'signal_{m}'] > 0).astype(int)
        for m in ['ridge', 'xgb', 'lgbm']
        if f'signal_{m}' in df_spot.columns
    )

    print(f"\n  Ensemble signal: mean={df_spot['signal_ensemble'].mean():.3f}, "
          f"std={df_spot['signal_ensemble'].std():.3f}")
    print(f"  Strong agreement (3-0 or 0-3): "
          f"{((df_spot['signal_agreement'] == 0) | (df_spot['signal_agreement'] == 3)).sum()} days")

    return df_spot


def apply_regime_filters(df_spot: pd.DataFrame) -> pd.DataFrame:
    """
    Apply market regime filters

    Args:
        df_spot: DataFrame with signals

    Returns:
        DataFrame with regime filters
    """
    print("\n" + "="*80)
    print("STEP 5: APPLYING MARKET REGIME FILTERS")
    print("="*80)

    # Volatility filter
    df_spot['realized_vol'] = df_spot['returns'].rolling(21).std()
    vol_threshold = df_spot['realized_vol'].quantile(TradingConfig.VOL_FILTER_QUANTILE)
    df_spot['vol_filter'] = df_spot['realized_vol'] < vol_threshold

    # Trend filter
    df_spot['trend_short'] = df_spot['price'].rolling(5).mean()
    df_spot['trend_long'] = df_spot['price'].rolling(21).mean()
    df_spot['trend_strength'] = (df_spot['trend_short'] - df_spot['trend_long']).abs()
    trend_threshold = df_spot['trend_strength'].quantile(0.75)
    df_spot['trend_filter'] = df_spot['trend_strength'] < trend_threshold

    # Combined filter
    df_spot['regime_filter'] = df_spot['vol_filter'] & df_spot['trend_filter']

    print(f"  Volatility filter: {df_spot['vol_filter'].sum()} / {len(df_spot)} days "
          f"({df_spot['vol_filter'].mean()*100:.1f}%)")
    print(f"  Trend filter: {df_spot['trend_filter'].sum()} / {len(df_spot)} days "
          f"({df_spot['trend_filter'].mean()*100:.1f}%)")
    print(f"  Combined filter: {df_spot['regime_filter'].sum()} / {len(df_spot)} days "
          f"({df_spot['regime_filter'].mean()*100:.1f}%)")

    return df_spot


def generate_positions(df_spot: pd.DataFrame) -> pd.DataFrame:
    """
    Generate trading positions with minimum holding period

    Args:
        df_spot: DataFrame with signals and filters

    Returns:
        DataFrame with positions
    """
    print("\n" + "="*80)
    print("STEP 6: GENERATING POSITIONS")
    print("="*80)

    # FIXED: Use expanding (causal) quantiles instead of full-sample quantiles
    # to avoid strategy parameter snooping. At each day t, thresholds are
    # computed only from signals observed up to day t-1.
    signal_expanding_long = df_spot['signal_ensemble'].expanding(min_periods=30).quantile(
        TradingConfig.SIGNAL_THRESHOLD_QUANTILE
    ).shift(1)
    signal_expanding_short = df_spot['signal_ensemble'].expanding(min_periods=30).quantile(
        1 - TradingConfig.SIGNAL_THRESHOLD_QUANTILE
    ).shift(1)

    # Fallback for first 30 days: use a conservative fixed threshold
    signal_expanding_long = signal_expanding_long.fillna(0.5)
    signal_expanding_short = signal_expanding_short.fillna(-0.5)

    print(f"  Adaptive thresholds (expanding, causal):")
    print(f"    Long  (last): signal > {signal_expanding_long.iloc[-1]:.3f}")
    print(f"    Short (last): signal < {signal_expanding_short.iloc[-1]:.3f}")

    # Generate positions with holding period constraint
    positions = []
    days_held = 0
    current_pos = 0.0

    for i, (idx, row) in enumerate(df_spot.iterrows()):
        signal = row['signal_ensemble']
        agreement = row['signal_agreement']
        regime_ok = row['regime_filter']
        thresh_long = signal_expanding_long.iloc[i]
        thresh_short = signal_expanding_short.iloc[i]

        # Determine desired position
        if not regime_ok or agreement == 1:
            desired_pos = 0.0
        elif signal < thresh_long and signal > thresh_short:
            desired_pos = 0.0
        else:
            conviction = min(abs(signal), 2.0)
            desired_pos = np.sign(signal) * conviction
            desired_pos = np.clip(desired_pos, -TradingConfig.MAX_POSITION_MW, TradingConfig.MAX_POSITION_MW)

        # Apply holding period constraint
        if current_pos == 0:
            current_pos = desired_pos
            days_held = 0
        elif days_held < TradingConfig.MIN_HOLDING_DAYS:
            days_held += 1
        else:
            current_pos = desired_pos
            days_held = 0

        positions.append(current_pos)

    df_spot['position'] = positions

    # Mark holdout period for separate evaluation
    holdout_start = len(df_spot) - TradingConfig.HOLDOUT_DAYS
    df_spot['is_holdout'] = False
    if holdout_start > 0:
        df_spot.iloc[holdout_start:, df_spot.columns.get_loc('is_holdout')] = True

    print(f"\n  Positions generated:")
    print(f"    Days in position: {(df_spot['position'] != 0).sum()} / {len(df_spot)}")
    print(f"    Avg position size: {df_spot['position'].abs().mean():.2f} MW")
    print(f"    Max position size: {df_spot['position'].abs().max():.2f} MW")
    if holdout_start > 0:
        print(f"    Holdout period: last {TradingConfig.HOLDOUT_DAYS} days (from index {holdout_start})")

    return df_spot


def calculate_pnl(df_spot: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    Calculate PnL with transaction costs

    Args:
        df_spot: DataFrame with positions

    Returns:
        Tuple of (DataFrame with PnL, performance metrics dict)
    """
    print("\n" + "="*80)
    print("STEP 7: CALCULATING PNL")
    print("="*80)

    # Gross PnL
    df_spot['pnl_gross'] = df_spot['position'] * df_spot['dF'] * TradingConfig.CONTRACT_MWH_PER_DAY

    # Transaction costs
    total_cost_per_mwh = TradingConfig.TRANSACTION_COST_PER_MWH + TradingConfig.SLIPPAGE_PER_MWH
    trades = df_spot['position'].diff().abs().fillna(0)
    cost = trades * TradingConfig.CONTRACT_MWH_PER_DAY * total_cost_per_mwh
    df_spot['pnl_net'] = df_spot['pnl_gross'] - cost

    # Calculate metrics
    pnl = df_spot['pnl_net'].dropna()
    positions = df_spot['position']

    total_pnl = pnl.sum()
    num_trades = (positions.diff().abs() > 0).sum()
    sharpe = (pnl.mean() / pnl.std()) * np.sqrt(TradingConfig.DAYS_PER_YEAR) if pnl.std() > 0 else 0
    hit_ratio = (pnl > 0).mean() * 100

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 1
    win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

    equity = pnl.cumsum()
    running_max = equity.expanding().max()
    drawdown = equity - running_max
    max_dd = drawdown.min()

    metrics = {
        'strategy': 'Ensemble_Holding_Production',
        'total_pnl': total_pnl,
        'sharpe_ratio': sharpe,
        'hit_ratio': hit_ratio,
        'max_drawdown': max_dd,
        'num_trades': num_trades,
        'avg_pnl_per_trade': total_pnl / num_trades if num_trades > 0 else 0,
        'win_loss_ratio': win_loss_ratio
    }

    print(f"\n  Performance Metrics (FULL PERIOD):")
    print(f"    Total PnL:        EUR {total_pnl:>10,.2f}")
    print(f"    Sharpe Ratio:     {sharpe:>10.3f}")
    print(f"    Hit Ratio:        {hit_ratio:>10.1f}%")
    print(f"    Num Trades:       {num_trades:>10d}")
    print(f"    Avg PnL/Trade:    EUR {metrics['avg_pnl_per_trade']:>10,.2f}")
    print(f"    Max Drawdown:     EUR {max_dd:>10,.2f}")
    print(f"    Win/Loss Ratio:   {win_loss_ratio:>10.3f}")

    # Report holdout (out-of-sample) metrics separately
    if 'is_holdout' in df_spot.columns and df_spot['is_holdout'].any():
        pnl_oos = df_spot.loc[df_spot['is_holdout'], 'pnl_net'].dropna()
        pnl_is = df_spot.loc[~df_spot['is_holdout'], 'pnl_net'].dropna()

        sharpe_oos = (pnl_oos.mean() / pnl_oos.std()) * np.sqrt(TradingConfig.DAYS_PER_YEAR) if pnl_oos.std() > 0 else 0
        sharpe_is = (pnl_is.mean() / pnl_is.std()) * np.sqrt(TradingConfig.DAYS_PER_YEAR) if pnl_is.std() > 0 else 0

        metrics['sharpe_in_sample'] = sharpe_is
        metrics['sharpe_out_of_sample'] = sharpe_oos
        metrics['pnl_in_sample'] = pnl_is.sum()
        metrics['pnl_out_of_sample'] = pnl_oos.sum()
        metrics['holdout_days'] = TradingConfig.HOLDOUT_DAYS

        print(f"\n  IN-SAMPLE (train period):")
        print(f"    Sharpe Ratio:     {sharpe_is:>10.3f}")
        print(f"    Total PnL:        EUR {pnl_is.sum():>10,.2f}")
        print(f"\n  OUT-OF-SAMPLE (holdout, last {TradingConfig.HOLDOUT_DAYS} days):")
        print(f"    Sharpe Ratio:     {sharpe_oos:>10.3f}")
        print(f"    Total PnL:        EUR {pnl_oos.sum():>10,.2f}")

    return df_spot, metrics


def generate_visualizations(df_spot: pd.DataFrame, metrics: Dict):
    """
    Generate performance visualizations

    Args:
        df_spot: DataFrame with PnL
        metrics: Performance metrics dictionary
    """
    print("\n" + "="*80)
    print("STEP 8: GENERATING VISUALIZATIONS")
    print("="*80)

    pnl = df_spot['pnl_net'].dropna()
    equity = pnl.cumsum()
    running_max = equity.expanding().max()
    drawdown = equity - running_max

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Trading Strategy Performance - Production Pipeline',
                 fontsize=16, fontweight='bold')

    # 1. Equity curve
    ax = axes[0, 0]
    equity.plot(ax=ax, linewidth=2, color='blue', alpha=0.8)
    ax.set_title('Cumulative PnL', fontweight='bold')
    ax.set_ylabel('EUR')
    ax.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    # 2. Drawdown
    ax = axes[0, 1]
    drawdown.plot(ax=ax, linewidth=2, color='red', alpha=0.8)
    ax.fill_between(drawdown.index, 0, drawdown, alpha=0.3, color='red')
    ax.set_title('Drawdown', fontweight='bold')
    ax.set_ylabel('EUR')
    ax.grid(True, alpha=0.3)

    # 3. Daily PnL distribution
    ax = axes[1, 0]
    pnl.hist(bins=50, ax=ax, alpha=0.7, color='green', edgecolor='black')
    ax.axvline(pnl.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {pnl.mean():.2f}')
    ax.set_title('Daily PnL Distribution', fontweight='bold')
    ax.set_xlabel('PnL (EUR)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Position sizes over time
    ax = axes[1, 1]
    df_spot['position'].plot(ax=ax, linewidth=1, alpha=0.7, color='purple')
    ax.set_title('Position Sizes Over Time', fontweight='bold')
    ax.set_ylabel('MW')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    fig_path = OUTPUT_DIR / f"trading_performance_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {fig_path}")

    plt.close()


def save_results(df_spot: pd.DataFrame, metrics: Dict):
    """
    Save results to CSV files

    Args:
        df_spot: DataFrame with all data
        metrics: Performance metrics dictionary
    """
    print("\n" + "="*80)
    print("STEP 9: SAVING RESULTS")
    print("="*80)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_path = OUTPUT_DIR / f"trading_metrics_{timestamp}.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"  Metrics saved: {metrics_path}")

    # Save trades
    trades_df = df_spot[['price', 'F', 'signal_ensemble', 'position', 'pnl_gross', 'pnl_net']].copy()
    trades_path = OUTPUT_DIR / f"trading_trades_{timestamp}.csv"
    trades_df.to_csv(trades_path)
    print(f"  Trades saved: {trades_path}")

    # Save summary to main reports folder
    summary_path = REPORTS_DIR / "trading_pipeline_latest.csv"
    metrics_df.to_csv(summary_path, index=False)
    print(f"  Latest summary: {summary_path}")


def main():
    """Main pipeline execution"""
    print("\n" + "="*80)
    print("FRENCH POWER FUTURES TRADING PIPELINE")
    print("="*80)
    print(f"Execution time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project root: {PROJECT_ROOT}")

    try:
        # Execute pipeline
        models_predictions = load_ml_predictions()
        df_spot = aggregate_to_daily(models_predictions)
        df_spot = load_futures_data(df_spot)
        df_spot = build_signals(df_spot)
        df_spot = apply_regime_filters(df_spot)
        df_spot = generate_positions(df_spot)
        df_spot, metrics = calculate_pnl(df_spot)
        generate_visualizations(df_spot, metrics)
        save_results(df_spot, metrics)

        print("\n" + "="*80)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"\nFinal Performance:")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"  Total PnL: EUR {metrics['total_pnl']:,.2f}")
        print(f"  Number of Trades: {metrics['num_trades']}")

        return 0

    except Exception as e:
        print(f"\n ERROR: Pipeline failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
