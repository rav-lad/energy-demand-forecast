"""
Evaluate TFT model and generate predictions for trading backtest.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "modified_data"
MODELS_DIR = BASE_DIR / "models"
CHECKPOINT_PATH = MODELS_DIR / "tft" / "checkpoints" / "best_tft.ckpt"
DATASET_PATH = MODELS_DIR / "tft" / "tft_training_dataset.pt"

def load_data():
    """Load train and test data."""
    train_df = pd.read_csv(DATA_DIR / "train_daily.csv")
    test_df = pd.read_csv(DATA_DIR / "test_daily.csv")

    df = pd.concat([train_df, test_df], ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # Add time_idx and group
    df["time_idx"] = (df["datetime"] - df["datetime"].min()).dt.days
    df["series"] = "FR"

    return df, 560  # train_cutoff

def evaluate_tft():
    """Evaluate TFT model and save predictions."""
    print("Loading data...")
    df, train_cutoff = load_data()

    print("Loading TFT model...")
    model = TemporalFusionTransformer.load_from_checkpoint(CHECKPOINT_PATH)

    print("Recreating training dataset...")
    df_train = df[df["time_idx"] <= train_cutoff]

    time_varying_known_reals = [
        "time_idx", "month", "day", "dayofweek",
        "month_sin", "month_cos", "day_sin", "day_cos", "dayofweek_sin", "dayofweek_cos",
        "is_weekend"
    ]

    time_varying_unknown_reals = [
        "temperature_2m_max", "temperature_2m_min",
        "precipitation_sum", "wind_speed_10m_max",
        "shortwave_radiation_sum", "et0_fao_evapotranspiration"
    ]

    from pytorch_forecasting.data.encoders import TorchNormalizer

    training = TimeSeriesDataSet(
        df_train,
        time_idx="time_idx",
        target="load_mw",
        group_ids=["series"],
        static_categoricals=["series"],
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
        max_encoder_length=30,
        max_prediction_length=1,
        allow_missing_timesteps=False,
        target_normalizer=TorchNormalizer(),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    # Create test dataset
    test_data = df[df["time_idx"] > train_cutoff]
    validation = TimeSeriesDataSet.from_dataset(training, df, predict=True, stop_randomization=True)

    print("Generating predictions...")
    val_dataloader = validation.to_dataloader(train=False, batch_size=128, num_workers=0)
    predictions = model.predict(val_dataloader, return_y=True, return_x=False)

    # Extract predictions and actuals
    y_pred = predictions.output.numpy().flatten()
    y_true = predictions.y[0].numpy().flatten()

    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"\nTFT Model Evaluation:")
    print(f"  MAE:  {mae:.2f} MW")
    print(f"  RMSE: {rmse:.2f} MW")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")

    # Save metrics
    metrics = {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'mape': float(mape)
    }

    metrics_file = MODELS_DIR / "tft" / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_file}")

    # Save predictions for trading backtest
    # Match test_data length
    test_dates = test_data["datetime"].values[:len(y_pred)]
    test_actual = test_data["load_mw"].values[:len(y_pred)]

    predictions_df = pd.DataFrame({
        'datetime': test_dates,
        'actual_load': test_actual,
        'predicted_load': y_pred
    })

    predictions_file = MODELS_DIR / "tft" / "predictions.csv"
    predictions_df.to_csv(predictions_file, index=False)
    print(f"Predictions saved to: {predictions_file}")

    return metrics

if __name__ == "__main__":
    evaluate_tft()
