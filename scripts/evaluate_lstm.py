"""Evaluate LSTM model and generate predictions for trading backtest."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import argparse

# Add model directory to path
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR / "model" / "DeepLearning"))

from train_lstm import LSTMModel, create_sequences

DATA_DIR = BASE_DIR / "data" / "modified_data"
MODELS_DIR = BASE_DIR / "models" / "lstm"


def evaluate_lstm(device: str = "cpu"):
    """Evaluate LSTM model and save predictions."""

    print("Loading LSTM model...")
    checkpoint = torch.load(MODELS_DIR / "best_lstm.pt", map_location=device, weights_only=False)

    # Extract metadata
    feature_cols = checkpoint['feature_cols']
    target_col = checkpoint['target_col']
    lookback = checkpoint['lookback']
    scaler_X = checkpoint['scaler_X']
    scaler_y = checkpoint['scaler_y']

    # Reconstruct model
    use_gru = checkpoint.get('use_gru', True)
    model = LSTMModel(
        input_size=checkpoint['input_size'],
        hidden_size=checkpoint['hidden_size'],
        num_layers=checkpoint['num_layers'],
        dropout=checkpoint['dropout'],
        use_gru=use_gru
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print("Loading data...")
    train_df = pd.read_csv(DATA_DIR / "train_daily.csv")
    test_df = pd.read_csv(DATA_DIR / "test_daily.csv")

    df = pd.concat([train_df, test_df], ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # Split at same point as training
    train_cutoff = 560
    test_data = df[train_cutoff:].copy()

    # Store original values for metrics
    y_test_original = test_data[target_col].values[lookback:]
    test_dates = test_data["datetime"].values[lookback:]

    # Normalize
    test_data[feature_cols] = scaler_X.transform(test_data[feature_cols])
    test_data[[target_col]] = scaler_y.transform(test_data[[target_col]])

    # Create sequences
    X_test, y_test = create_sequences(test_data, feature_cols, target_col, lookback)

    print(f"Generating predictions for {len(X_test)} samples...")

    # Predict
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_pred_normalized = model(X_test_tensor).cpu().numpy()

    # Denormalize predictions
    y_pred = scaler_y.inverse_transform(y_pred_normalized.reshape(-1, 1)).flatten()

    # Calculate metrics on original scale
    mae = mean_absolute_error(y_test_original, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_original, y_pred))
    r2 = r2_score(y_test_original, y_pred)
    mape = np.mean(np.abs((y_test_original - y_pred) / y_test_original)) * 100

    target_name = "price" if "price" in target_col else "load"
    unit = "EUR/MWh" if target_name == "price" else "MW"

    print(f"\nLSTM Model Evaluation ({target_name}):")
    print(f"  MAE:  {mae:.2f} {unit}")
    print(f"  RMSE: {rmse:.2f} {unit}")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")

    # Save metrics
    metrics = {
        'mae': float(mae),
        'rmse': float(rmse),
        'r2': float(r2),
        'mape': float(mape)
    }

    metrics_file = MODELS_DIR / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_file}")

    # Save predictions
    predictions_df = pd.DataFrame({
        'datetime': test_dates,
        f'actual_{target_name}': y_test_original,
        f'predicted_{target_name}': y_pred
    })

    predictions_file = MODELS_DIR / "predictions.csv"
    predictions_df.to_csv(predictions_file, index=False)
    print(f"Predictions saved to: {predictions_file}")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    evaluate_lstm(device=args.device)
