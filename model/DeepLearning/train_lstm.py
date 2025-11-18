#!/usr/bin/env python
"""LSTM model training for energy price forecasting.

Simple LSTM architecture for time series forecasting with lookback window.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import json

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "modified_data"
MODELS_DIR = BASE_DIR / "models" / "lstm"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class TimeSeriesDataset(Dataset):
    """Dataset for LSTM time series forecasting."""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    """Simple LSTM/GRU model for time series forecasting."""

    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, use_gru=True):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_gru = use_gru

        if use_gru:
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )
        else:
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True
            )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        rnn_out, _ = self.rnn(x)
        # Take last timestep output
        out = self.fc(rnn_out[:, -1, :])
        return out.squeeze()


def create_sequences(data, features, target, lookback=30):
    """Create sequences for LSTM training.

    Args:
        data: DataFrame with all data
        features: List of feature column names
        target: Target column name
        lookback: Number of timesteps to look back

    Returns:
        X, y numpy arrays
    """
    X, y = [], []

    data_values = data[features].values
    target_values = data[target].values

    for i in range(lookback, len(data)):
        X.append(data_values[i-lookback:i])
        y.append(target_values[i])

    return np.array(X), np.array(y)


def train_lstm(
    freq: str = "daily",
    target: str = "price",
    lookback: int = 30,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    batch_size: int = 32,
    max_epochs: int = 50,
    lr: float = 0.001,
    device: str = "cpu"
):
    """Train LSTM model for price/load forecasting.

    Args:
        freq: Data frequency ("daily" or "hourly")
        target: Target to predict ("price" or "load")
        lookback: Number of timesteps to look back
        hidden_size: LSTM hidden size
        num_layers: Number of LSTM layers
        dropout: Dropout rate
        batch_size: Batch size for training
        max_epochs: Maximum number of epochs
        lr: Learning rate
        device: Device to train on ("cpu" or "cuda")
    """
    print(f"Training LSTM | target={target}, lookback={lookback}, hidden={hidden_size}, layers={num_layers}")

    # Load data
    train_df = pd.read_csv(DATA_DIR / f"train_{freq}.csv")
    test_df = pd.read_csv(DATA_DIR / f"test_{freq}.csv")

    df = pd.concat([train_df, test_df], ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # Define features
    if target == "price":
        # Predict price, use load as feature
        feature_cols = [
            "load_mw",
            "temperature_2m_max", "temperature_2m_min",
            "precipitation_sum", "wind_speed_10m_max",
            "shortwave_radiation_sum", "et0_fao_evapotranspiration",
            "month_sin", "month_cos", "day_sin", "day_cos",
            "dayofweek_sin", "dayofweek_cos", "is_weekend"
        ]
        target_col = "price_eur_mwh"
    else:
        # Predict load, use price as feature
        feature_cols = [
            "price_eur_mwh",
            "temperature_2m_max", "temperature_2m_min",
            "precipitation_sum", "wind_speed_10m_max",
            "shortwave_radiation_sum", "et0_fao_evapotranspiration",
            "month_sin", "month_cos", "day_sin", "day_cos",
            "dayofweek_sin", "dayofweek_cos", "is_weekend"
        ]
        target_col = "load_mw"

    # Split train/val (560/140)
    train_cutoff = 560
    train_data = df[:train_cutoff].copy()
    val_data = df[train_cutoff:].copy()

    # Normalize features
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    train_data[feature_cols] = scaler_X.fit_transform(train_data[feature_cols])
    train_data[[target_col]] = scaler_y.fit_transform(train_data[[target_col]])

    val_data[feature_cols] = scaler_X.transform(val_data[feature_cols])
    val_data[[target_col]] = scaler_y.transform(val_data[[target_col]])

    # Create sequences
    X_train, y_train = create_sequences(train_data, feature_cols, target_col, lookback)
    X_val, y_val = create_sequences(val_data, feature_cols, target_col, lookback)

    print(f"Train sequences: {X_train.shape}, Val sequences: {X_val.shape}")

    # Create datasets and loaders
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model (use GRU by default, faster and often better)
    input_size = len(feature_cols)
    model = LSTMModel(input_size, hidden_size, num_layers, dropout, use_gru=True)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10

    for epoch in range(max_epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}/{max_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler_X': scaler_X,
                'scaler_y': scaler_y,
                'feature_cols': feature_cols,
                'target_col': target_col,
                'lookback': lookback,
                'input_size': input_size,
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'dropout': dropout,
                'use_gru': True,
            }, MODELS_DIR / "best_lstm.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    print(f"\nBest model saved to: {MODELS_DIR / 'best_lstm.pt'}")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Save metadata
    metadata = {
        'target': target,
        'lookback': lookback,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'best_val_loss': float(best_val_loss),
        'feature_cols': feature_cols,
        'target_col': target_col
    }

    with open(MODELS_DIR / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frequency", choices=["daily", "hourly"], default="daily")
    parser.add_argument("--target", choices=["price", "load"], default="price")
    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")

    args = parser.parse_args()

    train_lstm(
        freq=args.frequency,
        target=args.target,
        lookback=args.lookback,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        lr=args.lr,
        device=args.device
    )
