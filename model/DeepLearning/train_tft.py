#!/usr/bin/env python
"""Temporal Fusion Transformer (TFT) model training for energy demand forecasting.

This module implements training for the Temporal Fusion Transformer, a deep
learning architecture for multi-horizon time series forecasting with
interpretable attention mechanisms.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data.encoders import TorchNormalizer
from pytorch_forecasting.metrics import QuantileLoss

# === Paths ===
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "modified_data"
CHECKPOINT_DIR = BASE_DIR / "models" / "tft"
TRAINING_DATASET_PATH = CHECKPOINT_DIR / "tft_training_dataset.pt"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

def split_train_val(df: pd.DataFrame, train_cutoff_idx: int):
    """Split time series data into training and validation sets.

    Args:
        df: Input DataFrame with time series data
        train_cutoff_idx: time_idx value to split at

    Returns:
        Tuple of (train_df, val_df) DataFrames
    """
    return (
        df[df["time_idx"] <= train_cutoff_idx],
        df[df["time_idx"] > train_cutoff_idx],
    )


def train_tft(freq: str, max_epochs: int, batch_size: int, gpus: int, target: str = 'price'):
    """Train Temporal Fusion Transformer model for energy forecasting.

    Args:
        freq: Data frequency, either "daily" or "hourly"
        max_epochs: Maximum number of training epochs
        batch_size: Batch size for training
        gpus: Number of GPUs to use (0 for CPU training)
        target: Target variable - 'price' for price_eur_mwh, 'load' for load_mw
    """
    print(f"Training TFT | freq={freq}, epochs={max_epochs}, batch={batch_size}, gpus={gpus}, target={target}")

    # 1) Load dataset
    train_df = pd.read_csv(DATA_DIR / f"train_{freq}.csv")
    test_df = pd.read_csv(DATA_DIR / f"test_{freq}.csv")
    df = pd.concat([train_df, test_df], ignore_index=True)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # Add time_idx and group
    df["time_idx"] = (df["datetime"] - df["datetime"].min()).dt.days
    df["series"] = "FR"  # Single series for entire France

    # 2) Split into training and validation sets (560/140 split)
    train_cutoff = 560
    df_train, df_val = split_train_val(df, train_cutoff)

    # 3) Create TimeSeriesDataSet (forecast horizon = 1)
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

    # Add load_mw to time_varying_unknown_reals when predicting price
    # Add price_eur_mwh to time_varying_unknown_reals when predicting load
    if target == 'price':
        time_varying_unknown_reals.append("load_mw")
        target_var = "price_eur_mwh"
    else:
        time_varying_unknown_reals.append("price_eur_mwh")
        target_var = "load_mw"

    training = TimeSeriesDataSet(
        df_train,
        time_idx="time_idx",
        target=target_var,
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

    # Save training dataset
    training.save(TRAINING_DATASET_PATH)  # type: ignore

    # 4) DataLoaders
    val_dataset = TimeSeriesDataSet.from_dataset(training, df_val)
    train_loader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=4)
    val_loader = val_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=4)

    # 5) Model
    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=1e-3,
        hidden_size=32,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=16,
        loss=QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
        reduce_on_plateau_patience=4,
    )
    model._is_model_with_custom_step = True  # type: ignore[attr-defined]

    # 6) Training
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() and gpus > 0 else "cpu",
        devices=gpus if gpus > 0 else 1,
        gradient_clip_val=0.1,
        deterministic=True,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=7),
            ModelCheckpoint(
                dirpath=CHECKPOINT_DIR / "checkpoints",
                filename="best_tft",
                monitor="val_loss",
                save_top_k=1,
            ),
        ],
    )
    trainer.fit(model, train_loader, val_loader)
    print("Best model saved to:", (CHECKPOINT_DIR / "checkpoints/best_tft.ckpt").resolve())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frequency", choices=["daily", "hourly"], required=True)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--target", choices=["price", "load"], default="price")
    args = parser.parse_args()

    train_tft(
        freq=args.frequency,
        max_epochs=args.max_epochs,
        batch_size=args.batch_size,
        gpus=args.gpus,
        target=args.target,
    )
