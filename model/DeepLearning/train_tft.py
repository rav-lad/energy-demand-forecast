#!/usr/bin/env python
# train_tft.py

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data.encoders import MultiNormalizer, TorchNormalizer
from pytorch_forecasting.metrics import QuantileLoss

# === Paths ===
BASE_DIR               = Path(__file__).resolve().parents[2]
DATA_DIR               = BASE_DIR / "data" / "modified_data"
CHECKPOINT_DIR         = BASE_DIR / "models" / "tft"
TRAINING_DATASET_PATH  = CHECKPOINT_DIR / "tft_training_dataset.pt"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# === Import transformation function ===
sys.path.append(str(BASE_DIR))
from data_processing.transformation import transform_dl

def split_train_val(df: pd.DataFrame, val_size: float = 0.2):
    df_sorted    = df.sort_values("date").reset_index(drop=True)
    unique_dates = df_sorted["date"].unique()
    n_val        = int(len(unique_dates) * val_size)
    val_start    = unique_dates[-n_val]
    return (
        df_sorted[df_sorted["date"] <  val_start],
        df_sorted[df_sorted["date"] >= val_start],
    )

def train_tft(freq: str, max_epochs: int, batch_size: int, gpus: int):
    print(f"Training TFT | freq={freq}, epochs={max_epochs}, batch={batch_size}, gpus={gpus}")

    # 1) Load and transform dataset
    df = pd.read_csv(DATA_DIR / f"train_{freq}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df_trans   = transform_dl(df, filter_too_short=True)

    # 2) Split into training and validation sets
    df_train, df_val = split_train_val(df_trans)

    # 3) Create TimeSeriesDataSet (forecast horizon = 1)
    target_cols = ["conso_elec_mw", "conso_gaz_mw"]
    training = TimeSeriesDataSet(
        df_train,
        time_idx                = "time_idx",
        target                  = target_cols,
        group_ids               = ["insee_region"],
        static_categoricals     = ["insee_region"],
        time_varying_known_reals   = [
            "temperature_2m_max","temperature_2m_min","precipitation_sum","weather_code",
            "apparent_temperature_max","apparent_temperature_min","rain_sum","snowfall_sum",
            "precipitation_hours","sunrise","sunset","sunshine_duration","daylight_duration",
            "wind_speed_10m_max","wind_gusts_10m_max","wind_direction_10m_dominant",
            "shortwave_radiation_sum","et0_fao_evapotranspiration",
        ],
        time_varying_unknown_reals = target_cols,
        max_encoder_length      = 24,
        min_encoder_length      = 12,
        max_prediction_length   = 1,   # forecast horizon = 1
        min_prediction_length   = 1,   # forecast horizon = 1
        allow_missing_timesteps = True,
        target_normalizer       = MultiNormalizer([TorchNormalizer()] * len(target_cols)),
    )

    # Save training dataset
    training.save(TRAINING_DATASET_PATH)  # type: ignore

    # 4) DataLoaders
    val_dataset  = TimeSeriesDataSet.from_dataset(training, df_val)
    train_loader = training.to_dataloader(train=True,  batch_size=batch_size, num_workers=4)
    val_loader   = val_dataset.to_dataloader(train=False, batch_size=batch_size, num_workers=4)

    # 5) Model
    model = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate               = 1e-3,
        hidden_size                 = 32,
        attention_head_size         = 4,
        dropout                     = 0.1,
        hidden_continuous_size      = 16,
        loss                        = QuantileLoss(quantiles=[0.1, 0.5, 0.9]),
        reduce_on_plateau_patience = 4,
    )
    model._is_model_with_custom_step = True  # type: ignore[attr-defined]

    # 6) Training
    trainer = Trainer(
        max_epochs        = max_epochs,
        accelerator       = "gpu" if torch.cuda.is_available() and gpus > 0 else "cpu",
        devices           = gpus if gpus > 0 else 1,
        gradient_clip_val = 0.1,
        deterministic     = True,
        callbacks         = [
            EarlyStopping(monitor="val_loss", patience=7),
            ModelCheckpoint(
                dirpath    = CHECKPOINT_DIR / "checkpoints",
                filename   = "best_tft",
                monitor    = "val_loss",
                save_top_k = 1,
            ),
        ],
    )
    trainer.fit(model, train_loader, val_loader)
    print("Best model saved to:", (CHECKPOINT_DIR / "checkpoints/best_tft.ckpt").resolve())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frequency",  choices=["daily","hourly"], required=True)
    parser.add_argument("--max_epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gpus",       type=int, default=0)
    args = parser.parse_args()

    train_tft(
        freq       = args.frequency,
        max_epochs = args.max_epochs,
        batch_size = args.batch_size,
        gpus       = args.gpus,
    )
