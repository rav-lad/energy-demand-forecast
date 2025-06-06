#!/usr/bin/env python

import argparse
import pandas as pd
import torch
from pathlib import Path

from lightning.pytorch import Trainer
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

# === PATHS ===
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "modified_data"
CHECKPOINT_PATH = BASE_DIR / "models" / "tft" / "checkpoints" / "best_tft.ckpt"
TRAINING_DATASET_PATH = BASE_DIR / "models" / "tft" / "tft_training_dataset.pt"
OUT_PATH = BASE_DIR / "models" / "tft" / "preds_forecast_daily.csv"

# === IMPORT TRANSFORMATION FUNCTION ===
import sys
sys.path.append(str(BASE_DIR))
from data_processing.transformation import transform_dl

def predict_future_tft(freq: str):
    print("TFT | Forecasting from J+1 to J+N")

    # 1. Load weather forecast data
    df = pd.read_csv(DATA_DIR / f"meteo_forecast_{freq}.csv")
    df["date"] = pd.to_datetime(df["date"])

    # 2. Transform data (encoder and features)
    df_trans = transform_dl(df, filter_too_short=False)

    # 3. Reload the training TimeSeriesDataSet (structure only)
    training_dataset = TimeSeriesDataSet.load(TRAINING_DATASET_PATH)  # type: ignore

    # 4. Create prediction dataset
    predict_dataset = TimeSeriesDataSet.from_dataset(training_dataset, df_trans, stop_randomization=True)
    predict_loader = predict_dataset.to_dataloader(train=False, batch_size=128, num_workers=0)

    # 5. Load trained model
    model = TemporalFusionTransformer.load_from_checkpoint(CHECKPOINT_PATH)

    # 6. Predict
    trainer = Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1)
    predictions, index = model.predict(predict_loader, return_index=True)  # type: ignore

    # 7. Format predictions into a DataFrame
    df_preds = index[["date", "insee_region"]].copy()
    df_preds["conso_elec_mw"] = predictions[:, :, 0].squeeze().numpy()
    df_preds["conso_gaz_mw"] = predictions[:, :, 1].squeeze().numpy()

    # 8. Save predictions
    df_preds.to_csv(OUT_PATH, index=False)
    print(f"TFT predictions saved to {OUT_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frequency", choices=["daily", "hourly"], required=True)
    args = parser.parse_args()
    predict_future_tft(args.frequency)
