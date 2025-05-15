#!/usr/bin/env python
# test_tft.py

import argparse, sys
from pathlib import Path

import torch
import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

# ── chemins ───────────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).resolve().parents[2]
DATA_DIR         = BASE_DIR / "data" / "modified_data"
MODEL_DIR        = BASE_DIR / "models" / "tft"
CKPT_PATH        = MODEL_DIR / "checkpoints" / "best_tft.ckpt"
TRAINING_DS_PATH = MODEL_DIR / "tft_training_dataset.pt"

# pour transform_dl()
sys.path.append(str(BASE_DIR))
from data_processing.transformation import transform_dl

def get_test_dataset(freq: str):
    # 1) charger & transformer tout le test set
    df = pd.read_csv(DATA_DIR / f"test_{freq}.csv", parse_dates=["date"])
    df_trans = transform_dl(df, filter_too_short=True)

    # 2) recharger le schéma d'entraînement
    training_ds = TimeSeriesDataSet.load(TRAINING_DS_PATH)

    # 3) on crée un dataset EN SLIDING WINDOW (une fenêtre par date)
    test_ds = TimeSeriesDataSet.from_dataset(
        training_ds,
        df_trans,
        stop_randomization=True,  # on veut REPRODUIRE LES MÊMES longueurs qu'à l'entraînement
        predict=False            # sliding, pas un seul point par série
    )
    return test_ds, df_trans

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frequency",  choices=["daily","hourly"], required=True)
    p.add_argument("--batch_size", type=int, default=64)
    args = p.parse_args()

    print(f"🔮 Predicting TFT | freq={args.frequency} | batch={args.batch_size}")
    test_ds, df_trans = get_test_dataset(args.frequency)
    loader = test_ds.to_dataloader(train=False, batch_size=args.batch_size, num_workers=0)

    # 4) charger le modèle
    model = TemporalFusionTransformer.load_from_checkpoint(CKPT_PATH)
    model._is_model_with_custom_step = True

    # 5) prédire les quantiles + récupérer l'index
    #    retourne (predictions, index_df)
    preds, index_df = model.predict(
        loader,
        mode="quantiles",
        return_index=True,
        trainer_kwargs=dict(accelerator="cpu"),
    )
    # preds.shape == (N_windows, n_quantiles, 1, n_targets)
    # index_df contient notamment "insee_region" et "time_idx_first_prediction"

    # 6) extraire la première échéance (horizon=1) pour chaque quantile & target
    elec = preds[:, :, 0, 0].cpu().numpy()  # (N, Q)
    gaz  = preds[:, :, 0, 1].cpu().numpy()  # (N, Q)

    # 7) reformatter l'index pour merger
    df_idx = index_df.rename(columns={"time_idx_first_prediction": "time_idx"})
    df_idx = df_idx[["insee_region", "time_idx"]].reset_index(drop=True)

    # 8) récupérer la date correspondante depuis df_trans
    df_dates = df_trans[["insee_region", "time_idx", "date"]].drop_duplicates()
    df_out   = df_idx.merge(df_dates, on=["insee_region", "time_idx"], how="left")

    # 9) ajouter les colonnes de quantiles
    df_out["conso_elec_mw_q5"]  = elec[:, 0]
    df_out["conso_elec_mw_q50"] = elec[:, 1]
    df_out["conso_elec_mw_q95"] = elec[:, 2]
    df_out["conso_gaz_mw_q5"]   = gaz[:, 0]
    df_out["conso_gaz_mw_q50"]  = gaz[:, 1]
    df_out["conso_gaz_mw_q95"]  = gaz[:, 2]

    # 10) tri & sauvegarde
    df_out = df_out.sort_values(["date", "insee_region"])
    out_path = MODEL_DIR / f"tft_test_predictions_{args.frequency}.csv"
    df_out.to_csv(out_path, index=False)
    print("✅ Predictions saved to", out_path)

if __name__ == "__main__":
    main()
