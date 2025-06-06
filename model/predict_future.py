#!/usr/bin/env python

import argparse
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# ==== PATHS ====
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "modified_data"
MODELS_DIR = BASE_DIR / "models"
SCALER_DIR = MODELS_DIR / "scalers"

# ==== TRANSFORMATIONS ====
sys.path.append(str(BASE_DIR))
from data_processing.transformation import (
    transform_lightgbm_quantile,
    transform_regression_and_xgb,
)

# ==== MODELS ====
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import Ridge, Lasso


def predict_future_lightgbm(freq: str):
    print(" LightGBM (quantile) | mode: no lags")

    df_forecast = pd.read_csv(RAW_DIR / f"meteo_forecast_{freq}.csv")
    df_forecast["date"] = pd.to_datetime(df_forecast["date"])

    df_trans = transform_lightgbm_quantile(df_forecast, frequency=freq, save=False, lags=False)

    with open(MODELS_DIR / "Quantile" / "lightgbm_quantile" / f"features_{freq}_withoutlags.json") as f:
        expected_features = json.load(f)
    X = df_trans.reindex(columns=expected_features, fill_value=0).reset_index(drop=True)
    df_forecast = df_forecast.reset_index(drop=True)

    models = {}
    for target in ["conso_elec_mw", "conso_gaz_mw"]:
        for q in [5, 50, 95]:
            model_path = MODELS_DIR / "Quantile" / "lightgbm_quantile" / f"{target}_q{q}_{freq}_withoutlags.pkl"
            models[(target, q)] = joblib.load(model_path)

    df_preds = df_forecast[["date", "insee_region"]].copy()
    for target in ["conso_elec_mw", "conso_gaz_mw"]:
        for q in [5, 50, 95]:
            df_preds[f"{target}_q{q}"] = models[(target, q)].predict(X)

    out_path = MODELS_DIR / "Quantile" / "lightgbm_quantile" / f"preds_forecast_{freq}_withoutlags.csv"
    df_preds.to_csv(out_path, index=False)
    print(f" Prédictions LightGBM sauvegardées dans {out_path}")


def predict_future_xgboost(freq, mode):
    print(f" XGBoost | mode: {mode}")

    scaler_path = SCALER_DIR / f"scaler_{freq}_reglin_xgboost.pkl"
    df_raw = pd.read_csv(RAW_DIR / f"meteo_forecast_{freq}.csv")
    df = transform_regression_and_xgb(df_raw, frequency=freq, save=False, fit_scaler=False, scaler_path=scaler_path)

    meta = df_raw[["date", "insee_region"]].copy()
    X = df.drop(columns=["conso_elec_mw", "conso_gaz_mw"], errors="ignore")

    # Reindex to match training features
    features_path = MODELS_DIR / "xgboost" / f"features_{freq}.json"
    with open(features_path) as f:
        expected_features = json.load(f)
    X = X.reindex(columns=expected_features, fill_value=0)

    model_path = MODELS_DIR / "xgboost" / f"xgb_{freq}.pkl"
    if not model_path.exists():
        print(f" XGBoost model not found: {model_path}")
        return
    model = joblib.load(model_path)

    if mode == "classic":
        y_pred = model.predict(X)
    else:
        rng = np.random.default_rng(42)
        preds = [model.predict(X + rng.normal(0, 0.1, X.shape)) for _ in range(30)]
        y_pred = np.mean(preds, axis=0)

    df_preds = meta.copy()
    df_preds["conso_elec_mw"] = y_pred[:, 0]
    df_preds["conso_gaz_mw"] = y_pred[:, 1]

    out_path = MODELS_DIR / "xgboost" / f"preds_forecast_{freq}_{mode}.csv"
    df_preds.to_csv(out_path, index=False)
    print(f" Prédictions XGBoost sauvegardées dans {out_path}")


def predict_future_reglin(freq, method, mode):
    print(f" Régression Linéaire ({method}) | mode: {mode}")

    scaler_path = SCALER_DIR / f"scaler_{freq}_reglin_xgboost.pkl"
    df_raw = pd.read_csv(RAW_DIR / f"meteo_forecast_{freq}.csv")
    df = transform_regression_and_xgb(df_raw, frequency=freq, save=False, fit_scaler=False, scaler_path=scaler_path)

    meta = df_raw[["date", "insee_region"]].copy()
    X = df.drop(columns=["conso_elec_mw", "conso_gaz_mw"], errors="ignore")

    # Reindex to match training features
    features_path = MODELS_DIR / "reg_lin" / f"features_{freq}.json"
    with open(features_path) as f:
        expected_features = json.load(f)
    X = X.reindex(columns=expected_features, fill_value=0)

    model_path = MODELS_DIR / "reg_lin" / f"{method}_classic_{freq}.pkl"
    if not model_path.exists():
        print(f" Model {method} not found: {model_path}")
        return
    model = joblib.load(model_path)

    if mode == "classic":
        y_pred = model.predict(X)
    else:
        rng = np.random.default_rng(42)
        preds = [model.predict(X + rng.normal(0, 0.1, X.shape)) for _ in range(30)]
        y_pred = np.mean(preds, axis=0)

    df_preds = meta.copy()
    df_preds["conso_elec_mw"] = y_pred[:, 0]
    df_preds["conso_gaz_mw"] = y_pred[:, 1]

    out_path = MODELS_DIR / "reg_lin" / f"preds_forecast_{method}_{freq}_{mode}.csv"
    df_preds.to_csv(out_path, index=False)
    print(f" Prédictions {method} sauvegardées dans {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lightgbm", "xgboost", "reg_lin"], required=True)
    parser.add_argument("--frequency", choices=["daily", "hourly"], required=True)
    parser.add_argument("--mode", choices=["classic", "scenario"], default="classic")
    parser.add_argument("--method", choices=["ridge", "lasso"], help="Required if --model reg_lin")

    args = parser.parse_args()

    if args.model == "lightgbm":
        predict_future_lightgbm(args.frequency)
    elif args.model == "xgboost":
        predict_future_xgboost(args.frequency, args.mode)
    elif args.model == "reg_lin":
        if not args.method:
            parser.error("--method is required for reg_lin")
        predict_future_reglin(args.frequency, args.method, args.mode)


if __name__ == "__main__":
    main()
