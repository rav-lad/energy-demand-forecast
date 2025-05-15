import argparse
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
freq = "daily"

# ==== PATHS ====
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "modified_data"
MODELS_DIR = BASE_DIR / "models"
SCALER_DIR = BASE_DIR / "models" / "scalers"
scaler_path = SCALER_DIR / f"scaler_{freq}_reglin_xgboost.pkl"

# ==== MODELS ====
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge, Lasso
from xgboost import XGBRegressor

# ==== TRANSFORMATIONS ====
import sys
sys.path.append(str(BASE_DIR))
from data_processing.transformation import (
    transform_lightgbm_quantile,
    transform_regression_and_xgb,
)
from data_processing.rolling_forecast import rolling_forecast_lightgbm


def predict_lightgbm_quantile(freq: str, lags: str):
    suffix = "withlags" if lags == "with" else "withoutlags"

    if lags == "with":
        print("Rolling forecast mode activated (with predictive lags)")
        rolling_forecast_lightgbm(freq=freq, lags=lags)
        return

    print("Direct prediction mode (no rolling, no lags)")

    # 1) Load test set
    df_test = pd.read_csv(RAW_DIR / f"test_{freq}.csv")
    df_test["date"] = pd.to_datetime(df_test["date"])

    # 2) Transform without lags
    df_trans = transform_lightgbm_quantile(df_test, frequency=freq, save=False, lags=False)

    # 3) Load features
    with open(MODELS_DIR / "Quantile" / "lightgbm_quantile" / f"features_{freq}_withoutlags.json") as f:
        expected_features = json.load(f)
    X = df_trans.reindex(columns=expected_features, fill_value=0)

    # 4) Load models ONCE
    models = {}
    for target in ["conso_elec_mw", "conso_gaz_mw"]:
        for q in [5, 50, 95]:
            model_path = MODELS_DIR / "Quantile" / "lightgbm_quantile" / f"{target}_q{q}_{freq}_withoutlags.pkl"
            models[(target, q)] = joblib.load(model_path)

    # 5) Vectorized prediction
    preds = []
    for i, row in df_test.iterrows():
        input_row = X.iloc[[i]]
        out = {
            "date": row["date"],
            "insee_region": row["insee_region"]
        }
        for target in ["conso_elec_mw", "conso_gaz_mw"]:
            for q in [5, 50, 95]:
                out[f"{target}_q{q}"] = float(models[(target, q)].predict(input_row)[0])
        preds.append(out)

    # 6) Save predictions
    df_preds = pd.DataFrame(preds)
    out_path = MODELS_DIR / "Quantile" / "lightgbm_quantile" / f"preds_{freq}_direct_withoutlags.csv"
    df_preds.to_csv(out_path, index=False)
    print(f"LightGBM predictions (without lags) saved to {out_path}")


def predict_xgboost(freq, mode):
    df_raw = pd.read_csv(RAW_DIR / f"test_{freq}.csv")
    df = transform_regression_and_xgb(df_raw, frequency=freq, save=False, fit_scaler=False, scaler_path=scaler_path)

    meta = df_raw[["date", "insee_region"]].copy()
    y_true = df[["conso_elec_mw", "conso_gaz_mw"]]
    X = df.drop(columns=["conso_elec_mw", "conso_gaz_mw"], errors="ignore")

    model_path = MODELS_DIR / "xgboost" / f"xgb_{freq}.pkl"
    if not model_path.exists():
        print(f"XGBoost model not found: {model_path}")
        return

    model = joblib.load(model_path)

    if mode == "classic":
        y_pred = model.predict(X)
    else:
        rng = np.random.default_rng(42)
        n_samples = 30
        preds = [model.predict(X + rng.normal(0, 0.1, X.shape)) for _ in range(n_samples)]
        preds = np.array(preds)
        y_pred = preds.mean(axis=0)

        info = {
            "n_samples": n_samples,
            "seed": 42,
            "rmse_mean_elec": float(np.mean([np.sqrt(np.mean((p[:, 0] - y_true.iloc[:, 0])**2)) for p in preds])),
            "rmse_mean_gaz": float(np.mean([np.sqrt(np.mean((p[:, 1] - y_true.iloc[:, 1])**2)) for p in preds]))
        }
        info_path = MODELS_DIR / "xgboost" / f"info_scenario_{freq}.json"
        with open(info_path, "w") as f:
            json.dump(info, f, indent=4)

    df_preds = meta.copy()
    df_preds["conso_elec_mw"] = y_pred[:, 0]
    df_preds["conso_gaz_mw"] = y_pred[:, 1]

    out_path = MODELS_DIR / "xgboost" / f"preds_{freq}_{mode}.csv"
    df_preds.to_csv(out_path, index=False)
    print(f"XGBoost predictions ({mode}) saved to {out_path}")


def predict_reg_lin(freq, method, mode):
    df_raw = pd.read_csv(RAW_DIR / f"test_{freq}.csv")
    df = transform_regression_and_xgb(df_raw, frequency=freq, save=False, fit_scaler=False, scaler_path=scaler_path)

    meta = df_raw[["date", "insee_region"]].copy()
    X = df.drop(columns=["conso_elec_mw", "conso_gaz_mw"], errors="ignore")

    model_file = MODELS_DIR / "reg_lin" / f"{method}_classic_{freq}.pkl"
    if not model_file.exists():
        print(f"Model {method} not found: {model_file}")
        return

    model = joblib.load(model_file)

    if mode == "classic":
        y_pred = model.predict(X)
    else:
        rng = np.random.default_rng(42)
        n_samples = 30
        preds = [model.predict(X + rng.normal(0, 0.1, X.shape)) for _ in range(n_samples)]
        preds = np.array(preds)
        y_pred = preds.mean(axis=0)

        info = {
            "n_samples": n_samples,
            "seed": 42,
            "rmse_mean_elec": float(np.mean([np.sqrt(np.mean((p[:, 0] - model.predict(X)[:, 0])**2)) for p in preds])),
            "rmse_mean_gaz": float(np.mean([np.sqrt(np.mean((p[:, 1] - model.predict(X)[:, 1])**2)) for p in preds]))
        }
        info_path = MODELS_DIR / "reg_lin" / f"info_scenario_{method}_{freq}.json"
        with open(info_path, "w") as f:
            json.dump(info, f, indent=4)

    df_preds = meta.copy()
    df_preds["conso_elec_mw"] = y_pred[:, 0]
    df_preds["conso_gaz_mw"] = y_pred[:, 1]

    out_path = MODELS_DIR / "reg_lin" / f"preds_{method}_{freq}_{mode}.csv"
    df_preds.to_csv(out_path, index=False)
    print(f"Linear regression predictions ({method}, {mode}) saved to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lightgbm", "reg_lin", "xgboost"], required=True)
    parser.add_argument("--frequency", choices=["daily", "hourly"], required=True)
    parser.add_argument("--mode", choices=["classic", "scenario"], default="classic",
                    help="Prediction mode for XGBoost or reg_lin")
    parser.add_argument("--method", choices=["ridge", "lasso"], help="Model choice for reg_lin")
    parser.add_argument("--lags", choices=["with", "without"], default="with", help="Enable or disable lags for LightGBM")

    args = parser.parse_args()

    if args.model == "lightgbm":
        predict_lightgbm_quantile(args.frequency, args.lags)
    if args.model == "xgboost":
        predict_xgboost(args.frequency, args.mode)
    elif args.model == "reg_lin":
        if not args.method:
            parser.error('--method is required for reg_lin')
        predict_reg_lin(args.frequency, args.method, args.mode)


if __name__ == "__main__":
    main()
