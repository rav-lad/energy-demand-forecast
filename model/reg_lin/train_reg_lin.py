import argparse
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor

# ==== PATHS ====
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "modified_data"
MODELS_DIR = BASE_DIR / "models" / "reg_lin"
SCALER_DIR = BASE_DIR / "models" / "scalers"
RAW_DIR = DATA_DIR

SCALER_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# === IMPORT TRANSFORMATION ===
import sys
sys.path.append(str(BASE_DIR))
from data_processing.transformation import transform_regression_and_xgb

def train_reg_lin(model_type: str, frequency: str):
    # === 1. Load data ===
    df_raw = pd.read_csv(RAW_DIR / f"train_{frequency}.csv")
    df = transform_regression_and_xgb(df_raw, frequency=frequency, fit_scaler=False, save=False)
    y = df[["conso_elec_mw", "conso_gaz_mw"]]
    X = df.drop(columns=["conso_elec_mw", "conso_gaz_mw"])

    # === 2. Load best hyperparameters ===
    with open(MODELS_DIR / f"best_params_{frequency}.json", "r") as f:
        best_params_all = json.load(f)
    best_params = best_params_all[model_type]

    base = Lasso(**best_params, max_iter=10000) if model_type == "lasso" else Ridge(**best_params)
    model = MultiOutputRegressor(base)

    # === 3. Train model ===
    model.fit(X, y)
    preds = model.predict(X)
    mse_e = mean_squared_error(y.iloc[:, 0], preds[:, 0]) # type: ignore
    mse_g = mean_squared_error(y.iloc[:, 1], preds[:, 1]) # type: ignore
    print(f" Electricity RMSE: {np.sqrt(mse_e):.4f}")
    print(f" Gas RMSE:        {np.sqrt(mse_g):.4f}")

    # === 4. Save model ===
    model_path = MODELS_DIR / f"{model_type}_classic_{frequency}.pkl"
    joblib.dump(model, model_path)
    print(f" Model saved: {model_path}")

    # === 5. Save features used at training ===
    features = list(X.columns)
    features_path = MODELS_DIR / f"features_{frequency}.json"
    with open(features_path, "w") as f:
        json.dump(features, f, indent=4)
    print(f" Features saved: {features_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["lasso", "ridge"], required=True)
    p.add_argument("--frequency", choices=["daily", "hourly"], required=True)
    args = p.parse_args()

    train_reg_lin(args.model, args.frequency)
