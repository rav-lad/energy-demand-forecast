import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from xgboost import XGBRegressor

import sys
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))
from data_processing.transformation import transform_regression_and_xgb

# === Paths ===
RAW_DIR     = BASE_DIR / "data" / "modified_data"
MODELS_DIR  = BASE_DIR / "models" / "xgboost"
SCALER_DIR  = BASE_DIR / "models" / "scalers"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def train_xgboost(frequency: str):
    # 1) Load full training data
    df_raw = pd.read_csv(RAW_DIR / f"train_{frequency}.csv")

    # 2) Transform using pre-fitted scaler
    df = transform_regression_and_xgb(df_raw, frequency=frequency, fit_scaler=False, save=False)
    y = df[["conso_elec_mw", "conso_gaz_mw"]]
    X = df.drop(columns=["conso_elec_mw", "conso_gaz_mw"])

    # 3) Load best hyperparameters
    with open(MODELS_DIR / f"best_params_{frequency}.json", "r") as f:
        best_params = json.load(f)

    base_model = XGBRegressor(**best_params, n_jobs=-1, random_state=42)
    model = MultiOutputRegressor(base_model)

    # 4) Train and evaluate
    model.fit(X, y)
    y_pred = model.predict(X)

    mse_elec = mean_squared_error(y.iloc[:, 0], y_pred[:, 0]) # type: ignore
    mse_gaz = mean_squared_error(y.iloc[:, 1], y_pred[:, 1]) # type: ignore
    print(f" Electricity RMSE: {np.sqrt(mse_elec):.4f}")
    print(f" Gas RMSE:        {np.sqrt(mse_gaz):.4f}")

    # 5) Save model
    joblib.dump(model, MODELS_DIR / f"xgb_{frequency}.pkl")
    print(f" XGBoost model saved: {MODELS_DIR / f'xgb_{frequency}.pkl'}")

    # 6) Save training features
    features = list(X.columns)
    with open(MODELS_DIR / f"features_{frequency}.json", "w") as f:
        json.dump(features, f, indent=4)
    print(f" Training features saved: {MODELS_DIR / f'features_{frequency}.json'}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--frequency", choices=["daily", "hourly"], required=True)
    args = parser.parse_args()

    train_xgboost(frequency=args.frequency)
