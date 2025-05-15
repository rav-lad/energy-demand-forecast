import pandas as pd
import numpy as np
import joblib, json
from pathlib import Path
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

BASE_DIR   = Path(__file__).resolve().parents[2]
RAW_DIR    = BASE_DIR / "data" / "modified_data" 
MODELS_DIR = BASE_DIR / "models" / "Quantile" / "lightgbm_quantile"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

import sys
sys.path.append(str(BASE_DIR))
from data_processing.transformation import transform_lightgbm_quantile

QUANTILES = [0.05, 0.5, 0.95]

def train_lightgbm_quantile(frequency: str, lags: str):
    lags_bool = lags == "with"
    suffix = "withlags" if lags_bool else "withoutlags"

    # 1) Load final training data
    df_raw = pd.read_csv(RAW_DIR / f"train_{frequency}.csv")

    # 2) Transform data
    df = transform_lightgbm_quantile(df_raw, frequency=frequency, save=True, lags=lags_bool)

    # 3) Split X / y
    y = df[["conso_elec_mw", "conso_gaz_mw"]]
    X = df.drop(columns=["conso_elec_mw", "conso_gaz_mw", "date"], errors="ignore")

    # 4) Save feature names
    feature_names = X.columns.tolist()
    with open(MODELS_DIR / f"features_{frequency}_{suffix}.json", "w") as f:
        json.dump(feature_names, f)

    # 5) Load best hyperparameters
    with open(MODELS_DIR / f"best_params_{frequency}_{suffix}.json") as f:
        all_best_params = json.load(f)

    tscv = TimeSeriesSplit(n_splits=5)
    results = {}

    for target in y.columns:
        results[target] = {}
        y_target = y[target].values

        for q in QUANTILES:
            print(f"\nTraining LightGBM for {target} - quantile {q} ({int(q*100)}%)")

            best_model = None
            best_rmse = float("inf")

            # Optimal hyperparameters
            params = all_best_params[target][f"q{int(q*100)}"]

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y_target[train_idx], y_target[val_idx]

                model = LGBMRegressor(
                    objective="quantile",
                    alpha=q,
                    random_state=42,
                    **params
                )

                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    callbacks=[early_stopping(stopping_rounds=50), log_evaluation(0)]
                )

                pred_val = model.predict(X_val)
                rmse_val = np.sqrt(mean_squared_error(y_val, pred_val))
                print(f"  ➤ Fold {fold+1} : RMSE = {rmse_val:.4f}")

                if rmse_val < best_rmse:
                    best_rmse = rmse_val
                    best_model = model

            # Evaluation on full training set
            pred_train = best_model.predict(X)
            mse_train = mean_squared_error(y_target, pred_train)

            results[target][f"q{int(q*100)}"] = {
                "mse": round(mse_train, 6),
                "rmse": round(np.sqrt(mse_train), 6),
                "val_rmse": round(best_rmse, 6)
            }

            joblib.dump(best_model, MODELS_DIR / f"{target}_q{int(q*100)}_{frequency}_{suffix}.pkl")

    # 6) Save metrics
    with open(MODELS_DIR / f"metrics_{frequency}_{suffix}.json", "w") as f:
        json.dump(results, f, indent=4)

    # 7) Console summary
    print(f"\nLightGBM quantile training completed for {frequency} ({suffix})")
    for tgt, scores in results.items():
        print(f"{tgt}")
        for q, met in scores.items():
            print(f"   {q}: MSE={met['mse']:.4f}, RMSE={met['rmse']:.4f}, val_RMSE={met['val_rmse']:.4f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--frequency", choices=["daily", "hourly"], required=True)
    p.add_argument("--lags", choices=["with", "without"], default="with")
    args = p.parse_args()
    train_lightgbm_quantile(args.frequency, args.lags)
