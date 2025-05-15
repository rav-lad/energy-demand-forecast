import sys
from pathlib import Path
import json
import argparse

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, RandomizedSearchCV
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, make_scorer
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.stats import loguniform, randint  # for future flexible grids

# Import transformations
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(BASE_DIR))
from data_processing.transformation import (
    transform_regression_and_xgb,
    transform_lightgbm_quantile,
)

# Paths
DATA_DIR = BASE_DIR / "data" / "modified_data"
MODELS_DIR = BASE_DIR / "models"
SCALER_DIR = BASE_DIR / "models" / "scalers"

# Scoring function
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse_scorer = make_scorer(rmse, greater_is_better=False)

# Hyperparameter grids
ridge_params = {
    "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 50.0, 100.0, 200.0],
    "fit_intercept": [True, False],
    "solver": ["auto", "saga"],
}

lasso_params = {
    "alpha": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 50.0],
    "fit_intercept": [True, False],
    "selection": ["cyclic", "random"],
}

xgb_params = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.01, 0.1],
    "max_depth": [3, 5, 7],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "gamma": [0, 1, 5],
    "reg_alpha": [0, 0.1, 1],
    "reg_lambda": [1, 5, 10],
}

lightgbm_params = {
    "learning_rate": [0.01, 0.05, 0.1],
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 7],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
    "min_child_samples": [5, 10, 20],
    "reg_alpha": [0, 0.1, 1.0],
    "reg_lambda": [0, 1.0, 5.0],
}

def run_gridsearch(model_type: str, frequency: str, lags: str):
    lags_bool = lags == "with"
    suffix = "withlags" if lags_bool else "withoutlags"

    # Step 1: Load full training set
    df = pd.read_csv(DATA_DIR / f"train_{frequency}.csv")

    # Step 2: Apply transformation using existing scaler
    if model_type in ["reg_lin", "xgboost"]:
        scaler_path = SCALER_DIR / f"scaler_{frequency}_reglin_xgboost.pkl"
        df_transformed = transform_regression_and_xgb(
            df, frequency=frequency, save=False, fit_scaler=False, scaler_path=scaler_path
        )
    elif model_type == "lightgbm":
        df_transformed = transform_lightgbm_quantile(
            df, frequency=frequency, save=False, lags=lags_bool
        )
    else:
        raise ValueError("Unknown model type.")

    # Step 3: Split X / y
    y = df_transformed["conso_elec_mw"]
    X = df_transformed.drop(columns=["conso_elec_mw", "conso_gaz_mw", "date"], errors="ignore")
    tscv = TimeSeriesSplit(n_splits=5)

    # Step 4: Hyperparameter search
    if model_type == "reg_lin":
        best_models = {}

        ridge_search = RandomizedSearchCV(
            Ridge(),
            ridge_params,
            n_iter=25,
            scoring=rmse_scorer,
            cv=tscv,
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        ridge_search.fit(X, y)
        best_models["ridge"] = ridge_search.best_params_

        lasso_search = RandomizedSearchCV(
            Lasso(max_iter=10000),
            lasso_params,
            n_iter=25,
            scoring=rmse_scorer,
            cv=tscv,
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        lasso_search.fit(X, y)
        best_models["lasso"] = lasso_search.best_params_

        (MODELS_DIR / "reg_lin").mkdir(parents=True, exist_ok=True)
        with open(MODELS_DIR / "reg_lin" / f"best_params_{frequency}.json", "w") as f:
            json.dump(best_models, f, indent=4)

        print(f"Best hyperparameters for reg_lin ({frequency}) saved.")

    elif model_type == "xgboost":
        xgb_search = RandomizedSearchCV(
            XGBRegressor(n_jobs=4, random_state=42, objective="reg:squarederror"),
            xgb_params,
            n_iter=50,
            scoring=rmse_scorer,
            cv=tscv,
            n_jobs=-1,
            random_state=42,
            verbose=1
        )
        xgb_search.fit(X, y)
        best_params = xgb_search.best_params_

        (MODELS_DIR / "xgboost").mkdir(parents=True, exist_ok=True)
        with open(MODELS_DIR / "xgboost" / f"best_params_{frequency}.json", "w") as f:
            json.dump(best_params, f, indent=4)

        print(f"Best hyperparameters for XGBoost saved ({frequency}).")

    elif model_type == "lightgbm":
        best_params = {}
        target_dir = MODELS_DIR / "Quantile" / "lightgbm_quantile"
        target_dir.mkdir(parents=True, exist_ok=True)

        for target_name in ["conso_elec_mw", "conso_gaz_mw"]:
            y_target = df_transformed[target_name]
            best_params[target_name] = {}

            for alpha in [0.05, 0.5, 0.95]:
                model = LGBMRegressor(objective="quantile", alpha=alpha, random_state=42)
                search = GridSearchCV(
                    model,
                    lightgbm_params,
                    scoring=rmse_scorer,
                    cv=tscv,
                    n_jobs=-1,
                    verbose=1
                )
                search.fit(X, y_target)
                best_params[target_name][f"q{int(alpha * 100)}"] = search.best_params_

        with open(target_dir / f"best_params_{frequency}_{suffix}.json", "w") as f:
            json.dump(best_params, f, indent=4)

        print(f"Best hyperparameters for LightGBM quantile saved ({suffix}) for {frequency}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["reg_lin", "xgboost", "lightgbm"], required=True)
    parser.add_argument("--frequency", choices=["daily", "hourly"], required=True)
    parser.add_argument("--lags", choices=["with", "without"], default="with")
    args = parser.parse_args()

    run_gridsearch(args.model, args.frequency, args.lags)
