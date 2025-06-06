import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
from pathlib import Path
from sklearn.inspection import PartialDependenceDisplay
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.multioutput import MultiOutputRegressor

# === PATHS ===
BASE_DIR = Path(__file__).resolve().parents[0].parents[0]
RAW_DIR = BASE_DIR / "data" / "modified_data"
MODELS_DIR = BASE_DIR / "models"
TEST_DIR = RAW_DIR

# === IMPORT TRANSFORMATIONS ===
import sys
sys.path.append(str(BASE_DIR))
from data_processing.transformation import transform_lightgbm_quantile, transform_regression_and_xgb

def plot_pdp(model_type: str, freq: str, features_to_plot: list[str] = None, method: str = None, lags: str = "without"): # type: ignore
    """
    Plot Partial Dependence Plots (PDP) for XGBoost, Linear Regression, and LightGBM Quantile models.
    
    Args:
        model_type (str): "xgboost", "reg_lin", or "lightgbm"
        freq (str): "daily" or "hourly"
        features_to_plot (list, optional): list of feature names to plot. If None, automatically selected.
        method (str, optional): "lasso" or "ridge" (only for reg_lin)
        lags (str, optional): "with" or "without" (only for lightgbm)
    """

    suffix = "withlags" if lags == "with" else "withoutlags"

    # === Load model
    if model_type == "xgboost":
        model_path = MODELS_DIR / "xgboost" / f"xgb_{freq}.pkl"
        model = joblib.load(model_path)

    elif model_type == "reg_lin":
        if method not in ["lasso", "ridge"]:
            raise ValueError("For reg_lin, method must be 'lasso' or 'ridge'")
        model_path = MODELS_DIR / "reg_lin" / f"{method}_classic_{freq}.pkl"
        model = joblib.load(model_path)

    elif model_type == "lightgbm":
        model_path = MODELS_DIR / "Quantile" / "lightgbm_quantile" / f"conso_elec_mw_q50_{freq}_{suffix}.pkl"
        model = joblib.load(model_path)
    else:
        raise ValueError("model_type must be 'xgboost', 'reg_lin' or 'lightgbm'.")

    # === Load test data and apply appropriate transformation
    df_test = pd.read_csv(TEST_DIR / f"test_{freq}.csv")
    df_test["date"] = pd.to_datetime(df_test["date"])

    if model_type in ["xgboost", "reg_lin"]:
        X = transform_regression_and_xgb(df_test, frequency=freq, save=False)
        X = X.drop(columns=["conso_elec_mw", "conso_gaz_mw"], errors="ignore")
    elif model_type == "lightgbm":
        X = transform_lightgbm_quantile(df_test, frequency=freq, save=False, lags=(lags == "with"))
        X = X.drop(columns=["conso_elec_mw", "conso_gaz_mw", "date"], errors="ignore")

        # Reorder columns to match training set
        with open(MODELS_DIR / "Quantile" / "lightgbm_quantile" / f"features_{freq}_{suffix}.json") as f:
            expected_features = json.load(f)
        X = X.reindex(columns=expected_features, fill_value=0)

    # === Auto-detect features if not provided
    if features_to_plot is None:
        features_to_plot = [col for col in X.columns if col not in ["conso_elec_mw", "conso_gaz_mw", "date"]]

    # === PDP Plot
    n_features = len(features_to_plot)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols  

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
    axes = axes.flatten() # type: ignore

    for idx, feature in enumerate(features_to_plot):
        ax = axes[idx]
        display = PartialDependenceDisplay.from_estimator(
            model,
            X,
            [feature],
            ax=ax,
            target=0,  # target = 0 for electricity consumption (target=1 for gas)
            percentiles=(0, 1)
        )
        ax.set_title(f"PDP: {feature}")

    for i in range(n_features, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
