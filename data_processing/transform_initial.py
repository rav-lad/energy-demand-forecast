"""Initial data transformation script for training data.

This script applies transformation pipeline to training data and fits/saves
the scaler for use in model training and prediction.
"""

import sys
from pathlib import Path

import pandas as pd

# Compute BASE_DIR = project root directory
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "modified_data"

# Local import
sys.path.append(str(BASE_DIR))
from data_processing.transformation import transform_regression_and_xgb

FREQ = "daily"  # or "hourly"

df_train = pd.read_csv(DATA_DIR / f"train_{FREQ}.csv")
transform_regression_and_xgb(df_train, frequency=FREQ, fit_scaler=True, save=True)

print("Data transformed and scaler saved successfully")
