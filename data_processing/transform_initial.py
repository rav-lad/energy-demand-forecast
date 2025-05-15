# transform_initial.py
import pandas as pd
from pathlib import Path
import sys

# Compute BASE_DIR = project root directory (contains "data" and "data_processing")
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "modified_data"

# Local import
sys.path.append(str(BASE_DIR))
from data_processing.transformation import transform_regression_and_xgb

FREQ = "daily"  # or "hourly"

df_train = pd.read_csv(DATA_DIR / f"train_{FREQ}.csv")
transform_regression_and_xgb(df_train, frequency=FREQ, fit_scaler=True, save=True)

print("Data transformed and scaler saved successfully")
