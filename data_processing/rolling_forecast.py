import pandas as pd
import joblib
import json
from pathlib import Path
from data_processing.transformation import transform_lightgbm_quantile

BASE_DIR     = Path(__file__).resolve().parents[1]
MODELS_DIR   = BASE_DIR / "models" / "Quantile" / "lightgbm_quantile"
MODIFIED_DIR = BASE_DIR / "data" / "modified_data"

def rolling_forecast_lightgbm(freq="daily", lags="with"):
    suffix = "withlags" if lags == "with" else "withoutlags"
    
    # 1) Load train and test sets
    df_train = pd.read_csv(MODIFIED_DIR / f"train_{freq}.csv")
    df_test  = pd.read_csv(MODIFIED_DIR / f"test_{freq}.csv")
    for df in (df_train, df_test):
        df["date"] = pd.to_datetime(df["date"])

    # 2) Load used features
    with open(MODELS_DIR / f"features_{freq}_{suffix}.json") as f:
        expected_features = json.load(f)

    # 3) Concatenate train and test to build full historical data
    df_full = pd.concat([df_train, df_test], ignore_index=True)

    preds = []
    dates = sorted(df_test["date"].unique())

    for current_date in dates:

        df_window = df_full[df_full["date"] <= current_date].copy()

        # === Apply transformations ===
        df_trans = transform_lightgbm_quantile(df_window, frequency=freq, save=False, lags=(lags == "with"))
        df_trans["date"]         = df_window["date"].values
        df_trans["insee_region"] = df_window["insee_region"].values

        # Select rows to predict
        df_today_trans = df_trans[df_trans["date"] == current_date]

        # Make predictions by region
        regions_today = df_test[df_test["date"] == current_date]["insee_region"].unique()
        for region in regions_today:
            row = df_today_trans[df_today_trans["insee_region"] == region]
            if row.empty:
                continue

            X = row.drop(columns=["date", "insee_region", "conso_elec_mw", "conso_gaz_mw"], errors="ignore")
            X = X.reindex(columns=expected_features, fill_value=0)

            out = {
                "date": current_date,
                "insee_region": region
            }

            for target in ["conso_elec_mw", "conso_gaz_mw"]:
                for q in [10, 50, 90]:
                    model_path = MODELS_DIR / f"{target}_q{q}_{freq}_{suffix}.pkl"
                    mdl = joblib.load(model_path)
                    y_pred = float(mdl.predict(X)[0])
                    out[f"{target}_q{q}"] = y_pred

                    # Replace the "true" value in df_full by the median prediction (q50)
                    if q == 50 and lags == "with":
                        idx = (df_full["date"] == current_date) & (df_full["insee_region"] == region)
                        df_full.loc[idx, target] = y_pred

            preds.append(out)

    # 4) Save predictions
    df_preds = pd.DataFrame(preds)
    out_path = MODELS_DIR / f"preds_{freq}_rolling_{suffix}.csv"
    df_preds.to_csv(out_path, index=False)
    print(f"\nRolling forecast complete ({len(df_preds)} rows) → {out_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--frequency", choices=["daily", "hourly"], required=True)
    parser.add_argument("--lags", choices=["with", "without"], default="with")
    args = parser.parse_args()
    rolling_forecast_lightgbm(freq=args.frequency, lags=args.lags)
