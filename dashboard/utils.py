"""
Shared utilities for the Energy Price Forecast DSS dashboard.
Centralises path constants, data loaders, and statistical helpers.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
REPORTS     = ROOT / "outputs" / "reports"
MODELS      = ROOT / "outputs" / "models"
PREDICTIONS = ROOT / "outputs" / "predictions"
DATASET     = ROOT / "data" / "dataset.csv"
UPDATE_STATUS = ROOT / "data" / ".update_status.json"

# ── Model display constants ────────────────────────────────────────────────────
MODEL_DISPLAY = {"xgboost": "XGBoost", "ridge": "Ridge", "lightgbm": "LightGBM"}
MODEL_COLOR   = {"xgboost": "#EF553B", "ridge": "#636EFA", "lightgbm": "#00CC96"}


# ── Data loaders ───────────────────────────────────────────────────────────────

def load_metrics(name: str) -> dict | None:
    """Load overall walk-forward metrics from {name}_metrics.txt."""
    p = REPORTS / f"{name}_metrics.txt"
    if not p.exists():
        return None
    m: dict = {}
    for line in p.read_text().splitlines():
        if ":" in line and not line.startswith("=") and not line.startswith("Walk"):
            k, v = line.split(":", 1)
            try:
                m[k.strip()] = float(v.strip())
            except ValueError:
                m[k.strip()] = v.strip()
    return m


def load_predictions(name: str) -> pd.DataFrame | None:
    """Load hourly walk-forward predictions from {name}_predictions.csv."""
    p = REPORTS / f"{name}_predictions.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        if "datetime_hour" not in df.columns:
            return None
        df["datetime_hour"] = pd.to_datetime(df["datetime_hour"])
        return df
    except Exception:
        return None


def load_daily_metrics(name: str) -> pd.DataFrame | None:
    """Load per-day walk-forward metrics from {name}_daily_metrics.csv."""
    p = REPORTS / f"{name}_daily_metrics.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        if "datetime" not in df.columns:
            return None
        df["datetime"] = pd.to_datetime(df["datetime"])
        return df
    except Exception:
        return None


def load_quantile_predictions() -> pd.DataFrame | None:
    """Load LightGBM quantile walk-forward predictions (p10, p50, p90)."""
    p = REPORTS / "lightgbm_quantile_predictions.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        df["datetime_hour"] = pd.to_datetime(df["datetime_hour"])
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception:
        return None


def load_live_predictions() -> pd.DataFrame | None:
    """Load live inference output from predictions/predictions.csv."""
    p = PREDICTIONS / "predictions.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, parse_dates=["prediction_made_at", "target_datetime"])
        return df
    except Exception:
        return None


@st.cache_data(ttl=60)
def load_dataset() -> pd.DataFrame | None:
    """Load the main dataset.csv with 60-second cache."""
    if not DATASET.exists():
        return None
    try:
        df = pd.read_csv(DATASET)
        for col in ("datetime", "datetime_hour"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
                break
        return df
    except Exception:
        return None


def available_models() -> list[str]:
    """Return list of models that have metrics files."""
    return [
        name for name in ("xgboost", "lightgbm", "ridge")
        if (REPORTS / f"{name}_metrics.txt").exists()
    ]


def model_file_info(name: str) -> dict:
    """Return last_trained datetime, size_kb, and file name for a saved model."""
    for ext in (".json", ".pkl"):
        p = MODELS / f"{name}_final{ext}"
        if p.exists():
            return {
                "last_trained": datetime.fromtimestamp(os.path.getmtime(p)),
                "size_kb": p.stat().st_size // 1024,
                "path": p.name,
            }
    return {}


def get_dataset_dates() -> dict:
    """Return dataset_start and dataset_end timestamps."""
    if not DATASET.exists():
        return {}
    for col in ("datetime", "datetime_hour"):
        try:
            df = pd.read_csv(DATASET, usecols=[col], parse_dates=[col])
            return {"dataset_start": df[col].min(), "dataset_end": df[col].max()}
        except ValueError:
            continue
    return {}


# ── Statistical helpers ────────────────────────────────────────────────────────

def compute_psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index between two distributions.

    PSI < 0.1   → no significant shift (green)
    PSI 0.1–0.25 → moderate shift (amber)
    PSI > 0.25  → significant shift (red)
    """
    ref = reference[~np.isnan(reference)]
    cur = current[~np.isnan(current)]
    if len(ref) == 0 or len(cur) == 0:
        return np.nan

    # Bin edges from the reference distribution
    _, edges = np.histogram(ref, bins=bins)
    edges[0]  = -np.inf
    edges[-1] = np.inf

    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)

    ref_pct = ref_counts / len(ref) + 1e-6
    cur_pct = cur_counts / len(cur) + 1e-6

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def compute_ks(reference: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """
    Kolmogorov-Smirnov two-sample test.
    Returns (statistic, p_value). Low p-value → distributions differ.
    """
    ref = reference[~np.isnan(reference)]
    cur = current[~np.isnan(current)]
    if len(ref) == 0 or len(cur) == 0:
        return np.nan, np.nan
    result = stats.ks_2samp(ref, cur)
    return float(result.statistic), float(result.pvalue)


def detect_trend(series: pd.Series, window: int | None = None) -> dict:
    """
    Fit a linear trend to `series` (optionally restricted to last `window` points).
    Returns slope, intercept, r_squared, p_value.
    Positive slope → degrading (for error metrics); negative → improving.
    """
    s = series.dropna()
    if window is not None:
        s = s.iloc[-window:]
    if len(s) < 5:
        return {"slope": np.nan, "p_value": np.nan, "r_squared": np.nan}
    x = np.arange(len(s), dtype=float)
    slope, _, r, p, _ = stats.linregress(x, s.values)
    return {"slope": float(slope), "p_value": float(p), "r_squared": float(r ** 2)}


def psi_label(psi: float) -> tuple[str, str]:
    """Return (status_text, colour) for a PSI value."""
    if np.isnan(psi):
        return "N/A", "gray"
    if psi < 0.1:
        return "Stable", "green"
    if psi < 0.25:
        return "Moderate drift", "orange"
    return "Significant drift", "red"
