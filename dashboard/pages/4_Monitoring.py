"""
MLOps Monitoring page — model performance drift, data drift (PSI/KS),
data quality, and retraining recommendations.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    MODEL_COLOR,
    MODEL_DISPLAY,
    UPDATE_STATUS,
    available_models,
    compute_ks,
    compute_psi,
    detect_trend,
    load_daily_metrics,
    load_dataset,
    load_metrics,
    load_quantile_predictions,
    model_file_info,
    psi_label,
)

st.set_page_config(layout="wide")
st.title("MLOps Monitoring")
st.caption(
    "Model performance drift, data drift detection, data quality, "
    "and retraining recommendations — all derived from real pipeline outputs."
)

# ── Load data ─────────────────────────────────────────────────────────────────
dataset_df   = load_dataset()
quantile_df  = load_quantile_predictions()
models_avail = available_models()
daily_dfs    = {n: load_daily_metrics(n) for n in models_avail}
daily_dfs    = {k: v for k, v in daily_dfs.items() if v is not None}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 1 — Model performance drift
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.subheader("1 · Model Performance Drift")

if not daily_dfs:
    st.info("No daily metrics found. Run `python scripts/train.py` first.")
else:
    trend_window = st.select_slider(
        "Trend detection window (days)", options=[14, 30, 60, 90], value=30,
        key="trend_win"
    )

    # ── Summary table ─────────────────────────────────────────────────────────
    summary_rows = []
    for name, dm in daily_dfs.items():
        dm  = dm.sort_values("datetime")
        mae_all   = dm["mae"].mean()
        mae_30    = dm.tail(30)["mae"].mean()
        mae_7     = dm.tail(7)["mae"].mean()
        # R² from overall metrics.txt (not mean of daily R², which is unstable on 24 points)
        overall_m = load_metrics(name)
        r2_all    = overall_m.get("r2", np.nan) if overall_m else np.nan
        da_all    = dm["direction_accuracy"].mean() if "direction_accuracy" in dm.columns else np.nan
        trend_res = detect_trend(dm["mae"], window=trend_window)
        slope     = trend_res["slope"]
        p_val     = trend_res["p_value"]

        if np.isnan(slope):
            trend_str = "N/A"
        elif slope > 0 and p_val < 0.05:
            trend_str = "⬆ Degrading"
        elif slope < 0 and p_val < 0.05:
            trend_str = "⬇ Improving"
        else:
            trend_str = "→ Stable"

        summary_rows.append({
            "Model":          MODEL_DISPLAY.get(name, name),
            "All-time MAE":   round(mae_all, 2),
            "30d MAE":        round(mae_30, 2),
            "7d MAE":         round(mae_7, 2),
            "Δ (30d vs all)": round(mae_30 - mae_all, 2),
            "All-time R²":    round(r2_all, 3),
            "Direction acc.": f"{da_all:.1f}%" if not np.isnan(da_all) else "N/A",
            f"Trend ({trend_window}d)": trend_str,
        })

    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # ── MAE over time (rolling) ───────────────────────────────────────────────
    roll_days = st.select_slider(
        "Rolling window for chart", options=[7, 14, 30], value=14, key="roll_perf"
    )
    fig = go.Figure()
    for name, dm in daily_dfs.items():
        dm_s = dm.sort_values("datetime")
        roll = dm_s["mae"].rolling(roll_days, min_periods=3).mean()
        fig.add_trace(go.Scatter(
            x=dm_s["datetime"], y=roll,
            mode="lines",
            name=MODEL_DISPLAY.get(name, name),
            line=dict(color=MODEL_COLOR.get(name, "#888"), width=2),
        ))

    # Anomaly markers for the reference model (worst daily MAE)
    ref_name = max(daily_dfs, key=lambda k: len(daily_dfs[k]))
    ref_dm   = daily_dfs[ref_name].sort_values("datetime")
    mean_mae = ref_dm["mae"].mean()
    std_mae  = ref_dm["mae"].std()
    anomaly_mask = ref_dm["mae"] > mean_mae + 2 * std_mae
    if anomaly_mask.any():
        fig.add_trace(go.Scatter(
            x=ref_dm[anomaly_mask]["datetime"],
            y=ref_dm[anomaly_mask]["mae"],
            mode="markers",
            name=f"Anomaly ({MODEL_DISPLAY.get(ref_name, ref_name)})",
            marker=dict(color="red", size=8, symbol="x"),
        ))

    fig.add_hline(y=mean_mae, line_dash="dot", line_color="gray",
                  annotation_text=f"All-time mean ({mean_mae:.1f})",
                  annotation_position="bottom right")
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title=f"{roll_days}-day rolling MAE (EUR/MWh)",
        height=340,
        margin=dict(t=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Direction accuracy over time ──────────────────────────────────────────
    has_da = any("direction_accuracy" in dm.columns for dm in daily_dfs.values())
    if has_da:
        with st.expander("Direction accuracy over time"):
            fig_da = go.Figure()
            for name, dm in daily_dfs.items():
                if "direction_accuracy" not in dm.columns:
                    continue
                dm_s = dm.sort_values("datetime")
                roll = dm_s["direction_accuracy"].rolling(14, min_periods=3).mean()
                fig_da.add_trace(go.Scatter(
                    x=dm_s["datetime"], y=roll,
                    mode="lines",
                    name=MODEL_DISPLAY.get(name, name),
                    line=dict(color=MODEL_COLOR.get(name, "#888"), width=2),
                ))
            fig_da.add_hline(y=50, line_dash="dot", line_color="red",
                             annotation_text="Random baseline (50%)")
            fig_da.update_layout(
                xaxis_title="Date",
                yaxis_title="14-day rolling direction accuracy (%)",
                yaxis=dict(range=[30, 100]),
                height=280, margin=dict(t=20),
            )
            st.plotly_chart(fig_da, use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 2 — Data drift detection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.subheader("2 · Data Drift Detection")

if dataset_df is None:
    st.info("Dataset not available.")
else:
    datetime_col = "datetime" if "datetime" in dataset_df.columns else "datetime_hour"
    n_total = len(dataset_df)

    col_ref, col_cur = st.columns(2)
    with col_ref:
        ref_pct = st.slider(
            "Reference window (% of data from start)", 30, 80, 70, step=10,
            key="ref_pct",
            help="Training-period reference — typically matches the initial train split.",
        )
    with col_cur:
        cur_days = st.select_slider(
            "Monitoring window (most recent days)", options=[30, 60, 90, 180], value=60,
            key="cur_days",
        )

    ref_size = int(n_total * ref_pct / 100)
    ref_data  = dataset_df.iloc[:ref_size]
    cur_data  = dataset_df.tail(cur_days * 24)

    st.caption(
        f"Reference: {len(ref_data):,} rows ({ref_pct}% of dataset)  ·  "
        f"Monitoring: {len(cur_data):,} rows (last {cur_days} days)"
    )

    drift_features = [
        c for c in (
            "price", "load_mw", "temperature_2m", "wind_speed_10m",
            "shortwave_radiation", "renewable_production_mw", "net_load_mw"
        )
        if c in dataset_df.columns
    ]

    drift_rows = []
    for feat in drift_features:
        ref_vals = ref_data[feat].dropna().values
        cur_vals = cur_data[feat].dropna().values
        psi       = compute_psi(ref_vals, cur_vals)
        ks_stat, ks_p = compute_ks(ref_vals, cur_vals)
        label, colour = psi_label(psi)
        drift_rows.append({
            "Feature":  feat,
            "PSI":      round(psi, 4) if not np.isnan(psi) else None,
            "Status":   label,
            "KS stat":  round(ks_stat, 4) if not np.isnan(ks_stat) else None,
            "KS p-value": round(ks_p, 4) if not np.isnan(ks_p) else None,
            "_color":   colour,
        })

    drift_df = pd.DataFrame(drift_rows)
    display_df = drift_df.drop(columns=["_color"])
    color_lookup = drift_df.set_index("Feature")["_color"]

    # Map status to readable label with icon (no background colours)
    status_icon = {"green": "✅ Stable", "orange": "⚠️ Moderate drift", "red": "🔴 Significant drift"}
    display_df = display_df.copy()
    display_df["Status"] = [status_icon.get(color_lookup.get(f, ""), display_df.loc[i, "Status"])
                            for i, f in enumerate(display_df["Feature"])]

    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.caption(
        "PSI < 0.1 → Stable · 0.1–0.25 → Moderate drift · > 0.25 → Significant drift. "
        "KS p-value < 0.05 → distributions differ significantly."
    )

    # ── Distribution comparison for the most-drifted feature ─────────────────
    valid_drift = drift_df.dropna(subset=["PSI"])
    if not valid_drift.empty:
        most_drifted = valid_drift.loc[valid_drift["PSI"].idxmax(), "Feature"]
        with st.expander(f"Distribution comparison — most drifted feature: **{most_drifted}**"):
            ref_vals_plot = ref_data[most_drifted].dropna().values
            cur_vals_plot = cur_data[most_drifted].dropna().values

            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=ref_vals_plot, name=f"Reference ({ref_pct}% of data)",
                opacity=0.6, nbinsx=50,
                histnorm="probability density",
            ))
            fig_dist.add_trace(go.Histogram(
                x=cur_vals_plot, name=f"Monitoring (last {cur_days}d)",
                opacity=0.6, nbinsx=50,
                histnorm="probability density",
            ))
            fig_dist.update_layout(
                barmode="overlay",
                xaxis_title=most_drifted,
                yaxis_title="Density",
                height=300, margin=dict(t=20),
            )
            st.plotly_chart(fig_dist, use_container_width=True)

    # ── Monthly PSI heatmap ───────────────────────────────────────────────────
    with st.expander("Monthly PSI heatmap — seasonal drift vs anomalous shifts"):
        if datetime_col in dataset_df.columns:
            dataset_df["_month"] = dataset_df[datetime_col].dt.to_period("M")
            months = sorted(dataset_df["_month"].dropna().unique())

            if len(months) > 1 and len(drift_features) > 0:
                ref_df_base = dataset_df.iloc[:ref_size]
                heatmap_data = []
                for m in months[len(months) // 2:]:  # Only test-period months
                    month_data = dataset_df[dataset_df["_month"] == m]
                    row = {"Month": str(m)}
                    for feat in drift_features:
                        ref_v = ref_df_base[feat].dropna().values
                        cur_v = month_data[feat].dropna().values
                        row[feat] = round(compute_psi(ref_v, cur_v), 3) if len(cur_v) > 10 else np.nan
                    heatmap_data.append(row)

                hm_df = pd.DataFrame(heatmap_data).set_index("Month")
                fig_hm = px.imshow(
                    hm_df.values,
                    x=hm_df.columns.tolist(),
                    y=hm_df.index.tolist(),
                    color_continuous_scale=[[0, "green"], [0.1, "green"],
                                            [0.1, "orange"], [0.25, "orange"],
                                            [0.25, "red"], [1, "red"]],
                    zmin=0, zmax=0.5,
                    labels={"color": "PSI"},
                    text_auto=".2f",
                )
                fig_hm.update_layout(height=max(200, len(heatmap_data) * 28), margin=dict(t=20))
                st.plotly_chart(fig_hm, use_container_width=True)
                st.caption(
                    "Seasonal patterns produce predictable drift (e.g., summer solar vs winter). "
                    "Anomalous months stand out against the seasonal baseline."
                )

            dataset_df.drop(columns=["_month"], inplace=True, errors="ignore")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 3 — Data quality
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.subheader("3 · Data Quality")

if dataset_df is None:
    st.info("Dataset not available.")
else:
    dq_window = st.select_slider(
        "Quality check window (days)", options=[7, 30, 90], value=30, key="dq_win"
    )
    recent = dataset_df.tail(dq_window * 24)

    # ── Source freshness ──────────────────────────────────────────────────────
    if UPDATE_STATUS.exists():
        try:
            status = json.loads(UPDATE_STATUS.read_text())
            last_run_str = status.get("last_run", "")
            if last_run_str:
                last_run = datetime.fromisoformat(last_run_str)
                age_days = (datetime.now() - last_run).days
                if age_days == 0:
                    freshness, f_color = "Today", "green"
                elif age_days == 1:
                    freshness, f_color = "Yesterday", "orange"
                else:
                    freshness, f_color = f"{age_days} days ago", "red"
            else:
                freshness, f_color = "Unknown", "gray"
        except Exception:
            freshness, f_color = "Unknown", "gray"

        col1, col2, col3 = st.columns(3)
        col1.metric("Last pipeline run", freshness)
        col2.metric("Dataset build OK", "✓" if status.get("build_ok") else "✗")
        col3.metric("Dataset rows", f"{status.get('dataset_rows', '?'):,}" if isinstance(status.get('dataset_rows'), int) else "?")

    # ── Missing values ────────────────────────────────────────────────────────
    # Only check columns actually present in the dataset (load_mw is excluded from dataset.csv)
    key_features = [
        c for c in ("price", "temperature_2m", "wind_speed_10m",
                    "renewable_production_mw", "shortwave_radiation")
        if c in recent.columns
    ]
    missing_rows = []
    for feat in key_features:
        n_missing = recent[feat].isna().sum()
        missing_pct = n_missing / len(recent) * 100
        missing_rows.append({
            "Feature":     feat,
            "Missing (n)": int(n_missing),
            "Missing (%)": round(missing_pct, 2),
            "Status":      "✓ OK" if missing_pct < 1 else "⚠ Check" if missing_pct < 5 else "✗ Critical",
        })
    st.dataframe(pd.DataFrame(missing_rows), use_container_width=True, hide_index=True)

    # ── Input anomalies (z-score) ─────────────────────────────────────────────
    with st.expander("Input anomaly detection (|z-score| > 3.5)"):
        anomaly_rows = []
        for feat in key_features:
            col_data = recent[feat].dropna()
            if len(col_data) < 10:
                continue
            z = (col_data - col_data.mean()) / (col_data.std() + 1e-9)
            n_anom = (z.abs() > 3.5).sum()
            anomaly_rows.append({
                "Feature":          feat,
                "Anomalies (n)":    int(n_anom),
                "Rate (%)":         round(n_anom / len(col_data) * 100, 2),
                "Mean":             round(col_data.mean(), 2),
                "Std":              round(col_data.std(), 2),
            })
        if anomaly_rows:
            st.dataframe(pd.DataFrame(anomaly_rows), use_container_width=True, hide_index=True)
        st.caption(
            "Anomalies are hours where the feature value deviates more than 3.5 standard "
            "deviations from the recent mean. These may be sensor errors or extreme market events."
        )

    # ── Data health score ─────────────────────────────────────────────────────
    if missing_rows:
        # Freshness: 100 if today, 70 if yesterday, 30 if older
        freshness_score = 100 if f_color == "green" else (70 if f_color == "orange" else 30)
        # Completeness: mean completeness across key features (not penalised by one bad column)
        avg_missing = np.mean([r["Missing (%)"] for r in missing_rows])
        completeness_score = max(0, 100 - avg_missing * 2)
        health_score = int(freshness_score * 0.5 + completeness_score * 0.5)

        st.metric(
            "Data health score",
            f"{health_score} / 100",
            help=(
                "50% freshness (pipeline last run) + "
                f"50% completeness (avg missing rate: {avg_missing:.1f}% across key features)."
            ),
        )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 4 — Retraining recommendations
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.subheader("4 · Retraining Recommendations")

triggers = []

# Trigger A — performance degradation
if daily_dfs:
    ref_name = max(daily_dfs, key=lambda k: len(daily_dfs[k]))
    dm_ref   = daily_dfs[ref_name].sort_values("datetime")
    mae_90   = dm_ref.tail(90)["mae"].mean()
    mae_30   = dm_ref.tail(30)["mae"].mean()
    degradation_pct = (mae_30 - mae_90) / (mae_90 + 1e-9) * 100

    if mae_90 > 0:
        status_a = "✗ Triggered" if degradation_pct > 20 else "✓ OK"
        triggers.append({
            "Trigger": "A — Performance degradation",
            "Current value": f"{degradation_pct:+.1f}% (30d MAE vs 90d MAE)",
            "Threshold": "> +20%",
            "Status": status_a,
        })

# Trigger B — data drift
if dataset_df is not None and drift_features:
    max_psi = 0.0
    max_feat = ""
    ref_df_b  = dataset_df.iloc[:ref_size]
    cur_df_b  = dataset_df.tail(60 * 24)
    for feat in drift_features:
        psi = compute_psi(ref_df_b[feat].dropna().values, cur_df_b[feat].dropna().values)
        if not np.isnan(psi) and psi > max_psi:
            max_psi  = psi
            max_feat = feat

    status_b = "✗ Triggered" if max_psi > 0.25 else ("⚠ Monitor" if max_psi > 0.1 else "✓ OK")
    triggers.append({
        "Trigger": "B — Data drift",
        "Current value": f"PSI = {max_psi:.3f} ({max_feat})",
        "Threshold": "PSI > 0.25",
        "Status": status_b,
    })

# Trigger C — quantile calibration drift
if quantile_df is not None and "in_interval" in quantile_df.columns:
    recent_cov = quantile_df.tail(30 * 24)["in_interval"].mean() * 100
    cov_drift  = abs(recent_cov - 80.0)
    status_c   = "✗ Triggered" if cov_drift > 10 else "✓ OK"
    triggers.append({
        "Trigger": "C — Calibration drift (quantile intervals)",
        "Current value": f"{recent_cov:.0f}% coverage (target 80%)",
        "Threshold": "Deviation > 10pp from 80%",
        "Status": status_c,
    })

# Trigger D — model staleness
for name in models_avail:
    info = model_file_info(name)
    if info:
        age_days = (datetime.now() - info["last_trained"]).days
        status_d = "✗ Triggered" if age_days > 30 else "✓ OK"
        triggers.append({
            "Trigger": f"D — Model staleness ({MODEL_DISPLAY.get(name, name)})",
            "Current value": f"{age_days} days since last train",
            "Threshold": "> 30 days",
            "Status": status_d,
        })

if triggers:
    t_df = pd.DataFrame(triggers)

    # Replace status symbols with readable icons
    t_df["Status"] = t_df["Status"].replace({
        "✗ Triggered": "🔴 Triggered",
        "⚠ Monitor":   "🟡 Monitor",
        "✓ OK":        "✅ OK",
    })

    st.dataframe(t_df, use_container_width=True, hide_index=True)

    triggered = [t for t in triggers if "✗" in t["Status"]]
    monitoring = [t for t in triggers if "⚠" in t["Status"]]

    if triggered:
        st.error(
            f"**Retraining recommended.** "
            f"{len(triggered)} trigger(s) fired: "
            + ", ".join(t['Trigger'].split('—')[0].strip() for t in triggered)
            + ".\n\nRun: `python scripts/train.py`"
        )
    elif monitoring:
        st.warning(
            "**Monitor closely.** "
            f"{len(monitoring)} trigger(s) in warning zone. "
            "Consider retraining if the trend continues."
        )
    else:
        st.success(
            "**No retraining needed.** All triggers are within acceptable bounds."
        )
else:
    st.info("Not enough data to evaluate retraining triggers.")
