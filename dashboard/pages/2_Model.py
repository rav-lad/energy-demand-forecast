"""
Model page — live inference, train/test dates, walk-forward metrics,
actual vs predicted, error analysis, feature importance, model metadata.
"""

import os
import subprocess
import threading
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(layout="wide")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent.parent
REPORTS     = ROOT / "outputs/reports"
MODELS      = ROOT / "outputs/models"
PREDICTIONS = ROOT / "outputs/predictions"
DATASET     = ROOT / "data/dataset.csv"

# ── Helpers ────────────────────────────────────────────────────────────────────

MODEL_DISPLAY = {"xgboost": "XGBoost", "ridge": "Ridge", "lightgbm": "LightGBM"}
MODEL_COLOR   = {"xgboost": "#EF553B", "ridge": "#636EFA", "lightgbm": "#00CC96"}


def load_metrics(name: str) -> dict | None:
    p = REPORTS / f"{name}_metrics.txt"
    if not p.exists():
        return None
    m = {}
    for line in p.read_text().splitlines():
        if ":" in line and not line.startswith("=") and not line.startswith("Walk"):
            k, v = line.split(":", 1)
            try:
                m[k.strip()] = float(v.strip())
            except ValueError:
                m[k.strip()] = v.strip()
    return m


def load_predictions(name: str) -> pd.DataFrame | None:
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


def model_file_info(name: str) -> dict:
    for ext in [".json", ".pkl"]:
        p = MODELS / f"{name}_final{ext}"
        if p.exists():
            mtime = os.path.getmtime(p)
            size_kb = p.stat().st_size // 1024
            return {
                "last_trained": datetime.fromtimestamp(mtime),
                "size_kb": size_kb,
                "path": p.name,
            }
    return {}


def available_models() -> list[str]:
    found = []
    for name in ["xgboost", "lightgbm", "ridge"]:
        if (REPORTS / f"{name}_metrics.txt").exists():
            found.append(name)
    return found


def load_live_predictions() -> pd.DataFrame | None:
    p = PREDICTIONS / "predictions.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p, parse_dates=["prediction_made_at", "target_datetime"])
    return df


def get_dataset_dates() -> dict:
    """Return first/last dates of the full dataset."""
    if not DATASET.exists():
        return {}
    # dataset.csv uses 'datetime', processed uses 'datetime_hour'
    for col in ("datetime", "datetime_hour"):
        try:
            df = pd.read_csv(DATASET, usecols=[col], parse_dates=[col])
            return {
                "dataset_start": df[col].min(),
                "dataset_end":   df[col].max(),
            }
        except ValueError:
            continue
    return {}


# ── Page ───────────────────────────────────────────────────────────────────────

st.title("Model monitoring")

models = available_models()
if not models:
    st.error("No trained model found. Run `python scripts/train.py` first.")
    st.stop()

# ── Selector ──────────────────────────────────────────────────────────────────
selected = st.selectbox(
    "Model",
    options=models,
    format_func=lambda x: MODEL_DISPLAY.get(x, x),
)

metrics     = load_metrics(selected)
preds_df    = load_predictions(selected)
daily_df    = load_daily_metrics(selected)
model_info  = model_file_info(selected)
color       = MODEL_COLOR.get(selected, "#888")
ds_dates    = get_dataset_dates()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 0 — Live inference (most important — shown first)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.subheader("Live inference — latest prediction")

live_df = load_live_predictions()

# ── Refresh button ────────────────────────────────────────────────────────────
dataset_last = ds_dates.get("dataset_end")
today = datetime.now().date()
data_gap_days = (pd.Timestamp(today) - dataset_last).days if dataset_last else None

col_btn, col_status = st.columns([1, 3])
with col_btn:
    run_pipeline = st.button("Refresh data + predict J+1", type="primary", use_container_width=True)

if run_pipeline:
    log_placeholder = col_status.empty()
    log_lines = []

    def stream_script(script):
        proc = subprocess.Popen(
            ["python", str(ROOT / script)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(ROOT),
        )
        for line in proc.stdout:
            log_lines.append(line.rstrip())
            log_placeholder.code("\n".join(log_lines[-20:]))  # show last 20 lines
        proc.wait()
        return proc.returncode

    with st.spinner("Updating data..."):
        rc1 = stream_script("scripts/update_data.py")
    if rc1 == 0:
        with st.spinner("Running inference J+1..."):
            rc2 = stream_script("scripts/infer.py")
        if rc2 == 0:
            st.cache_data.clear()
            st.success("Done — prediction updated.")
            st.rerun()
        else:
            st.error("Inference failed — check logs above.")
    else:
        st.error("Data update failed — check logs above.")

    live_df = load_live_predictions()  # reload after run

elif data_gap_days is not None and data_gap_days > 7:
    col_status.warning(
        f"Dataset ends **{dataset_last.date()}** ({data_gap_days}d ago) — "
        f"ODRE publishes with ~5 weeks delay. Click the button to try refreshing.",
    )

if live_df is None or live_df.empty:
    st.info("No inference run yet. Click the button above.", icon="ℹ️")
else:
    latest_run   = live_df["prediction_made_at"].max()
    latest_batch = live_df[live_df["prediction_made_at"] == latest_run].copy()
    target_date  = latest_batch["target_datetime"].dt.date.iloc[0]
    age_min = int((datetime.now() - latest_run).total_seconds() / 60)
    age_str = f"{age_min}min ago" if age_min < 120 else f"{age_min // 60}h ago"

    # ── Metrics row ───────────────────────────────────────────────────────────
    li1, li2, li3, li4 = st.columns(4)
    li1.metric("Forecast for", str(target_date))
    li2.metric("Prediction made at", latest_run.strftime("%Y-%m-%d %H:%M"), age_str)

    best_pred_col = next(
        (c for c in [f"pred_{selected}", "pred_ensemble", "pred_xgboost", "pred_lightgbm", "pred_ridge"]
         if c in latest_batch.columns),
        None
    )
    if best_pred_col and best_pred_col != f"pred_{selected}" and selected != "xgboost":
        st.caption(
            f"⚠️ {MODEL_DISPLAY.get(selected, selected)} was skipped at inference (NaN lag features — data gap). "
            f"Showing {best_pred_col.replace('pred_', '')} forecast instead."
        )
    if best_pred_col:
        ens = latest_batch[best_pred_col].dropna()
        if not ens.empty:
            li3.metric("Predicted mean", f"{ens.mean():.1f} €/MWh")
            li4.metric(
                "Peak / trough",
                f"{ens.max():.1f} / {ens.min():.1f} €/MWh",
                f"spread {ens.max()-ens.min():.1f} €/MWh",
            )

    # ── Combined chart: recent actual prices + walk-forward predictions + live forecast ──
    n_days_back = st.slider("Days of history to show", min_value=3, max_value=60, value=14)

    cutoff = pd.Timestamp.now() - pd.Timedelta(days=n_days_back)

    # Actual prices: walk-forward period first, then raw dataset for recent weeks
    if preds_df is not None:
        recent_wf = preds_df[preds_df["datetime_hour"] >= cutoff][["datetime_hour", "actual", "predicted"]].copy()
    else:
        recent_wf = pd.DataFrame(columns=["datetime_hour", "actual", "predicted"])

    # Load raw prices from dataset.csv for dates after walk-forward ends
    wf_end = preds_df["datetime_hour"].max() if preds_df is not None else pd.Timestamp("2000-01-01")
    try:
        df_ds = pd.read_csv(DATASET, usecols=["datetime", "price"], parse_dates=["datetime"])
        df_ds = df_ds.rename(columns={"datetime": "datetime_hour", "price": "actual"})
        df_ds = df_ds[(df_ds["datetime_hour"] > wf_end) & (df_ds["datetime_hour"] >= cutoff)]
        df_ds["predicted"] = float("nan")
    except Exception:
        df_ds = pd.DataFrame(columns=["datetime_hour", "actual", "predicted"])

    recent = pd.concat([recent_wf, df_ds], ignore_index=True).sort_values("datetime_hour")

    fig_combined = go.Figure()

    # Actual price (solid dark)
    if not recent.empty:
        fig_combined.add_trace(go.Scatter(
            x=recent["datetime_hour"], y=recent["actual"],
            name="Actual price",
            line=dict(color="#222", width=2),
        ))
        # Walk-forward prediction over the same window (dashed, model color)
        fig_combined.add_trace(go.Scatter(
            x=recent_wf["datetime_hour"], y=recent_wf["predicted"],
            name=f"{MODEL_DISPLAY.get(selected, selected)} — walk-forward",
            line=dict(color=color, width=1.5, dash="dash"),
            opacity=0.8,
        ))

    # All inference predictions in the window (one per target hour — latest run wins)
    if best_pred_col and live_df is not None:
        all_preds = live_df[live_df["target_datetime"] >= cutoff].copy()
        if best_pred_col not in all_preds.columns:
            # fallback: pick first available pred column
            fallback_cols = [c for c in ["pred_ensemble", "pred_xgboost", "pred_lightgbm", "pred_ridge"] if c in all_preds.columns]
            if fallback_cols:
                all_preds = all_preds.rename(columns={fallback_cols[0]: best_pred_col})
        if best_pred_col in all_preds.columns:
            # Keep most recent prediction per target_datetime
            all_preds = (
                all_preds.sort_values("prediction_made_at")
                .drop_duplicates("target_datetime", keep="last")
                .sort_values("target_datetime")
            )
            live_clean = all_preds[["target_datetime", best_pred_col]].dropna()
            if not live_clean.empty:
                pred_label = best_pred_col.replace("pred_", "").capitalize()
                fig_combined.add_trace(go.Scatter(
                    x=live_clean["target_datetime"],
                    y=live_clean[best_pred_col],
                    name=f"{pred_label} — inference (J+1)",
                    line=dict(color=color, width=1.5, dash="dot"),
                    opacity=0.9,
                ))

    # Vertical line at "now" / last known price
    if not recent.empty:
        last_known = recent["datetime_hour"].max().isoformat()
        fig_combined.add_shape(
            type="line",
            x0=last_known, x1=last_known,
            y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(color="gray", width=1, dash="dot"),
        )
        fig_combined.add_annotation(
            x=last_known, y=1, xref="x", yref="paper",
            text="last known", showarrow=False,
            xanchor="left", yanchor="top",
            font=dict(color="gray", size=11),
        )

    fig_combined.update_layout(
        height=360,
        margin=dict(t=20, b=30),
        yaxis_title="Price (€/MWh)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    st.plotly_chart(fig_combined, use_container_width=True)

    with st.expander("Previous inference runs"):
        agg_col = best_pred_col if best_pred_col else live_df.columns[-1]
        prev = (
            live_df.groupby(["prediction_made_at", "target_date"])
            .agg(mean_pred=(agg_col, "mean"), n_hours=("target_datetime", "count"))
            .reset_index()
            .sort_values("prediction_made_at", ascending=False)
        )
        prev["prediction_made_at"] = prev["prediction_made_at"].dt.strftime("%Y-%m-%d %H:%M")
        prev.columns = ["Made at", "Target date", "Mean pred (€/MWh)", "Hours"]
        st.dataframe(prev, use_container_width=True, hide_index=True)

st.divider()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 1 — Train / Test date summary
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.subheader("Train / test split")

# Derive train/test boundaries from predictions file
_ds_start = ds_dates.get("dataset_start")
_ds_end   = ds_dates.get("dataset_end")
train_start = pd.Timestamp(_ds_start) if _ds_start else None
test_end    = pd.Timestamp(_ds_end)   if _ds_end   else None
test_start  = preds_df["datetime_hour"].min() if preds_df is not None else None
train_end   = preds_df["datetime_hour"].min() - timedelta(hours=1) if preds_df is not None else None

ts1, ts2, ts3, ts4 = st.columns(4)
ts1.metric(
    "Train start",
    train_start.strftime("%Y-%m-%d") if train_start else "—",
    help="First row in the processed dataset",
)
ts2.metric(
    "Train end",
    train_end.strftime("%Y-%m-%d") if train_end else "—",
    f"{int((train_end - train_start).days)} days" if train_end and train_start else "",
    help="Last hour seen before the walk-forward test period begins",
)
ts3.metric(
    "Test start",
    test_start.strftime("%Y-%m-%d") if test_start else "—",
    help="First walk-forward prediction date",
)
ts4.metric(
    "Test end",
    test_end.strftime("%Y-%m-%d") if test_end else "—",
    f"{int((test_end - test_start).days)} days" if test_end and test_start else "",
    help="Last date in the dataset",
)

# Visual timeline
if train_start and train_end and test_start and test_end:
    fig_tl = go.Figure()
    fig_tl.add_trace(go.Bar(
        x=[(train_end - train_start).days],
        y=[""],
        base=[(train_start - train_start).days],
        orientation="h",
        marker_color="#AAAAAA",
        name=f"Train ({int((train_end - train_start).days)}d)",
        hovertemplate=f"Train: {train_start.date()} → {train_end.date()}<extra></extra>",
    ))
    fig_tl.add_trace(go.Bar(
        x=[(test_end - test_start).days],
        y=[""],
        base=[(test_start - train_start).days],
        orientation="h",
        marker_color=color,
        name=f"Test ({int((test_end - test_start).days)}d)",
        hovertemplate=f"Test: {test_start.date()} → {test_end.date()}<extra></extra>",
    ))
    fig_tl.update_layout(
        barmode="stack",
        height=90,
        margin=dict(t=10, b=10, l=10, r=10),
        showlegend=True,
        legend=dict(orientation="h", x=0, y=1.6),
        xaxis=dict(
            tickvals=[
                0,
                (test_start - train_start).days,
                (test_end - train_start).days,
            ],
            ticktext=[
                str(train_start.date()),
                str(test_start.date()),
                str(test_end.date()),
            ],
        ),
        yaxis=dict(showticklabels=False),
    )
    st.plotly_chart(fig_tl, use_container_width=True)

st.divider()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 2 — Model metadata
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.subheader("Model info")

c1, c2, c3, c4 = st.columns(4)

if model_info:
    last_trained = model_info["last_trained"]
    age_hours = (datetime.now() - last_trained).total_seconds() / 3600
    age_str = (
        f"{int(age_hours)}h ago"
        if age_hours < 48
        else f"{int(age_hours/24)}d ago"
    )
    c1.metric("Last trained", last_trained.strftime("%Y-%m-%d %H:%M"), age_str)
    c2.metric("Model file", model_info["path"], f"{model_info['size_kb']} KB")
else:
    c1.metric("Last trained", "—")
    c2.metric("Model file", "—")

if metrics:
    n_pred = int(metrics.get("n_predictions", 0))
    n_iter = int(metrics.get("n_iterations", 0))
    cov_h  = metrics.get("coverage_hours", 0)
    c3.metric("Walk-forward predictions", f"{n_pred:,} h", f"{n_iter} iterations")
    c4.metric("Test period covered", f"{int(cov_h/24)} days")

# ── Section 2 — Key metrics ───────────────────────────────────────────────────
st.subheader("Walk-forward performance")

if metrics:
    m1, m2, m3, m4, m5 = st.columns(5)
    mae  = metrics.get("mae", float("nan"))
    rmse = metrics.get("rmse", float("nan"))
    r2   = metrics.get("r2", float("nan"))
    da   = metrics.get("direction_accuracy", float("nan"))
    bias = metrics.get("bias", float("nan"))

    m1.metric("MAE",  f"{mae:.2f} €/MWh")
    m2.metric("RMSE", f"{rmse:.2f} €/MWh")
    m3.metric("R²",   f"{r2:.3f}")
    m4.metric("Direction accuracy", f"{da:.1f}%")
    m5.metric("Bias", f"{bias:+.2f} €/MWh",
              help="Positive = over-predicts, Negative = under-predicts")

    # Error quantiles
    with st.expander("Error distribution details"):
        ec1, ec2, ec3, ec4 = st.columns(4)
        ec1.metric("Error std",       f"{metrics.get('error_std', 0):.2f} €/MWh")
        ec2.metric("P5 error",        f"{metrics.get('error_q05', 0):.2f} €/MWh")
        ec3.metric("P95 error",       f"{metrics.get('error_q95', 0):.2f} €/MWh")
        ec4.metric("Max over-pred",   f"{metrics.get('max_overpredict', 0):.2f} €/MWh")

# ── Section 3 — Compare all models ───────────────────────────────────────────
with st.expander("Compare all models", expanded=False):
    rows = []
    for m in models:
        mx = load_metrics(m)
        if mx:
            info = model_file_info(m)
            rows.append({
                "Model": MODEL_DISPLAY.get(m, m),
                "MAE (€/MWh)": round(mx.get("mae", float("nan")), 2),
                "RMSE (€/MWh)": round(mx.get("rmse", float("nan")), 2),
                "R²": round(mx.get("r2", float("nan")), 3),
                "Direction acc.": f"{mx.get('direction_accuracy', 0):.1f}%",
                "Bias": f"{mx.get('bias', 0):+.2f}",
                "Last trained": info.get("last_trained", "—"),
            })
    if rows:
        st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

# ── Section 4 — Actual vs Predicted ──────────────────────────────────────────
st.subheader("Actual vs predicted price")

if preds_df is not None:
    # Date range picker
    min_date = preds_df["datetime_hour"].dt.date.min()
    max_date = preds_df["datetime_hour"].dt.date.max()

    col_l, col_r = st.columns([3, 1])
    with col_r:
        date_range = st.date_input(
            "Period",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )
        agg = st.radio("Aggregation", ["Hourly", "Daily mean"], horizontal=True)

    start, end = (date_range[0], date_range[1]) if len(date_range) == 2 else (min_date, max_date)
    mask = (preds_df["datetime_hour"].dt.date >= start) & (preds_df["datetime_hour"].dt.date <= end)
    view = preds_df[mask].copy()

    if agg == "Daily mean":
        view = (
            view.groupby(view["datetime_hour"].dt.date)
            .agg(actual=("actual", "mean"), predicted=("predicted", "mean"), error=("error", "mean"))
            .reset_index()
            .rename(columns={"datetime_hour": "datetime_hour"})
        )
        view["datetime_hour"] = pd.to_datetime(view["datetime_hour"])

    with col_l:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=view["datetime_hour"], y=view["actual"],
            name="Actual", line=dict(color="#444", width=1.5), opacity=0.85,
        ))
        fig.add_trace(go.Scatter(
            x=view["datetime_hour"], y=view["predicted"],
            name="Predicted", line=dict(color=color, width=1.5, dash="dot"), opacity=0.85,
        ))
        fig.update_layout(
            height=320, margin=dict(t=20, b=30),
            yaxis_title="Price (€/MWh)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Scatter actual vs predicted
    st.markdown("**Scatter — actual vs predicted**")
    sc1, sc2 = st.columns(2)
    with sc1:
        fig_sc = px.scatter(
            view, x="actual", y="predicted",
            opacity=0.4, color_discrete_sequence=[color],
            labels={"actual": "Actual (€/MWh)", "predicted": "Predicted (€/MWh)"},
        )
        mn = min(view["actual"].min(), view["predicted"].min())
        mx2 = max(view["actual"].max(), view["predicted"].max())
        fig_sc.add_trace(go.Scatter(
            x=[mn, mx2], y=[mn, mx2],
            mode="lines", line=dict(color="gray", dash="dash", width=1),
            name="Perfect", showlegend=False,
        ))
        fig_sc.update_layout(height=300, margin=dict(t=20, b=30))
        st.plotly_chart(fig_sc, use_container_width=True)

    with sc2:
        fig_err = px.histogram(
            view, x="error", nbins=60,
            color_discrete_sequence=[color],
            labels={"error": "Prediction error (€/MWh)"},
            title="Error distribution",
        )
        fig_err.add_vline(x=0, line_dash="dash", line_color="gray")
        fig_err.add_vline(x=view["error"].mean(), line_dash="dot", line_color="red",
                          annotation_text=f"mean={view['error'].mean():.1f}")
        fig_err.update_layout(height=300, margin=dict(t=40, b=30), showlegend=False)
        st.plotly_chart(fig_err, use_container_width=True)

# ── Section 5 — Error over time (daily MAE) ──────────────────────────────────
st.subheader("MAE evolution over the test period")

if daily_df is not None:
    # Rolling 7-day smoothing
    daily_df = daily_df.sort_values("datetime")
    daily_df["mae_7d"] = daily_df["mae"].rolling(7, min_periods=1).mean()

    fig_mae = go.Figure()
    fig_mae.add_trace(go.Scatter(
        x=daily_df["datetime"], y=daily_df["mae"],
        name="Daily MAE", line=dict(color=color, width=1), opacity=0.4,
    ))
    fig_mae.add_trace(go.Scatter(
        x=daily_df["datetime"], y=daily_df["mae_7d"],
        name="7-day rolling", line=dict(color=color, width=2.5),
    ))
    mean_mae = daily_df["mae"].mean()
    fig_mae.add_hline(y=mean_mae, line_dash="dash", line_color="gray",
                      annotation_text=f"avg {mean_mae:.1f} €/MWh")
    fig_mae.update_layout(
        height=280, margin=dict(t=20, b=30),
        yaxis_title="MAE (€/MWh)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        hovermode="x unified",
    )
    st.plotly_chart(fig_mae, use_container_width=True)

    # Monthly breakdown
    with st.expander("Monthly performance breakdown"):
        if preds_df is not None:
            monthly = (
                preds_df.copy()
                .assign(month=preds_df["datetime_hour"].dt.to_period("M"))
                .groupby("month")
                .agg(
                    MAE=("abs_error", "mean"),
                    RMSE=("error", lambda x: np.sqrt((x**2).mean())),
                    Bias=("error", "mean"),
                    Direction_acc=("error", lambda x: (
                        np.mean(np.sign(x.diff().dropna()) == np.sign(
                            preds_df.loc[x.index, "actual"].diff().dropna()
                        )) * 100
                    )),
                )
                .round(2)
                .reset_index()
            )
            monthly["month"] = monthly["month"].astype(str)

            fig_m = px.bar(
                monthly, x="month", y="MAE",
                color_discrete_sequence=[color],
                labels={"month": "", "MAE": "MAE (€/MWh)"},
            )
            fig_m.add_hline(y=monthly["MAE"].mean(), line_dash="dash", line_color="gray")
            fig_m.update_layout(height=260, margin=dict(t=20, b=30))
            st.plotly_chart(fig_m, use_container_width=True)

# ── Section 6 — Hourly error profile ─────────────────────────────────────────
st.subheader("Error profile by hour of day")

if preds_df is not None:
    preds_df["hour"] = preds_df["datetime_hour"].dt.hour
    hourly = (
        preds_df.groupby("hour")
        .agg(MAE=("abs_error", "mean"), Bias=("error", "mean"), Std=("error", "std"))
        .reset_index()
    )

    fig_h = make_subplots(rows=1, cols=2, subplot_titles=["MAE by hour", "Bias by hour"])
    fig_h.add_trace(
        go.Bar(x=hourly["hour"], y=hourly["MAE"], marker_color=color, name="MAE"),
        row=1, col=1,
    )
    fig_h.add_trace(
        go.Bar(x=hourly["hour"], y=hourly["Bias"],
               marker_color=[color if v >= 0 else "#888" for v in hourly["Bias"]],
               name="Bias"),
        row=1, col=2,
    )
    fig_h.add_hline(y=0, line_dash="dash", line_color="gray", row=1, col=2)
    fig_h.update_layout(height=280, margin=dict(t=40, b=30), showlegend=False,
                        xaxis_title="Hour", xaxis2_title="Hour")
    st.plotly_chart(fig_h, use_container_width=True)

# ── Section 7 — Feature importance ───────────────────────────────────────────
st.subheader("Feature importance")

fi_path = REPORTS / f"{selected}_feature_importance.csv"
if fi_path.exists():
    fi = pd.read_csv(fi_path)
    # Top N slider
    top_n = st.slider("Show top N features", min_value=10, max_value=min(50, len(fi)), value=20)
    fi_top = fi.head(top_n).sort_values("importance")

    category_colors = {
        "Historical Lags": "#EF553B",
        "Rolling Stats":   "#FF7F0E",
        "Calendar":        "#636EFA",
        "Weather Forecast":"#00CC96",
        "Load Forecast":   "#AB63FA",
        "Renewable":       "#19D3F3",
        "Other":           "#888",
    }
    fi_top["color"] = fi_top.get("category", pd.Series(["Other"] * len(fi_top))).map(
        lambda c: category_colors.get(c, "#888")
    )

    fig_fi = go.Figure(go.Bar(
        x=fi_top["importance"],
        y=fi_top["feature"],
        orientation="h",
        marker_color=fi_top["color"],
        text=[f"{v:.3f}" for v in fi_top["importance"]],
        textposition="outside",
    ))
    fig_fi.update_layout(
        height=max(350, top_n * 22),
        margin=dict(t=20, b=30, r=80),
        xaxis_title="Importance",
        yaxis=dict(tickfont=dict(size=11)),
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    # Category breakdown
    if "category" in fi.columns:
        cat_sum = fi.groupby("category")["importance"].sum().reset_index().sort_values("importance", ascending=False)
        fig_cat = px.pie(
            cat_sum, names="category", values="importance",
            color="category",
            color_discrete_map=category_colors,
            hole=0.4,
        )
        fig_cat.update_layout(height=300, margin=dict(t=20, b=20))
        st.plotly_chart(fig_cat, use_container_width=True)
elif selected == "ridge":
    # Ridge has no tree importance — derive from absolute coefficients
    try:
        import joblib
        features_path = ROOT / "data/features.txt"
        model_path = MODELS / "ridge_final.pkl"
        if model_path.exists() and features_path.exists():
            feature_cols = [l.strip() for l in features_path.read_text().splitlines() if l.strip()]
            pipe = joblib.load(model_path)
            coefs = pipe.named_steps["ridge"].coef_
            fi = pd.DataFrame({"feature": feature_cols, "importance": np.abs(coefs)})
            fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)
            top_n = st.slider("Show top N features", min_value=10, max_value=min(50, len(fi)), value=20)
            fi_top = fi.head(top_n).sort_values("importance")
            fig_fi = go.Figure(go.Bar(
                x=fi_top["importance"],
                y=fi_top["feature"],
                orientation="h",
                marker_color=color,
                text=[f"{v:.2f}" for v in fi_top["importance"]],
                textposition="outside",
            ))
            fig_fi.update_layout(
                height=max(350, top_n * 22),
                margin=dict(t=20, b=30, r=80),
                xaxis_title="|Coefficient|",
                yaxis=dict(tickfont=dict(size=11)),
            )
            st.plotly_chart(fig_fi, use_container_width=True)
            st.caption("Ridge: feature importance = absolute value of standardized coefficients")
        else:
            st.info("Ridge model not found. Run `python scripts/train.py --model ridge`.")
    except Exception as e:
        st.warning(f"Could not load Ridge coefficients: {e}")
else:
    st.info(f"No feature importance file found for {selected}. Run `python scripts/train.py --model {selected}`.")
