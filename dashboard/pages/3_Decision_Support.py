"""
Decision Support page — J+1 price forecast with confidence intervals,
price alerts, recent accuracy tracker, and model agreement indicator.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Allow importing from the dashboard package
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    MODEL_COLOR,
    MODEL_DISPLAY,
    load_daily_metrics,
    load_dataset,
    load_live_predictions,
    load_predictions,
    load_quantile_predictions,
)

st.set_page_config(layout="wide")
st.title("Decision Support")
st.caption(
    "J+1 price forecast with uncertainty bounds, contextual alerts, "
    "recent accuracy tracking, and ensemble confidence indicator."
)

# ── Load data ─────────────────────────────────────────────────────────────────
live_df    = load_live_predictions()
quantile_df = load_quantile_predictions()
dataset_df  = load_dataset()

if live_df is None or live_df.empty:
    st.warning(
        "No inference predictions found. "
        "Run `python scripts/infer.py` to generate J+1 forecasts."
    )
    st.stop()

# Latest inference batch
latest_run   = live_df["prediction_made_at"].max()
latest_batch = live_df[live_df["prediction_made_at"] == latest_run].copy()
latest_batch = latest_batch.sort_values("target_datetime")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 1 — J+1 Forecast with confidence intervals
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.subheader("J+1 Price Forecast")

target_dates = sorted(latest_batch["target_date"].unique())
selected_date = st.selectbox(
    "Forecast date",
    options=target_dates,
    index=len(target_dates) - 1,
    format_func=lambda d: str(d),
)

day_df = latest_batch[latest_batch["target_date"] == selected_date].copy()

if day_df.empty:
    st.info("No forecast data for the selected date.")
else:
    ensemble_col = "pred_ensemble" if "pred_ensemble" in day_df.columns else day_df.columns[-1]
    hours    = day_df["target_datetime"]
    ensemble = day_df[ensemble_col].values

    # ── Build confidence bands ────────────────────────────────────────────────
    # Priority 1: live inference quantile columns (pred_q10/q90) in predictions.csv
    # Priority 2: walk-forward quantile CSV (same date in test period)
    # Priority 3: estimate from historical mean interval width
    lower_band = upper_band = None
    band_source = ""

    if "pred_q10" in day_df.columns and "pred_q90" in day_df.columns:
        lower_band  = day_df["pred_q10"].values
        upper_band  = day_df["pred_q90"].values
        band_source = "live quantile model (p10–p90)"
    elif quantile_df is not None and not quantile_df.empty:
        target_dt = pd.to_datetime(selected_date)
        day_q = quantile_df[quantile_df["datetime_hour"].dt.date == target_dt.date()]
        if not day_q.empty and "pred_p10" in day_q.columns:
            day_q      = day_q.sort_values("datetime_hour")
            lower_band = day_q["pred_p10"].values
            upper_band = day_q["pred_p90"].values
            band_source = "walk-forward quantile model (p10–p90)"
        else:
            # Estimate bandwidth from historical mean interval width
            if "interval_width" in quantile_df.columns:
                mean_half  = quantile_df["interval_width"].mean() / 2
                lower_band = ensemble - mean_half
                upper_band = ensemble + mean_half
                band_source = "estimated from historical uncertainty"

    # ── Summary metrics ───────────────────────────────────────────────────────
    mean_price = float(np.mean(ensemble))
    min_price  = float(np.min(ensemble))
    max_price  = float(np.max(ensemble))
    spread     = max_price - min_price

    # Historical coverage probability
    coverage_pct = None
    if quantile_df is not None and "in_interval" in quantile_df.columns:
        recent_q = quantile_df.tail(30 * 24)
        coverage_pct = recent_q["in_interval"].mean() * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Predicted date", str(selected_date))
    col2.metric("Mean price (EUR/MWh)", f"{mean_price:.1f}")
    col3.metric("Peak–offpeak spread", f"{spread:.1f} EUR/MWh")
    if coverage_pct is not None:
        col4.metric("Interval coverage (last 30 d)", f"{coverage_pct:.0f}%",
                    delta=f"{coverage_pct - 80:.0f}pp vs 80% target",
                    delta_color="normal")
    else:
        col4.metric("Min / Max", f"{min_price:.0f} / {max_price:.0f} EUR/MWh")

    # ── Chart ─────────────────────────────────────────────────────────────────
    fig = go.Figure()

    if lower_band is not None and upper_band is not None:
        fig.add_trace(go.Scatter(
            x=list(hours) + list(hours)[::-1],
            y=list(upper_band) + list(lower_band)[::-1],
            fill="toself",
            fillcolor="rgba(99,110,250,0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name=f"Confidence band ({band_source})",
            hoverinfo="skip",
        ))

    fig.add_trace(go.Scatter(
        x=hours, y=ensemble,
        mode="lines+markers",
        name="Ensemble forecast",
        line=dict(color="#636EFA", width=2.5),
        marker=dict(size=5),
    ))

    # Add individual model traces if available
    for col_name, model_key in [
        ("pred_xgboost", "xgboost"), ("pred_ridge", "ridge")
    ]:
        if col_name in day_df.columns:
            fig.add_trace(go.Scatter(
                x=hours, y=day_df[col_name].values,
                mode="lines",
                name=MODEL_DISPLAY.get(model_key, model_key),
                line=dict(color=MODEL_COLOR.get(model_key, "#888"), width=1, dash="dot"),
                opacity=0.6,
            ))

    fig.update_layout(
        xaxis_title="Hour",
        yaxis_title="Price (EUR/MWh)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=380,
        margin=dict(t=30),
    )
    st.plotly_chart(fig, use_container_width=True)

    if band_source:
        st.caption(f"Confidence band: {band_source}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 2 — Price alerts
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.subheader("Price Alerts")

if dataset_df is None:
    st.info("Dataset not available — cannot compute historical price thresholds.")
else:
    price_col = "price" if "price" in dataset_df.columns else None

    if price_col is None:
        st.info("No price column found in dataset.")
    else:
        historical_prices = dataset_df[price_col].dropna().values

        col_sens, _ = st.columns([1, 3])
        with col_sens:
            p_low  = st.slider("Low-price percentile threshold", 5, 30, 10, step=5,
                               help="Hours below this percentile are flagged as cheap.")
            p_high = st.slider("High-price percentile threshold", 70, 95, 90, step=5,
                               help="Hours above this percentile are flagged as expensive.")

        thr_low_warn  = float(np.percentile(historical_prices, p_low))
        thr_low_ext   = float(np.percentile(historical_prices, max(p_low - 5, 1)))
        thr_high_warn = float(np.percentile(historical_prices, p_high))
        thr_high_ext  = float(np.percentile(historical_prices, min(p_high + 5, 99)))

        if not day_df.empty:
            alerts = []
            for _, row in day_df.iterrows():
                price = row[ensemble_col]
                hour  = pd.to_datetime(row["target_datetime"]).hour
                if price <= thr_low_ext:
                    level = "Extreme low"
                    color = "🟢"
                    tip   = "Very cheap — ideal for high-consumption tasks"
                elif price <= thr_low_warn:
                    level = "Low"
                    color = "🟡"
                    tip   = "Below-average price — good for flexible loads"
                elif price >= thr_high_ext:
                    level = "Extreme high"
                    color = "🔴"
                    tip   = "Very expensive — avoid non-essential consumption"
                elif price >= thr_high_warn:
                    level = "High"
                    color = "🟠"
                    tip   = "Above-average price — reduce flexible loads"
                else:
                    level = "Normal"
                    color = "⚪"
                    tip   = ""
                alerts.append({
                    "Hour": f"{hour:02d}:00",
                    "Forecast (EUR/MWh)": round(price, 1),
                    "Alert": f"{color} {level}",
                    "Recommendation": tip,
                })

            alerts_df = pd.DataFrame(alerts)

            # Summary callouts
            cheap_hours = [a for a in alerts if "low" in a["Alert"].lower()]
            expensive_hours = [a for a in alerts if "high" in a["Alert"].lower()]

            if cheap_hours:
                cheap_str = ", ".join(a["Hour"] for a in cheap_hours)
                st.success(f"**Cheap hours:** {cheap_str} — consider scheduling flexible loads")
            if expensive_hours:
                exp_str = ", ".join(a["Hour"] for a in expensive_hours)
                st.warning(f"**Expensive hours:** {exp_str} — reduce non-essential consumption")

            # Alert table — filter out normal if nothing to show
            show_all = st.checkbox("Show all hours (including normal)", value=False)
            display_df = alerts_df if show_all else alerts_df[alerts_df["Alert"] != "⚪ Normal"]
            if display_df.empty:
                st.info("No price alerts for this day — prices are within the normal range.")
            else:
                st.dataframe(display_df, use_container_width=True, hide_index=True)

            st.caption(
                f"Thresholds: low < {thr_low_warn:.0f} EUR/MWh (p{p_low}), "
                f"high > {thr_high_warn:.0f} EUR/MWh (p{p_high}). "
                f"Computed from {len(historical_prices):,} historical hours."
            )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 3 — Recent accuracy tracker
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.subheader("Recent Forecast Accuracy")

daily_dfs = {
    name: load_daily_metrics(name)
    for name in ("xgboost", "ridge", "lightgbm")
}
daily_dfs = {k: v for k, v in daily_dfs.items() if v is not None}

if not daily_dfs:
    st.info("No daily metrics found. Run `python scripts/train.py` first.")
else:
    # Pick the model with most data as the reference
    ref_name = max(daily_dfs, key=lambda k: len(daily_dfs[k]))
    ref_df   = daily_dfs[ref_name].sort_values("datetime")

    window_days = st.select_slider(
        "Rolling window", options=[7, 14, 30, 60], value=30
    )

    col1, col2, col3 = st.columns(3)

    overall_mae  = ref_df["mae"].mean()
    last30_mae   = ref_df.tail(30)["mae"].mean()
    last7_mae    = ref_df.tail(7)["mae"].mean()
    delta30      = last30_mae - overall_mae
    delta7       = last30_mae - last7_mae

    col1.metric("All-time MAE", f"{overall_mae:.1f} EUR/MWh")
    col2.metric("Last 30 d MAE", f"{last30_mae:.1f} EUR/MWh",
                delta=f"{delta30:+.1f} vs all-time",
                delta_color="inverse")
    col3.metric("Last 7 d MAE", f"{last7_mae:.1f} EUR/MWh",
                delta=f"{delta7:+.1f} vs last 30 d",
                delta_color="inverse")

    # Rolling MAE chart for all available models
    fig = go.Figure()
    for name, dm in daily_dfs.items():
        dm_sorted = dm.sort_values("datetime")
        rolling_mae = dm_sorted["mae"].rolling(window_days, min_periods=3).mean()
        fig.add_trace(go.Scatter(
            x=dm_sorted["datetime"],
            y=rolling_mae,
            mode="lines",
            name=MODEL_DISPLAY.get(name, name),
            line=dict(color=MODEL_COLOR.get(name, "#888"), width=2),
        ))

    fig.add_hline(
        y=overall_mae,
        line_dash="dot",
        line_color="gray",
        annotation_text=f"All-time mean ({overall_mae:.1f})",
        annotation_position="bottom right",
    )
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title=f"{window_days}-day rolling MAE (EUR/MWh)",
        height=300,
        margin=dict(t=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Quantile calibration
    if quantile_df is not None and "in_interval" in quantile_df.columns:
        alltime_cov = quantile_df["in_interval"].mean() * 100
        delta_cov   = alltime_cov - 80.0
        q_start = quantile_df["datetime_hour"].min().strftime("%Y-%m-%d")
        q_end   = quantile_df["datetime_hour"].max().strftime("%Y-%m-%d")
        st.metric(
            f"Quantile interval coverage (p10–p90, {q_start} → {q_end})",
            f"{alltime_cov:.0f}%",
            delta=f"{delta_cov:+.0f}pp vs 80% target",
            delta_color="normal",
            help=(
                "Percentage of actual prices that fell inside the p10–p90 walk-forward interval. "
                "A well-calibrated model should cover ~80% of actuals."
            ),
        )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SECTION 4 — Model agreement indicator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
st.subheader("Model Agreement")

# Merge walk-forward predictions across models
pred_frames: dict[str, pd.DataFrame] = {}
for name in ("xgboost", "ridge"):
    pf = load_predictions(name)
    if pf is not None:
        pred_frames[name] = pf.rename(columns={"predicted": f"pred_{name}"})
if quantile_df is not None and "pred_p50" in quantile_df.columns:
    pred_frames["lightgbm"] = quantile_df[["datetime_hour", "pred_p50", "actual"]].rename(
        columns={"pred_p50": "pred_lightgbm"}
    )

if len(pred_frames) < 2:
    st.info("Need at least two models to compute agreement. Run `python scripts/train.py`.")
else:
    # Join on datetime_hour
    merged = None
    for name, df in pred_frames.items():
        keep_cols = ["datetime_hour"] + [c for c in df.columns if c.startswith("pred_")]
        if merged is None:
            merged = df[keep_cols + (["actual"] if "actual" in df.columns else [])].copy()
        else:
            merged = merged.merge(
                df[keep_cols], on="datetime_hour", how="inner"
            )

    pred_cols = [c for c in merged.columns if c.startswith("pred_")]

    if len(pred_cols) >= 2:
        merged["model_spread"] = merged[pred_cols].max(axis=1) - merged[pred_cols].min(axis=1)
        merged["mean_pred"]    = merged[pred_cols].mean(axis=1)
        merged["agreement_pct"] = (
            1 - merged["model_spread"] / merged["mean_pred"].replace(0, np.nan)
        ).clip(0, 1) * 100

        # ── Current forecast confidence ────────────────────────────────────────
        # Use historical spread distribution (from walk-forward period) as baseline.
        # A day's spread is "high" if it exceeds the 75th percentile of historical spreads.
        if "pred_xgboost" in day_df.columns and "pred_ridge" in day_df.columns:
            day_spread_mean = (
                day_df[["pred_xgboost", "pred_ridge"]].max(axis=1)
                - day_df[["pred_xgboost", "pred_ridge"]].min(axis=1)
            ).mean()

            # Historical spread baseline from walk-forward predictions
            hist_spread = merged["model_spread"].dropna()
            if len(hist_spread) > 10:
                p25 = float(np.percentile(hist_spread, 25))
                p75 = float(np.percentile(hist_spread, 75))
                if day_spread_mean <= p25:
                    conf_label, conf_color, conf_pct = "High confidence", "green", 90
                elif day_spread_mean <= p75:
                    conf_label, conf_color, conf_pct = "Moderate confidence", "orange", 65
                else:
                    conf_label, conf_color, conf_pct = "Low confidence", "red", 30
            else:
                conf_pct = 50
                conf_label, conf_color = "Insufficient history", "gray"

            col1, col2 = st.columns(2)
            col1.metric(
                f"Forecast confidence ({selected_date})",
                conf_label,
                help=(
                    f"Mean model spread today: {day_spread_mean:.1f} EUR/MWh. "
                    f"Historical p25={p25:.1f}, p75={p75:.1f} EUR/MWh. "
                    "Spread below p25 → high confidence; above p75 → low."
                ),
            )
            col2.metric("Inter-model spread", f"{day_spread_mean:.1f} EUR/MWh",
                        help="Average absolute difference between XGBoost and Ridge forecasts.")

        # ── Scatter: spread vs error ───────────────────────────────────────────
        if "actual" in merged.columns:
            merged["abs_error"] = (merged["actual"] - merged["mean_pred"]).abs()

            with st.expander("Spread vs. actual error — validation that disagreement predicts uncertainty"):
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=merged["model_spread"],
                    y=merged["abs_error"],
                    mode="markers",
                    marker=dict(color="#636EFA", opacity=0.3, size=4),
                    name="Hourly observations",
                ))

                # Binned averages
                spread_clean = merged["model_spread"].dropna()
                if len(spread_clean) > 20:
                    n_bins  = min(20, len(spread_clean) // 10)
                    bins    = pd.cut(merged["model_spread"], bins=n_bins)
                    binned  = merged.groupby(bins, observed=True)["abs_error"].mean().reset_index()
                    binned["spread_mid"] = binned["model_spread"].apply(lambda iv: iv.mid)
                    fig2.add_trace(go.Scatter(
                        x=binned["spread_mid"],
                        y=binned["abs_error"],
                        mode="lines+markers",
                        line=dict(color="#EF553B", width=2),
                        name="Binned average",
                    ))

                # Correlation annotation
                corr = merged[["model_spread", "abs_error"]].corr().iloc[0, 1]
                fig2.add_annotation(
                    x=0.98, y=0.04, xref="paper", yref="paper",
                    text=f"Pearson r = {corr:.2f}",
                    showarrow=False, font=dict(size=12),
                )
                fig2.update_layout(
                    xaxis_title="Model spread (EUR/MWh)",
                    yaxis_title="Absolute error (EUR/MWh)",
                    height=320,
                    margin=dict(t=20),
                )
                st.plotly_chart(fig2, use_container_width=True)
                st.caption(
                    "When models disagree (large spread), actual forecast errors tend to be larger. "
                    "This validates using spread as a real-time confidence proxy."
                )

        # ── Rolling agreement over time ────────────────────────────────────────
        with st.expander("Rolling model agreement over time"):
            merged_sorted = merged.sort_values("datetime_hour")
            roll_agree    = merged_sorted["agreement_pct"].rolling(24 * 7, min_periods=24).mean()
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=merged_sorted["datetime_hour"],
                y=roll_agree,
                mode="lines",
                line=dict(color="#636EFA", width=2),
                name="7-day rolling agreement",
            ))
            fig3.add_hline(y=70, line_dash="dot", line_color="orange",
                           annotation_text="70% threshold")
            fig3.update_layout(
                xaxis_title="Date",
                yaxis_title="Agreement (%)",
                yaxis=dict(range=[0, 105]),
                height=280,
                margin=dict(t=20),
            )
            st.plotly_chart(fig3, use_container_width=True)
