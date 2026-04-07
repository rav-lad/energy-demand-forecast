"""
Data page — dataset status, feature exploration, data quality.
"""

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "dataset.csv"
STATUS_PATH = PROJECT_ROOT / "data" / ".update_status.json"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURE_GROUPS = {
    "Spot price": [
        "price", "price_lag_24h", "price_lag_48h", "price_lag_72h", "price_lag_168h",
        "price_roll_mean_24h", "price_roll_std_24h", "price_roll_mean_168h", "price_roll_std_168h",
    ],
    "Load": [
        "load_mw", "load_lag_24h", "load_lag_48h", "load_lag_72h", "load_lag_168h",
        "load_roll_mean_24h", "load_roll_std_24h", "load_roll_mean_168h", "load_roll_std_168h",
        "net_load_mw",
    ],
    "Weather": ["temperature_2m", "wind_speed_10m", "precipitation", "shortwave_radiation"],
    "Renewables": [
        "wind_production_mw", "solar_production_mw", "renewable_production_mw",
        "renewable_penetration_pct",
    ],
    "Calendar": ["hour", "day_of_week", "month", "is_weekend", "is_peak_hour", "is_night",
                 "is_winter", "is_summer"],
}

FEATURE_LABELS = {
    "price": "FR Spot Price (EUR/MWh)",
    "load_mw": "Electricity Load (MW)",
    "temperature_2m": "Temperature (°C)",
    "wind_speed_10m": "Wind Speed 10m (m/s)",
    "precipitation": "Precipitation (mm)",
    "shortwave_radiation": "Solar Radiation (W/m²)",
    "wind_production_mw": "Wind Production (MW)",
    "solar_production_mw": "Solar Production (MW)",
    "renewable_production_mw": "Total Renewable Production (MW)",
    "renewable_penetration_pct": "Renewable Penetration (%)",
    "net_load_mw": "Net Load (MW)",
    "price_lag_24h": "Spot Price Lag 24h (EUR/MWh)",
    "price_lag_168h": "Spot Price Lag 168h (EUR/MWh)",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)
def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)
    df["datetime"] = pd.to_datetime(df["datetime"])
    return df


def load_status() -> dict:
    if STATUS_PATH.exists():
        return json.loads(STATUS_PATH.read_text())
    return {}


def days_since(date_str: str) -> int:
    try:
        return (date.today() - date.fromisoformat(date_str)).days
    except Exception:
        return -1


def freshness_label(days: int) -> str:
    if days < 0:
        return "unknown"
    if days == 0:
        return "today"
    if days == 1:
        return "yesterday"
    return f"D-{days}"


def freshness_color(days: int) -> str:
    if days <= 1:
        return "green"
    if days <= 7:
        return "orange"
    return "red"


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
if not DATASET_PATH.exists():
    st.error("dataset.csv not found. Run `python scripts/build_dataset.py`.")
    st.stop()

df = load_dataset()
status = load_status()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Filters")
    min_date = df["datetime"].min().date()
    max_date = df["datetime"].max().date()

    date_range = st.date_input(
        "Date range",
        value=(max_date - timedelta(days=90), max_date),
        min_value=min_date,
        max_value=max_date,
    )
    start_filter, end_filter = (date_range if len(date_range) == 2
                                else (min_date, max_date))

    resample = st.selectbox("Resolution", ["Hourly", "Daily", "Weekly"], index=1)
    resample_freq = {"Hourly": None, "Daily": "D", "Weekly": "W"}[resample]

    st.divider()
    if st.button("Refresh data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

df_view = df[
    (df["datetime"].dt.date >= start_filter) &
    (df["datetime"].dt.date <= end_filter)
].copy()

# ---------------------------------------------------------------------------
# Section 1 — Data sources status
# ---------------------------------------------------------------------------
st.title("Data")
st.header("1. Data sources")

dataset_end = df["datetime"].max().date()
dataset_start = df["datetime"].min().date()
dataset_rows = len(df)
days_old = days_since(str(dataset_end))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Dataset start", str(dataset_start))
col2.metric("Dataset end", str(dataset_end))
col3.metric("Total hours", f"{dataset_rows:,}")
col4.metric("Freshness", freshness_label(days_old),
            help="Days since the last available data point")

st.caption(
    f"Charts show the selected period ({start_filter} to {end_filter}). "
    "Adjust the date range in the sidebar to see the full history from 2023."
)

st.subheader("Raw sources")
sources = [
    ("ENTSO-E FR",  "data/raw/fr_spot_hourly.csv",               "datetime",   None),
    ("ENTSO-E DE",  "data/raw/de_spot_hourly.csv",               "datetime",   None),
    ("Open-Meteo",  None,                                         "datetime",   None),
    ("ODRE Consumption", "data/raw/energy_consumption_2023-2026.csv", "date_heure", ";"),
]

cols = st.columns(len(sources))
for col, (name, rel_path, dcol, sep) in zip(cols, sources):
    with col:
        if rel_path is None:
            weather_files = sorted(
                (PROJECT_ROOT / "data" / "external").glob("weather_hourly_*.csv")
            )
            if weather_files:
                try:
                    tmp = pd.read_csv(weather_files[-1], usecols=[dcol])
                    tmp[dcol] = pd.to_datetime(tmp[dcol], errors="coerce")
                    last = str(tmp[dcol].max().date())
                    rows = len(tmp)
                except Exception:
                    last, rows = "error", 0
            else:
                last, rows = "missing", 0
        else:
            fpath = PROJECT_ROOT / rel_path
            if fpath.exists():
                try:
                    tmp = (pd.read_csv(fpath, sep=sep, usecols=[dcol])
                           if sep else pd.read_csv(fpath, usecols=[dcol]))
                    tmp[dcol] = pd.to_datetime(tmp[dcol], utc=True, errors="coerce")
                    last = str(tmp[dcol].max().date())
                    rows = len(tmp)
                except Exception:
                    last, rows = "error", 0
            else:
                last, rows = "missing", 0

        d = days_since(last) if last not in ("error", "missing") else 999
        color = freshness_color(d)
        st.markdown(f"**{name}**")
        st.markdown(f"Last date: `{last}`")
        st.markdown(f"Rows: `{rows:,}`")
        if d < 999:
            st.markdown(
                f"<span style='color:{color}; font-weight:600'>"
                f"{freshness_label(d)}</span>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(f"<span style='color:red'>unavailable</span>",
                        unsafe_allow_html=True)

if status:
    last_run = status.get("last_run", "never")
    ok_str = "OK" if status.get("build_ok", False) else "FAILED"
    st.caption(f"Last scheduled update: {last_run} — {ok_str}")

st.divider()

# ---------------------------------------------------------------------------
# Section 2 — Dataset preview
# ---------------------------------------------------------------------------
st.header("2. Dataset preview")

tab1, tab2 = st.tabs(["Table", "Statistics"])

with tab1:
    n_show = st.slider("Rows to display", 10, 200, 50)
    st.dataframe(df_view.tail(n_show), use_container_width=True, height=400)

with tab2:
    numeric_cols = df_view.select_dtypes(include=np.number).columns.tolist()
    stats = df_view[numeric_cols].describe().T
    stats["missing"] = df_view[numeric_cols].isna().sum()
    stats["missing_%"] = (stats["missing"] / len(df_view) * 100).round(2)
    st.dataframe(
        stats.style.background_gradient(subset=["missing_%"], cmap="Reds"),
        use_container_width=True,
    )

st.divider()

# ---------------------------------------------------------------------------
# Section 3 — Feature exploration
# ---------------------------------------------------------------------------
st.header("3. Feature exploration")

group = st.selectbox("Feature group", list(FEATURE_GROUPS.keys()))
available = [c for c in FEATURE_GROUPS[group] if c in df_view.columns]
feature = st.selectbox(
    "Feature", available,
    format_func=lambda c: FEATURE_LABELS.get(c, c),
)

df_plot = df_view[["datetime", feature]].copy()
if resample_freq:
    df_plot = df_plot.set_index("datetime").resample(resample_freq).mean().reset_index()

label = FEATURE_LABELS.get(feature, feature)

col_ts, col_dist = st.columns([3, 1])

with col_ts:
    fig = px.line(
        df_plot, x="datetime", y=feature,
        title=f"{label}  —  {start_filter} to {end_filter}",
        labels={"datetime": "", feature: label},
        template="plotly_white",
    )
    fig.update_traces(line=dict(width=1.2))
    fig.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)

with col_dist:
    fig2 = px.histogram(
        df_plot, x=feature, nbins=50,
        title="Distribution",
        labels={feature: label},
        template="plotly_white",
        color_discrete_sequence=["#636EFA"],
    )
    fig2.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0), showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)

s = df_plot[feature].dropna()
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Mean", f"{s.mean():.2f}")
c2.metric("Median", f"{s.median():.2f}")
c3.metric("Std", f"{s.std():.2f}")
c4.metric("Min", f"{s.min():.2f}")
c5.metric("Max", f"{s.max():.2f}")

st.divider()

# ---------------------------------------------------------------------------
# Section 4 — Correlations
# ---------------------------------------------------------------------------
st.header("4. Correlations with spot price")

core_features = [
    "load_mw", "temperature_2m", "wind_speed_10m", "shortwave_radiation",
    "renewable_production_mw", "net_load_mw", "renewable_penetration_pct",
    "price_lag_24h", "price_lag_168h",
]
available_core = [c for c in core_features if c in df_view.columns]
corr = df_view[["price"] + available_core].corr()["price"].drop("price").sort_values()

fig_corr = go.Figure(go.Bar(
    x=corr.values,
    y=[FEATURE_LABELS.get(c, c) for c in corr.index],
    orientation="h",
    marker_color=["#EF553B" if v < 0 else "#00CC96" for v in corr.values],
))
fig_corr.update_layout(
    title="Pearson correlation with spot price",
    template="plotly_white",
    height=400,
    margin=dict(l=0, r=0, t=40, b=0),
    xaxis_title="Correlation",
    yaxis_title="",
)
st.plotly_chart(fig_corr, use_container_width=True)

st.divider()

# ---------------------------------------------------------------------------
# Section 5 — Data coverage
# ---------------------------------------------------------------------------
st.header("5. Data coverage")

key_cols = [c for c in ["price", "load_mw", "temperature_2m", "wind_production_mw"]
            if c in df.columns]
df_cov = df[["datetime"] + key_cols].copy()
df_cov["date"] = df_cov["datetime"].dt.date
daily_null = (
    df_cov.drop(columns=["datetime"])
    .groupby("date")
    .apply(lambda x: x.isna().any(axis=1).sum())
)
total_missing = int(daily_null.sum())

if total_missing == 0:
    st.success(
        f"No missing values across {len(df):,} hours for key columns: "
        f"{', '.join(key_cols)}"
    )
    df_cov["month"] = pd.to_datetime(df_cov["date"]).dt.to_period("M").astype(str)
    monthly_count = df_cov.groupby("month").size().reset_index(name="hours")
    fig_cov = px.bar(
        monthly_count, x="month", y="hours",
        title="Available hours per month",
        labels={"month": "Month", "hours": "Hours"},
        template="plotly_white",
        color_discrete_sequence=["#636EFA"],
    )
    fig_cov.add_hline(
        y=24 * 28, line_dash="dot", line_color="gray",
        annotation_text="28 full days (672h)",
    )
    fig_cov.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_cov, use_container_width=True)
else:
    st.warning(f"{total_missing} hours with missing values")
    fig_cov = px.bar(
        x=daily_null[daily_null > 0].index,
        y=daily_null[daily_null > 0].values,
        title="Days with missing values",
        labels={"x": "Date", "y": "Hours with NaN"},
        template="plotly_white",
        color_discrete_sequence=["#EF553B"],
    )
    fig_cov.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_cov, use_container_width=True)

st.caption(
    f"Dataset: `{DATASET_PATH}` — {dataset_rows:,} hours — last updated: {dataset_end}"
)
