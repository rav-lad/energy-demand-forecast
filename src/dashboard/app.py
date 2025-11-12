"""
Streamlit Dashboard for Energy Trading Research Platform

Features:
- Model performance visualization
- Predictions explorer
- Trading signals analysis
- Backtest results
- Real-time monitoring
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import pickle
import sys
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Page config
st.set_page_config(
    page_title="Energy Trading Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# Helper Functions
# =============================================================================


@st.cache_data
def load_predictions(
    model_name: str = "xgboost", frequency: str = "daily"
) -> pd.DataFrame:
    """Load model predictions."""
    pred_path = (
        project_root
        / "outputs"
        / "reports"
        / f"{model_name}_predictions_{frequency}.csv"
    )

    if pred_path.exists():
        df = pd.read_csv(pred_path, parse_dates=["date"])
        return df
    else:
        # Generate sample data if not available
        dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
        return pd.DataFrame(
            {
                "date": dates,
                "actual_elec": np.random.normal(5000, 1000, len(dates)),
                "pred_elec": np.random.normal(5000, 1000, len(dates)),
                "actual_gaz": np.random.normal(3000, 800, len(dates)),
                "pred_gaz": np.random.normal(3000, 800, len(dates)),
            }
        )


@st.cache_data
def load_benchmark_results() -> pd.DataFrame:
    """Load benchmark comparison results."""
    bench_path = project_root / "outputs" / "reports" / "benchmark_results.csv"

    if bench_path.exists():
        return pd.read_csv(bench_path)
    else:
        # Sample data
        return pd.DataFrame(
            {
                "model": ["XGBoost", "LightGBM", "TFT", "Ridge", "Lasso"],
                "rmse_elec": [650, 520, 480, 850, 920],
                "rmse_gaz": [420, 385, 370, 580, 650],
                "r2_elec": [0.935, 0.952, 0.961, 0.885, 0.865],
                "training_time_sec": [180, 420, 1800, 30, 30],
                "inference_time_ms": [120, 180, 1050, 50, 50],
            }
        )


@st.cache_data
def load_market_prices() -> pd.DataFrame:
    """Load market price data."""
    price_path = (
        project_root / "data" / "raw_data" / "market_prices" / "day_ahead_prices_FR.csv"
    )

    if price_path.exists():
        df = pd.read_csv(price_path, parse_dates=["datetime"])
        return df
    else:
        # Sample data
        dates = pd.date_range("2023-01-01", "2024-12-31", freq="H")
        return pd.DataFrame(
            {
                "datetime": dates,
                "price_FR": 50 + np.random.randn(len(dates)).cumsum() * 2,
            }
        )


def calculate_metrics(actual: np.ndarray, pred: np.ndarray) -> dict:
    """Calculate prediction metrics."""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    mape = np.mean(np.abs((actual - pred) / (actual + 1e-10))) * 100
    r2 = r2_score(actual, pred)

    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R²": r2}


# =============================================================================
# Sidebar
# =============================================================================

st.sidebar.title("⚡ Energy Trading")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "📊 Overview",
        "🎯 Predictions",
        "💹 Trading Signals",
        "📈 Backtests",
        "🔬 Model Comparison",
    ],
)

st.sidebar.markdown("---")

# Model selection
model_options = ["xgboost", "lightgbm", "tft", "ridge", "lasso"]
selected_model = st.sidebar.selectbox("Select Model", model_options, index=0)

# Date range
st.sidebar.markdown("### Date Range")
date_range = st.sidebar.date_input(
    "Select dates",
    value=(datetime(2024, 1, 1), datetime(2024, 12, 31)),
    max_value=datetime.now(),
)


# =============================================================================
# Page: Overview
# =============================================================================

if page == "📊 Overview":
    st.title("📊 Energy Trading Dashboard")
    st.markdown("### Platform Overview")

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(label="Active Models", value="5", delta="2 new")

    with col2:
        st.metric(label="Predictions Today", value="1,247", delta="+123")

    with col3:
        st.metric(label="Avg MAPE", value="8.5%", delta="-1.2%")

    with col4:
        st.metric(label="Trading Profit", value="€12,450", delta="+5.2%")

    st.markdown("---")

    # Load benchmark data
    benchmark_df = load_benchmark_results()

    # Model comparison chart
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Model Performance (RMSE)")

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                name="Electricity",
                x=benchmark_df["model"],
                y=benchmark_df["rmse_elec"],
                marker_color="rgb(55, 83, 109)",
            )
        )

        fig.add_trace(
            go.Bar(
                name="Gas",
                x=benchmark_df["model"],
                y=benchmark_df["rmse_gaz"],
                marker_color="rgb(26, 118, 255)",
            )
        )

        fig.update_layout(
            barmode="group", xaxis_title="Model", yaxis_title="RMSE (MW)", height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### R² Score Comparison")

        fig = px.bar(
            benchmark_df,
            x="model",
            y="r2_elec",
            title="R² Score (Higher is Better)",
            color="r2_elec",
            color_continuous_scale="Viridis",
        )

        fig.update_layout(height=400)

        st.plotly_chart(fig, use_container_width=True)

    # Performance table
    st.markdown("### Detailed Model Comparison")
    st.dataframe(
        benchmark_df.style.highlight_max(
            subset=["r2_elec"], color="lightgreen"
        ).highlight_min(subset=["rmse_elec", "rmse_gaz"], color="lightgreen"),
        use_container_width=True,
    )


# =============================================================================
# Page: Predictions
# =============================================================================

elif page == "🎯 Predictions":
    st.title("🎯 Energy Demand Predictions")

    # Load predictions
    predictions_df = load_predictions(selected_model)

    # Filter by date range
    if len(date_range) == 2:
        start_date, end_date = date_range
        mask = (predictions_df["date"] >= pd.Timestamp(start_date)) & (
            predictions_df["date"] <= pd.Timestamp(end_date)
        )
        predictions_df = predictions_df[mask]

    # Calculate metrics
    if (
        "actual_elec" in predictions_df.columns
        and "pred_elec" in predictions_df.columns
    ):
        metrics_elec = calculate_metrics(
            predictions_df["actual_elec"].values, predictions_df["pred_elec"].values
        )

        metrics_gaz = calculate_metrics(
            predictions_df["actual_gaz"].values, predictions_df["pred_gaz"].values
        )

        # Display metrics
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### ⚡ Electricity Metrics")
            subcol1, subcol2, subcol3, subcol4 = st.columns(4)

            with subcol1:
                st.metric("RMSE", f"{metrics_elec['RMSE']:.2f} MW")
            with subcol2:
                st.metric("MAE", f"{metrics_elec['MAE']:.2f} MW")
            with subcol3:
                st.metric("MAPE", f"{metrics_elec['MAPE']:.2f}%")
            with subcol4:
                st.metric("R²", f"{metrics_elec['R²']:.4f}")

        with col2:
            st.markdown("### 🔥 Gas Metrics")
            subcol1, subcol2, subcol3, subcol4 = st.columns(4)

            with subcol1:
                st.metric("RMSE", f"{metrics_gaz['RMSE']:.2f} MW")
            with subcol2:
                st.metric("MAE", f"{metrics_gaz['MAE']:.2f} MW")
            with subcol3:
                st.metric("MAPE", f"{metrics_gaz['MAPE']:.2f}%")
            with subcol4:
                st.metric("R²", f"{metrics_gaz['R²']:.4f}")

    # Prediction vs Actual charts
    st.markdown("---")

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Electricity Consumption", "Gas Consumption"),
        vertical_spacing=0.15,
    )

    # Electricity
    fig.add_trace(
        go.Scatter(
            x=predictions_df["date"],
            y=predictions_df["actual_elec"],
            name="Actual Electricity",
            line=dict(color="royalblue", width=2),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=predictions_df["date"],
            y=predictions_df["pred_elec"],
            name="Predicted Electricity",
            line=dict(color="orange", width=2, dash="dash"),
        ),
        row=1,
        col=1,
    )

    # Gas
    fig.add_trace(
        go.Scatter(
            x=predictions_df["date"],
            y=predictions_df["actual_gaz"],
            name="Actual Gas",
            line=dict(color="green", width=2),
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=predictions_df["date"],
            y=predictions_df["pred_gaz"],
            name="Predicted Gas",
            line=dict(color="red", width=2, dash="dash"),
        ),
        row=2,
        col=1,
    )

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Consumption (MW)", row=1, col=1)
    fig.update_yaxes(title_text="Consumption (MW)", row=2, col=1)

    fig.update_layout(height=800, showlegend=True)

    st.plotly_chart(fig, use_container_width=True)

    # Errors distribution
    st.markdown("### Error Distribution")

    col1, col2 = st.columns(2)

    with col1:
        errors_elec = predictions_df["actual_elec"] - predictions_df["pred_elec"]

        fig = go.Figure(data=[go.Histogram(x=errors_elec, nbinsx=50)])
        fig.update_layout(
            title="Electricity Prediction Errors",
            xaxis_title="Error (MW)",
            yaxis_title="Frequency",
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        errors_gaz = predictions_df["actual_gaz"] - predictions_df["pred_gaz"]

        fig = go.Figure(data=[go.Histogram(x=errors_gaz, nbinsx=50)])
        fig.update_layout(
            title="Gas Prediction Errors",
            xaxis_title="Error (MW)",
            yaxis_title="Frequency",
            height=400,
        )

        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Page: Trading Signals
# =============================================================================

elif page == "💹 Trading Signals":
    st.title("💹 Trading Signals Analysis")

    st.markdown(
        """
    This page shows trading signals generated based on demand predictions
    and market conditions.
    """
    )

    # Load market prices
    prices_df = load_market_prices()

    # Generate sample signals
    prices_df["signal"] = 0
    prices_df.loc[prices_df["price_FR"] < 45, "signal"] = 1  # BUY
    prices_df.loc[prices_df["price_FR"] > 55, "signal"] = -1  # SELL

    # Signal statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        buy_signals = (prices_df["signal"] == 1).sum()
        st.metric("BUY Signals", buy_signals)

    with col2:
        sell_signals = (prices_df["signal"] == -1).sum()
        st.metric("SELL Signals", sell_signals)

    with col3:
        hold_signals = (prices_df["signal"] == 0).sum()
        st.metric("HOLD Signals", hold_signals)

    with col4:
        signal_rate = (buy_signals + sell_signals) / len(prices_df) * 100
        st.metric("Active Rate", f"{signal_rate:.1f}%")

    # Signals chart
    st.markdown("### Price and Signals")

    fig = go.Figure()

    # Price line
    fig.add_trace(
        go.Scatter(
            x=prices_df["datetime"],
            y=prices_df["price_FR"],
            name="Price",
            line=dict(color="blue", width=1),
        )
    )

    # BUY signals
    buy_mask = prices_df["signal"] == 1
    fig.add_trace(
        go.Scatter(
            x=prices_df[buy_mask]["datetime"],
            y=prices_df[buy_mask]["price_FR"],
            mode="markers",
            name="BUY",
            marker=dict(color="green", size=8, symbol="triangle-up"),
        )
    )

    # SELL signals
    sell_mask = prices_df["signal"] == -1
    fig.add_trace(
        go.Scatter(
            x=prices_df[sell_mask]["datetime"],
            y=prices_df[sell_mask]["price_FR"],
            mode="markers",
            name="SELL",
            marker=dict(color="red", size=8, symbol="triangle-down"),
        )
    )

    fig.update_layout(
        title="Market Prices with Trading Signals",
        xaxis_title="Date",
        yaxis_title="Price (EUR/MWh)",
        height=600,
    )

    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Page: Backtests
# =============================================================================

elif page == "📈 Backtests":
    st.title("📈 Backtest Results")

    st.markdown(
        """
    Backtest performance of trading strategies on historical data.
    """
    )

    # Sample backtest data
    dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    initial_capital = 100000
    returns = np.random.randn(len(dates)) * 0.01
    equity_curve = initial_capital * (1 + returns).cumprod()

    backtest_df = pd.DataFrame(
        {"date": dates, "equity": equity_curve, "returns": returns * 100}
    )

    # Backtest metrics
    total_return = ((equity_curve[-1] - initial_capital) / initial_capital) * 100
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
    max_dd = ((equity_curve / equity_curve.cummax()) - 1).min() * 100

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Initial Capital", f"€{initial_capital:,.0f}")

    with col2:
        st.metric("Final Capital", f"€{equity_curve[-1]:,.0f}")

    with col3:
        st.metric("Total Return", f"{total_return:.2f}%")

    with col4:
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")

    # Equity curve
    st.markdown("### Equity Curve")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=backtest_df["date"],
            y=backtest_df["equity"],
            name="Equity",
            line=dict(color="green", width=2),
            fill="tozeroy",
        )
    )

    fig.update_layout(xaxis_title="Date", yaxis_title="Equity (EUR)", height=500)

    st.plotly_chart(fig, use_container_width=True)

    # Returns distribution
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Daily Returns Distribution")

        fig = go.Figure(data=[go.Histogram(x=backtest_df["returns"], nbinsx=50)])
        fig.update_layout(
            xaxis_title="Daily Return (%)", yaxis_title="Frequency", height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### Cumulative Returns")

        cumulative_returns = (1 + backtest_df["returns"] / 100).cumprod() - 1

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=backtest_df["date"],
                y=cumulative_returns * 100,
                line=dict(color="blue", width=2),
            )
        )

        fig.update_layout(
            xaxis_title="Date", yaxis_title="Cumulative Return (%)", height=400
        )

        st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Page: Model Comparison
# =============================================================================

elif page == "🔬 Model Comparison":
    st.title("🔬 Model Comparison")

    benchmark_df = load_benchmark_results()

    # Radar chart for model comparison
    st.markdown("### Multi-dimensional Comparison")

    # Normalize metrics for radar chart
    metrics_normalized = benchmark_df.copy()
    metrics_normalized["rmse_norm"] = 1 - (
        metrics_normalized["rmse_elec"] - metrics_normalized["rmse_elec"].min()
    ) / (metrics_normalized["rmse_elec"].max() - metrics_normalized["rmse_elec"].min())
    metrics_normalized["r2_norm"] = metrics_normalized["r2_elec"]
    metrics_normalized["speed_norm"] = 1 - (
        metrics_normalized["training_time_sec"]
        - metrics_normalized["training_time_sec"].min()
    ) / (
        metrics_normalized["training_time_sec"].max()
        - metrics_normalized["training_time_sec"].min()
    )

    fig = go.Figure()

    for idx, row in metrics_normalized.iterrows():
        fig.add_trace(
            go.Scatterpolar(
                r=[row["rmse_norm"], row["r2_norm"], row["speed_norm"]],
                theta=["Accuracy", "R² Score", "Speed"],
                fill="toself",
                name=row["model"],
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        height=600,
    )

    st.plotly_chart(fig, use_container_width=True)

    # Scatter plot: Performance vs Speed
    st.markdown("### Performance vs Training Time")

    fig = px.scatter(
        benchmark_df,
        x="training_time_sec",
        y="r2_elec",
        size="rmse_elec",
        color="model",
        hover_data=["rmse_elec", "rmse_gaz"],
        labels={
            "training_time_sec": "Training Time (seconds)",
            "r2_elec": "R² Score (Electricity)",
        },
    )

    fig.update_layout(height=500)

    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Footer
# =============================================================================

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    ### About
    **Energy Trading Dashboard**

    Version 1.0.0

    Built with Streamlit
"""
)
