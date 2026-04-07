"""
Energy Forecast Dashboard — Entry point.

Run:
  streamlit run dashboard/app.py
"""

import streamlit as st

st.set_page_config(
    page_title="Energy Price Forecast DSS",
    layout="wide",
)

st.title("Energy Price Forecast DSS")
st.markdown(
    "End-to-end ML system for French electricity spot price forecasting "
    "with decision support and MLOps monitoring.\n\n"
    "Use the navigation on the left to switch between pages:\n\n"
    "- **Data** — Dataset status, feature exploration, data quality\n"
    "- **Model** — Walk-forward validation, live inference, performance metrics\n"
    "- **Decision Support** — J+1 forecast with confidence intervals and price alerts\n"
    "- **Monitoring** — Data drift, model health, data quality, retraining signals"
)
