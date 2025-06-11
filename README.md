<h1 align="center">
  <br>
  <img src="https://github.com/rav-lad/energy-demand-forecast/blob/main/energy_forcaster_logo.png" width="400">
  <br>
</h1>

<h4 align="center">Forecasting regional energy consumption using weather data and multiple ML models</h4>

<p align="center">
  <a href="https://pytorch-forecasting.readthedocs.io/">
    <img src="https://img.shields.io/badge/Model-TFT-blue?logo=pytorch&logoColor=white">
  </a>
  <a href="https://xgboost.ai/">
    <img src="https://img.shields.io/badge/Model-XGBoost-orange?logo=xgboost&logoColor=white">
  </a>
  <a href="https://lightgbm.readthedocs.io/">
    <img src="https://img.shields.io/badge/Model-LightGBM-green?logo=lightgbm">
  </a>
  <a href="https://scikit-learn.org/">
    <img src="https://img.shields.io/badge/Model-LinearRegression-blue?logo=scikit-learn">
  </a>
  <a href="https://www.python.org">
    <img src="https://img.shields.io/badge/Python-3.10-blue.svg?logo=python&logoColor=white">
  </a>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#features">Features</a> •
  <a href="#dataset">Dataset</a> •
  <a href="#models">Models</a> •
  <a href="#usage">Usage</a> •
  <a href="#future-work">Future Work</a> •
  <a href="#team">Team</a>
</p>

---

# Energy Demand Forecast

## Overview

**Energy Demand Forecast** is a project focused on predicting daily and hourly energy consumption (electricity and gas) at the regional level in France using historical weather data. It includes a range of machine learning models:

* Temporal Fusion Transformer (TFT)
* XGBoost Regressor
* Quantile LightGBM Regressor
* Linear models (Ridge and Lasso)

These models use open-source weather forecasts and historical energy usage to predict short-term demand across geographic regions.

Each model is accompanied by several **Jupyter notebooks** designed to explore various forecasting scenarios. A "scenario" refers to a specific modeling context — for example, training XGBoost with or without lag features, or comparing Ridge vs. Lasso regressions with different feature engineering strategies. These notebooks allow detailed inspection of performance trade-offs, feature importance, and forecast behaviors.

---

## Features

* **Multivariate Forecasting**

  * Predicts both electricity and gas consumption
* **Weather-conditioned Inputs**

  * High-resolution meteorological variables (temperature, precipitation, radiation, wind, etc.)
* **Multiple Model Types**

  * Deep learning (TFT), gradient boosting, linear models
* **Regional granularity**

  * Based on INSEE region codes
* **Preprocessed public dataset**

  * Ready-to-use format, available on Kaggle

---

## Dataset

**France Energy and Weather Data – Daily (2013–2024)**

🔗 Kaggle: [France Energy Weather Hourly](https://www.kaggle.com/datasets/ravvvvvvvvvvvv/france-energy-weather-hourly)

This dataset combines daily electricity and gas consumption with regional weather conditions across France from **Jan 1, 2013 to Dec 31, 2024**.

### Weather Data

* Collected from the [Open-Meteo API](https://open-meteo.com/)
* Data includes: temperature (min/max), precipitation, wind, radiation, sunshine duration, and more
* One representative city selected per region for weather retrieval

### Energy Data

* Sourced from the [data.gouv.fr ODRE API](https://odre.opendatasoft.com/explore/dataset/consommation-quotidienne-brute-regionale/table/?disjunctive.region&disjunctive.code_insee_region)
* Electricity and gas consumption (in MW) for each region
* Aggregated using INSEE codes and aligned with weather data

### Format

* Available as CSV
* `date` column is used as index — reset it if needed for manipulation

---

## Models

### 1. Temporal Fusion Transformer (TFT)

* Deep interpretable time-series model
* Captures temporal dependencies and static covariates
* Quantile forecasting with uncertainty bounds

### 2. XGBoost

* Gradient boosting decision trees
* Feature-based tabular model using lagged values and weather inputs
* Multiple scenario analyses implemented in notebooks (e.g., with/without lags)

### 3. LightGBM Quantile Regressor

* Predicts upper/lower bounds with quantile loss
* Optimized for forecasting scenarios with distributional outputs

### 4. Linear Models

* Ridge and Lasso regression for baseline comparison
* Lightweight and interpretable
* Tested across scenarios with different regularization and input representations

---
Results
This section presents key visual results comparing model performance across selected forecasting scenarios. Each figure is taken from the notebook of the specific model.

XGBoost – Classic Setup
<p align="center"> <img src="https://github.com/rav-lad/energy-demand-forecast/blob/main/xgboost_classic.png"" width="600"> </p>
This figure shows the predictions of the XGBoost model trained on standard features. It captures key consumption trends with relatively low variance and performs well on medium-term fluctuations.

Ridge Regression – Classic Setup
<p align="center"> <img src="https://github.com/rav-lad/energy-demand-forecast/blob/main/reglin_ridge_classic.png" width="600"> </p>
The Ridge regression model provides a smoothed estimate of energy demand, with regularization mitigating overfitting. Although interpretable, its ability to model non-linear dependencies is limited compared to tree-based models.

Lasso Regression – Classic Setup
<p align="center"> <img src="https://github.com/rav-lad/energy-demand-forecast/blob/main/reglin_lasso_classic.png" width="600"> </p>
Lasso regression applies L1 regularization, which results in feature selection. While it yields sparser models than Ridge, performance is slightly lower in this case due to the model discarding potentially relevant inputs.

LightGBM Quantile Regressor – Region 52 
<p align="center"> <img src="https://github.com/rav-lad/energy-demand-forecast/blob/main/lightgbm_52.png" width="600"> </p>
This plot illustrates the LightGBM quantile predictions for Region 52. It shows the median forecast along with upper and lower quantiles, providing an estimate of predictive uncertainty. This is particularly useful for probabilistic forecasting and risk-aware planning.


## Usage

### Setup

```bash
pip install -r requirements.txt
```

### Train TFT

```bash
python scripts/train_tft.py --frequency daily --max_epochs 30 --batch_size 128 --gpus 1
```

### Predict with TFT

```bash
python scripts/predict_future_tft.py --frequency daily
```

### Train and evaluate traditional ML models

```bash
python scripts/train_xgboost.py
python scripts/train_linear_models.py
python scripts/train_lightgbm_quantile.py
```

---

## Future Work

* Add inference notebooks for **hourly resolution**
* Build a dashboard for real-time forecast inspection
* Extend to **load forecasting** and **renewable production**
