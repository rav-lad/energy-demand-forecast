# 🧠 Machine Learning Models Documentation

## Vue d'ensemble

Le projet implémente **5 familles de modèles** pour la prédiction de consommation énergétique :

| Modèle | Type | Incertitude | Lags | Meilleur usage |
|--------|------|-------------|------|----------------|
| **XGBoost** | Gradient Boosting | Monte Carlo | ❌ | Point forecasts haute précision |
| **LightGBM Quantile** | Gradient Boosting | Quantiles natifs | ✅ | Forecasts probabilistes |
| **TFT** | Deep Learning | Quantiles natifs | ✅ | Multi-horizon avec interprétabilité |
| **Ridge** | Linear (L2) | Monte Carlo | ❌ | Baseline régularisé |
| **Lasso** | Linear (L1) | Monte Carlo | ❌ | Sélection de features |

---

## 1. XGBoost

### Architecture

```python
MultiOutputRegressor(XGBRegressor)
```

- **Objectif** : Régression multi-sortie
- **Cibles** : [conso_elec_mw, conso_gaz_mw]
- **Framework** : xgboost >= 2.0.0

### Hyperparamètres clés

```python
{
    'n_estimators': 1000,          # Nombre d'arbres
    'learning_rate': 0.01,         # Taux d'apprentissage
    'max_depth': 7,                # Profondeur max des arbres
    'subsample': 0.8,              # Fraction d'échantillons
    'colsample_bytree': 0.8,       # Fraction de features
    'gamma': 0,                    # Seuil de split
    'reg_alpha': 0.1,              # Régularisation L1
    'reg_lambda': 5,               # Régularisation L2
    'objective': 'reg:squarederror'
}
```

### Features

**Types** :
- Temporelles : month, day, weekday, dayofyear, sin/cos transformations
- Météo : temperature, precipitation, wind, radiation, sunshine
- Dérivées : temp_mean, temp_range, apparent_temp, radiation ratios
- Catégorielles (one-hot) : weather_code, wind_sector, insee_region
- Vacances : jours fériés français

**Preprocessing** :
```python
StandardScaler()  # Normalisation Z-score
```

### Formats

**Input** :
```python
X.shape = (n_samples, n_features)
# Exemple : (1000, 45)
```

**Output** :
```python
y.shape = (n_samples, 2)  # [elec, gaz]
```

### Persistence

```
models/xgboost/
├── xgb_daily.pkl                    # Modèle entraîné
├── features_daily.json              # Liste des features
└── scalers/scaler_daily_reglin_xgboost.pkl  # Scaler fitted
```

### Pipeline

```bash
# 1. Hyperparameter search
python model/gridsearch.py --model xgboost --frequency daily

# 2. Training
python model/xgboost/train_xgboost.py --frequency daily

# 3. Inference
python model/predict_future.py --model xgboost --frequency daily --mode classic
```

### Performance attendue

- **RMSE** : ~500-800 MW pour électricité
- **R²** : ~0.92-0.95
- **Training time** : 2-5 minutes (CPU)

---

## 2. LightGBM Quantile

### Architecture

```python
6 modèles indépendants :
- conso_elec_mw : q5, q50, q95
- conso_gaz_mw : q5, q50, q95
```

- **Objectif** : Régression quantile
- **Framework** : lightgbm >= 4.0.0
- **Validation** : Time Series CV (5 folds)

### Hyperparamètres clés

```python
{
    'objective': 'quantile',
    'alpha': [0.05, 0.5, 0.95],    # Quantile à prédire
    'learning_rate': 0.01,
    'n_estimators': 200,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_samples': 10,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0
}
```

### Features

**Deux modes** :

#### Mode "withlags" (pour backtesting)
```python
Features = [
    # Lags
    'conso_elec_mw_lag1', 'conso_elec_mw_lag7',
    'conso_gaz_mw_lag1', 'conso_gaz_mw_lag7',
    'temperature_2m_mean_lag1', 'temperature_2m_mean_lag7',

    # Rolling windows
    'conso_elec_mw_rolling_3', 'conso_elec_mw_rolling_7',

    # Delta features
    'temp_delta_1', 'temp_delta_7',

    # + toutes les features instantanées
]
```

#### Mode "withoutlags" (pour forecasting futur)
```python
Features = [
    # Temporelles
    'month', 'day', 'weekday', 'week_number',
    'month_sin', 'month_cos', 'day_sin', 'day_cos',

    # Météo brutes
    'temperature_*', 'precipitation_*', 'wind_*', 'radiation_*',

    # Interactions
    'temp_ferie_interaction',
    'temp_radiation_interaction',
    'sun_week_interaction',

    # Catégorielles (one-hot)
    'weather_code_*', 'wind_sector_*', 'insee_region_*'
]
```

**Preprocessing** :
```python
# AUCUNE normalisation !
# Les modèles à base d'arbres ne nécessitent pas de scaling
```

### Formats

**Input** :
```python
X.shape = (n_samples, n_features)
# withlags : ~80 features
# withoutlags : ~60 features
```

**Output** :
```python
# Par modèle (1 quantile, 1 cible)
y.shape = (n_samples,)

# Combinés (6 modèles)
predictions.shape = (n_samples, 6)
# [elec_q5, elec_q50, elec_q95, gaz_q5, gaz_q50, gaz_q95]
```

### Persistence

```
models/Quantile/lightgbm_quantile/
├── conso_elec_mw_q5_daily_withlags.pkl
├── conso_elec_mw_q50_daily_withlags.pkl
├── conso_elec_mw_q95_daily_withlags.pkl
├── conso_gaz_mw_q5_daily_withlags.pkl
├── conso_gaz_mw_q50_daily_withlags.pkl
├── conso_gaz_mw_q95_daily_withlags.pkl
├── features_daily_withlags.json
└── metrics_daily_withlags.json
```

### Pipeline

```bash
# 1. Hyperparameter search
python model/gridsearch.py --model lightgbm_quantile --frequency daily --lags with

# 2. Training
python model/Quantile/train_lightgbm_quantile.py --frequency daily --lags with

# 3. Inference
python model/predict_future.py --model lightgbm_quantile --frequency daily --lags without
```

### Performance attendue

- **RMSE (médiane)** : ~480-750 MW
- **Couverture quantiles** : 90-95% des valeurs dans [q5, q95]
- **Training time** : 5-10 minutes (6 modèles)

---

## 3. Temporal Fusion Transformer (TFT)

### Architecture

```python
TFT (PyTorch Forecasting)
├── Encoder (lookback : 24 timesteps)
├── Variable Selection Network
├── LSTM Layers
├── Multi-Head Attention
├── Gate Add & Norm
└── Decoder (forecast : 1 timestep)
```

- **Framework** : pytorch-forecasting >= 1.0.0
- **Backend** : PyTorch Lightning
- **Type** : Sequence-to-sequence avec attention

### Configuration

```python
{
    # Architecture
    'hidden_size': 32,
    'attention_head_size': 4,
    'dropout': 0.1,
    'hidden_continuous_size': 16,

    # Training
    'learning_rate': 1e-3,
    'max_epochs': 30,
    'batch_size': 128,
    'gradient_clip_val': 0.1,

    # Time series
    'max_encoder_length': 24,      # Lookback window
    'max_prediction_length': 1,    # Forecast horizon

    # Loss
    'loss': QuantileLoss(quantiles=[0.1, 0.5, 0.9])
}
```

### Features

**Static categoricals** :
```python
['insee_region']  # Identifiant de la région
```

**Time-varying known** (18 features) :
```python
[
    'temperature_2m_max', 'temperature_2m_min',
    'apparent_temperature_max', 'apparent_temperature_min',
    'precipitation_sum', 'rain_sum', 'snowfall_sum',
    'precipitation_hours',
    'weather_code',
    'sunrise', 'sunset',  # En secondes depuis minuit
    'sunshine_duration', 'daylight_duration',
    'wind_speed_10m_max', 'wind_gusts_10m_max',
    'wind_direction_10m_dominant',
    'shortwave_radiation_sum',
    'et0_fao_evapotranspiration'
]
```

**Time-varying unknown** (targets) :
```python
['conso_elec_mw', 'conso_gaz_mw']
```

**Preprocessing** :
```python
# Normalisation interne avec TorchNormalizer
# Pas de scaling externe requis
```

### Formats

**Input** :
```python
# TimeSeriesDataSet structure
{
    'encoder_cont': (batch, encoder_length, n_features),
    'encoder_cat': (batch, encoder_length, n_categoricals),
    'decoder_cont': (batch, decoder_length, n_features),
    'decoder_cat': (batch, decoder_length, n_categoricals)
}
```

**Output** :
```python
predictions.shape = (n_samples, forecast_horizon, n_targets)
# Exemple : (365, 1, 2)  pour daily sur 1 an
```

### Persistence

```
models/tft/
├── checkpoints/
│   └── best_tft.ckpt               # PyTorch Lightning checkpoint
├── tft_training_dataset.pt         # TimeSeriesDataSet structure
└── logs/                            # TensorBoard logs
```

### Pipeline

```bash
# 1. Training
python model/DeepLearning/train_tft.py \
    --frequency daily \
    --max_epochs 30 \
    --batch_size 128 \
    --gpus 1  # ou 0 pour CPU

# 2. Inference
python model/predict_future_tft.py --frequency daily
```

### Performance attendue

- **RMSE** : ~450-700 MW
- **Training time** : 10-30 minutes (GPU) / 2-4 heures (CPU)
- **Inference time** : ~1 seconde pour 365 jours

### Avantages

✅ **Interprétabilité** : Variable importance via attention weights
✅ **Multi-horizon** : Peut prédire plusieurs jours en avance
✅ **Missing data** : Robuste aux données manquantes
✅ **Quantiles natifs** : Incertitude intégrée

---

## 4. Ridge Regression

### Architecture

```python
MultiOutputRegressor(Ridge)
```

- **Type** : Régression linéaire avec régularisation L2
- **Complexité** : O(n * p) - très rapide
- **Framework** : scikit-learn

### Hyperparamètres clés

```python
{
    'alpha': 10.0,           # Force de la régularisation L2
    'fit_intercept': True,
    'solver': 'auto'
}
```

### Features

Identiques à XGBoost (voir section XGBoost).

**Preprocessing** :
```python
StandardScaler()  # OBLIGATOIRE pour modèles linéaires
```

### Performance attendue

- **RMSE** : ~700-1000 MW
- **R²** : ~0.85-0.90
- **Training time** : < 1 minute

### Avantages

✅ **Rapidité** : Entraînement et inférence ultra-rapides
✅ **Stabilité** : Régularisation L2 prévient l'overfitting
✅ **Interprétabilité** : Coefficients linéaires analysables

---

## 5. Lasso Regression

### Architecture

```python
MultiOutputRegressor(Lasso)
```

- **Type** : Régression linéaire avec régularisation L1
- **Particularité** : Sélection automatique de features (coefficients → 0)
- **Framework** : scikit-learn

### Hyperparamètres clés

```python
{
    'alpha': 1.0,               # Force de la régularisation L1
    'fit_intercept': True,
    'selection': 'cyclic',      # Méthode de sélection
    'max_iter': 10000
}
```

### Features

Identiques à XGBoost, mais le Lasso va **sélectionner automatiquement** les plus importantes.

### Performance attendue

- **RMSE** : ~720-1050 MW (légèrement moins bon que Ridge)
- **R²** : ~0.83-0.88
- **Features sélectionnées** : ~30-40% des features initiales

### Avantages

✅ **Sélection de features** : Identifie automatiquement les variables importantes
✅ **Modèle sparse** : Moins de features = plus simple à interpréter
✅ **Rapidité** : Comparable à Ridge

---

## Comparaison des modèles

### Performance (RMSE moyen sur test)

```
TFT           : 450-700 MW  ⭐⭐⭐⭐⭐
LightGBM      : 480-750 MW  ⭐⭐⭐⭐⭐
XGBoost       : 500-800 MW  ⭐⭐⭐⭐
Ridge         : 700-1000 MW ⭐⭐⭐
Lasso         : 720-1050 MW ⭐⭐⭐
```

### Temps d'entraînement

```
Lasso         : < 1 min      ⚡⚡⚡⚡⚡
Ridge         : < 1 min      ⚡⚡⚡⚡⚡
XGBoost       : 2-5 min      ⚡⚡⚡⚡
LightGBM      : 5-10 min     ⚡⚡⚡
TFT (GPU)     : 10-30 min    ⚡⚡
TFT (CPU)     : 2-4 heures   ⚡
```

### Incertitude

```
LightGBM      : Quantiles natifs     ⭐⭐⭐⭐⭐
TFT           : Quantiles natifs     ⭐⭐⭐⭐⭐
XGBoost       : Monte Carlo          ⭐⭐⭐
Ridge         : Monte Carlo          ⭐⭐⭐
Lasso         : Monte Carlo          ⭐⭐⭐
```

### Interprétabilité

```
Lasso         : Coefficients + sélection  ⭐⭐⭐⭐⭐
Ridge         : Coefficients linéaires    ⭐⭐⭐⭐⭐
XGBoost       : Feature importance        ⭐⭐⭐⭐
LightGBM      : Feature importance        ⭐⭐⭐⭐
TFT           : Attention weights         ⭐⭐⭐
```

### Complexité d'utilisation

```
Ridge         : Très simple           ⭐⭐⭐⭐⭐
Lasso         : Très simple           ⭐⭐⭐⭐⭐
XGBoost       : Simple                ⭐⭐⭐⭐
LightGBM      : Modérée (6 modèles)  ⭐⭐⭐
TFT           : Complexe              ⭐⭐
```

---

## Recommandations d'usage

### Production / Trading en temps réel
**→ XGBoost** ou **LightGBM (withoutlags)**
- Rapides en inférence
- Bonnes performances
- Pas besoin de séquences historiques

### Recherche / Analyse de risque
**→ LightGBM (withlags)** ou **TFT**
- Quantiles natifs pour intervalles de confiance
- Meilleure estimation de l'incertitude

### Baseline / Prototypage rapide
**→ Ridge** ou **Lasso**
- Entraînement ultra-rapide
- Bon pour valider pipeline de données

### Multi-horizon forecasting
**→ TFT**
- Seul modèle capable de prédire plusieurs jours en avance
- Attention mechanisms pour dépendances temporelles

### Feature engineering
**→ Lasso**
- Utiliser pour identifier les features importantes
- Ensuite réentraîner XGBoost avec features sélectionnées

---

## Formats de données

### Fichiers d'entrée attendus

```
data/modified_data/
├── train_daily.csv              # Données d'entraînement
├── test_daily.csv               # Données de test
├── train_hourly.csv
└── test_hourly.csv
```

**Colonnes requises** :
```python
required_columns = [
    'date',                      # Index temporel
    'insee_region',              # Code région
    'conso_elec_mw',            # Cible 1
    'conso_gaz_mw',             # Cible 2

    # Météo
    'temperature_2m_max',
    'temperature_2m_min',
    'precipitation_sum',
    'wind_speed_10m_max',
    'shortwave_radiation_sum',
    # ... autres variables météo

    # Dérivées (créées automatiquement si absentes)
    'temperature_2m_mean',
    'apparent_temperature_mean',
    # ...
]
```

### Fichiers de sortie

```
models/{model_name}/
├── *.pkl                        # Modèles entraînés (joblib)
├── *.ckpt                       # Checkpoints PyTorch (TFT)
├── features_*.json              # Liste des features utilisées
├── metrics_*.json               # Métriques d'évaluation
└── scalers/*.pkl                # Scalers fitted
```

---

## Pipeline de données unifié

```
1. Data Collection
   ↓
2. Data Processing (transformation.py)
   ├── transform_regression_and_xgb()    → XGBoost, Ridge, Lasso
   ├── transform_lightgbm_quantile()     → LightGBM
   └── transform_dl()                     → TFT
   ↓
3. Train/Test Split (split_train_test.py)
   ↓
4. Hyperparameter Search (gridsearch.py)
   ↓
5. Model Training
   ├── train_xgboost.py
   ├── train_lightgbm_quantile.py
   ├── train_tft.py
   └── train_reg_lin.py
   ↓
6. Model Evaluation (test.py)
   ↓
7. Inference (predict_future.py, predict_future_tft.py)
```

---

## Préparation pour GraphCast

### Actuellement : Open-Meteo API
```python
# data_recuperation/data_recuperation_meteo.py
variables = [
    'temperature_2m_max',
    'temperature_2m_min',
    'precipitation_sum',
    # ... 18 variables météo
]
```

### Futur : GraphCast
```python
# data_recuperation/data_graphcast.py (à créer)

# GraphCast fournit :
- Résolution : 0.25° (28 km)
- Fréquence : 6 heures
- Horizon : jusqu'à 10 jours
- Variables : ~100+ (ERA5 reanalysis)

# Avantages :
✅ Couverture mondiale (pas juste Europe)
✅ Meilleure résolution spatiale
✅ Plus de variables atmosphériques
✅ Meilleures prévisions éoliennes/solaires
```

### Architecture pour GraphCast

```python
# Structure proposée
data/raw_data/graphcast/
├── forecasts/
│   ├── 2024-01-01_00h.nc       # NetCDF files
│   ├── 2024-01-01_06h.nc
│   └── ...
├── processed/
│   └── france_regions_daily.csv
└── metadata/
    └── grid_mapping.json        # Mapping coords → régions

# Processing pipeline
1. Download GraphCast outputs (NetCDF)
2. Extract France grid points (lat/lon)
3. Aggregate by INSEE region
4. Resample to daily/hourly
5. Merge with existing data
```

---

## Scripts de benchmark

### Benchmark complet

```bash
# À créer : scripts/benchmark_models.py
python scripts/benchmark_models.py \
    --frequency daily \
    --test_period 2023-01-01:2024-12-31 \
    --models xgboost lightgbm tft ridge lasso \
    --output outputs/benchmark_results.csv
```

**Métriques calculées** :
- RMSE, MAE, MAPE par modèle
- R² score
- Temps d'entraînement
- Temps d'inférence
- Mémoire utilisée

### Output attendu

```csv
model,frequency,rmse_elec,rmse_gaz,r2_elec,r2_gaz,train_time,inference_time
xgboost,daily,650.2,420.5,0.935,0.918,180.5,0.12
lightgbm,daily,520.8,385.2,0.952,0.935,450.3,0.18
tft,daily,480.5,370.1,0.961,0.942,1800.2,1.05
ridge,daily,850.3,580.4,0.885,0.875,45.2,0.05
lasso,daily,920.5,650.8,0.865,0.852,50.8,0.05
```

---

## Dépendances par modèle

### XGBoost
```
xgboost>=2.0.0
scikit-learn>=1.3.0
joblib>=1.3.0
```

### LightGBM
```
lightgbm>=4.0.0
scikit-learn>=1.3.0
joblib>=1.3.0
```

### TFT
```
torch>=2.0.0
pytorch-lightning>=2.0.0
pytorch-forecasting>=1.0.0
```

### Linear Models
```
scikit-learn>=1.3.0
joblib>=1.3.0
```

### GraphCast (futur)
```
xarray>=2023.0.0
netCDF4>=1.6.0
cfgrib>=0.9.0
```

---

## Troubleshooting

### XGBoost : "DMatrix not initialized"
→ Vérifier que features ont la bonne dimension
→ S'assurer que scaler est bien chargé

### LightGBM : "NaN in predictions"
→ Vérifier handling des lags (forward/backfill)
→ S'assurer qu'il n'y a pas de NaN dans les features

### TFT : "CUDA out of memory"
→ Réduire batch_size (128 → 64)
→ Réduire hidden_size (32 → 16)
→ Utiliser CPU au lieu de GPU

### Linear : "ValueError: array contains inf"
→ Vérifier le scaling (StandardScaler)
→ Éliminer features avec variance nulle

---

## Ressources

### Documentation
- XGBoost : https://xgboost.readthedocs.io/
- LightGBM : https://lightgbm.readthedocs.io/
- PyTorch Forecasting : https://pytorch-forecasting.readthedocs.io/
- GraphCast Paper : https://www.science.org/doi/10.1126/science.adi2336

### Papiers de recherche
- TFT (2021) : https://arxiv.org/abs/1912.09363
- GraphCast (2023) : https://www.science.org/doi/10.1126/science.adi2336
- Energy Forecasting Review : https://doi.org/10.1016/j.apenergy.2020.114983

---

**Dernière mise à jour** : 2024-11-12
