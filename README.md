# Energy Demand Forecasting System

**Système de prévision de la demande énergétique avec apprentissage automatique**

Prédiction de la consommation d'électricité et de gaz par région française basée sur des données météorologiques historiques et temps réel.

---

## 📋 Table des matières

- [Architecture du système](#-architecture-du-système)
- [Installation](#-installation)
- [Commandes Make](#-commandes-make)
- [Workflow complet](#-workflow-complet)
- [Design du système ML](#-design-du-système-ml)
- [Modèles disponibles](#-modèles-disponibles)
- [Structure des données](#-structure-des-données)
- [API et Dashboard](#-api-et-dashboard)

---

## 🏗️ Architecture du système

### Pipeline ML complet

```
┌─────────────────────┐
│  1. COLLECTE DATA   │  Open-Meteo API, ODRE, ENTSO-E
│  data_collection/   │  → data/raw_data/
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  2. TRAITEMENT      │  Feature engineering
│  data_processing/   │  → Scaler fitted & saved
│  transformation.py  │  → data/transformed_data/
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  3. ENTRAÎNEMENT    │  XGBoost, LightGBM, TFT
│  model/*/train_*.py │  → models/xgboost/xgb_daily.pkl ✅
│                     │  → models/scalers/scaler_*.pkl ✅
│                     │  → models/*/features_*.json ✅
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  4. INFÉRENCE       │  Load weights + scaler
│  src/api/main.py    │  → Prédictions temps réel
│  model/predict_*.py │
└─────────────────────┘
```

### Points clés

✅ **Les poids sont sauvegardés** : `joblib.dump(model, "models/xgboost/xgb_daily.pkl")`
✅ **Le scaler est sauvegardé** : `joblib.dump(scaler, "models/scalers/scaler_daily.pkl")`
✅ **L'ordre des features est sauvegardé** : `features_daily.json`
✅ **Même transformation pour training et inférence** : `fit_scaler=True/False`

---

## 🚀 Installation

### Prérequis

- Python 3.10+
- Docker (optionnel)
- Git

### Installation locale

```bash
# Cloner le repo
git clone https://github.com/rav-lad/energy-demand-forecast.git
cd energy-demand-forecast

# Créer environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer dépendances
pip install -r requirements.txt
```

### Configuration

```bash
# Créer fichier .env pour les API keys
cp .env.example .env

# Éditer .env avec votre clé ENTSO-E
nano .env
```

Ajouter votre clé API :
```
ENTSOE_API_KEY=your_api_key_here
```

Obtenir une clé gratuite : https://transparency.entsoe.eu/

---

## 🛠️ Commandes Make

Make simplifie l'exécution des commandes longues. Utilisez `make help` pour voir toutes les commandes.

### Commandes principales

#### 📦 Installation et setup

```bash
make install          # Installe toutes les dépendances Python
make setup            # Setup complet : install + création des répertoires
```

#### 📊 Collecte de données

```bash
make collect-all      # Collecte toutes les données (météo, énergie, marché)
make collect-weather  # Collecte uniquement les données météo historiques
make collect-market   # Collecte prix électricité et fondamentaux ENTSO-E
```

#### 🔄 Traitement des données

```bash
make process-data     # Transforme les données brutes en features ML
make split-data       # Split train/validation/test sets
```

#### 🤖 Entraînement des modèles

```bash
make train-xgboost    # Entraîne XGBoost (daily)
make train-lightgbm   # Entraîne LightGBM quantile
make train-tft        # Entraîne Temporal Fusion Transformer
make train-all        # Entraîne tous les modèles
```

#### 🔍 Optimisation hyperparamètres

```bash
make optim-xgboost    # Optuna pour XGBoost
make optim-lightgbm   # Optuna pour LightGBM
```

#### 🎯 Prédictions

```bash
make predict-xgboost  # Prédictions avec XGBoost
make predict-future   # Prédictions futures (14 jours)
```

#### 🌐 API et Dashboard

```bash
make api              # Lance API FastAPI (http://localhost:8000)
make dashboard        # Lance dashboard Streamlit (http://localhost:8501)
make api-docker       # Lance API dans Docker
```

#### 📈 Backtesting et Trading

```bash
make backtest         # Lance backtest de stratégies trading
make run-trading      # Exécute système de trading
```

#### 🧪 Tests et qualité

```bash
make test             # Lance tous les tests pytest
make lint             # Lint avec flake8
make format           # Format code avec black
make check            # Lint + format + tests
```

#### 🧹 Nettoyage

```bash
make clean            # Nettoie fichiers cache Python
make clean-data       # Supprime toutes les données collectées
make clean-models     # Supprime tous les modèles entraînés
make clean-all        # Nettoyage complet
```

#### 📚 Documentation

```bash
make docs             # Génère documentation
make help             # Affiche toutes les commandes disponibles
```

### Exemple d'utilisation

```bash
# Workflow complet du début
make setup              # 1. Setup initial
make collect-all        # 2. Collecte données
make process-data       # 3. Traitement
make train-xgboost      # 4. Entraînement
make api                # 5. Lancement API

# Ou workflow d'optimisation
make optim-xgboost      # Recherche meilleurs hyperparamètres
make train-xgboost      # Réentraîne avec meilleurs params
make predict-xgboost    # Prédictions
```

---

## 📖 Workflow complet

### Scénario 1 : Premier entraînement (from scratch)

```bash
# 1. Installer dépendances
make install

# 2. Collecter données brutes
make collect-weather     # → data/raw_data/weather/
make collect-market      # → data/raw_data/market_prices/

# 3. Traiter données (fit scaler)
python data_processing/transformation.py --frequency daily --fit_scaler

# Résultat :
# - data/transformed_data/train_daily_reglin_xgboost.csv
# - models/scalers/scaler_daily_reglin_xgboost.pkl ✅ SAUVEGARDÉ

# 4. Entraîner modèle
make train-xgboost

# Résultat :
# - models/xgboost/xgb_daily.pkl ✅ POIDS SAUVEGARDÉS
# - models/xgboost/features_daily.json ✅ ORDRE FEATURES

# 5. Vérifier que tout fonctionne
make api
# Ouvrir http://localhost:8000/docs
# Tester endpoint /predict
```

### Scénario 2 : Nouvelle prédiction (inférence)

```bash
# 1. Charger nouvelles données météo
python data_collection/pipeline.py --forecast

# 2. Transformer avec scaler existant (fit_scaler=False)
python data_processing/transformation.py --frequency daily --no-fit-scaler

# Code interne :
# scaler = joblib.load("models/scalers/scaler_daily.pkl")  # ✅ CHARGE SCALER
# df[numeric_cols] = scaler.transform(df[numeric_cols])

# 3. Charger modèle et prédire
python model/predict_future.py --model xgboost --frequency daily

# Code interne :
# model = joblib.load("models/xgboost/xgb_daily.pkl")  # ✅ CHARGE POIDS
# predictions = model.predict(X_transformed)
```

### Scénario 3 : Optimisation hyperparamètres

```bash
# 1. Optuna recherche meilleurs hyperparamètres
make optim-xgboost

# Résultat :
# - models/xgboost/best_params_daily.json

# 2. Réentraîner avec meilleurs params
make train-xgboost

# Le script train_xgboost.py charge automatiquement best_params_daily.json
```

---

## 🧠 Design du système ML

### 1. Transformation des données (`data_processing/transformation.py`)

Le fichier `transformation.py` est **crucial** car il gère :
- La création de features dérivées
- Le fit/transform du scaler
- La sauvegarde/chargement du scaler

#### Code clé : `transform_regression_and_xgb()`

```python
def transform_regression_and_xgb(
    df: pd.DataFrame,
    frequency: str = "daily",
    fit_scaler: bool = True,  # ← PARAMÈTRE CLÉ
    save: bool = True,
    scaler_path: Path | None = None,
):
    # 1. Feature engineering
    df['temp_mean'] = (df['temperature_2m_max'] + df['temperature_2m_min']) / 2
    df['temp_range'] = df['temperature_2m_max'] - df['temperature_2m_min']
    df['wind_range'] = df['wind_gusts_10m_max'] - df['wind_speed_10m_max']
    # ... 30+ features

    # 2. One-hot encoding
    df = pd.get_dummies(df, columns=['weather_code', 'wind_sector', 'insee_region'])

    # 3. Normalisation - PARTIE CRITIQUE
    target_cols = ["conso_elec_mw", "conso_gaz_mw"]
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.difference(target_cols)

    scaler_path = scaler_path or SCALER_DIR / f"scaler_{frequency}_reglin_xgboost.pkl"

    if fit_scaler:
        # TRAINING : Fit et sauvegarde ✅
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        joblib.dump(scaler, scaler_path)
        print(f"Scaler saved: {scaler_path}")
    else:
        # INFERENCE : Charge et transforme ✅
        scaler = joblib.load(scaler_path)
        df[numeric_cols] = scaler.transform(df[numeric_cols])
        print(f"Scaler loaded: {scaler_path}")

    return df
```

**Pourquoi c'est important ?**
- Le StandardScaler **doit être fitted sur les données d'entraînement**
- Puis **réutilisé tel quel pour l'inférence** (pas de re-fit!)
- Si on re-fit le scaler sur les nouvelles données → prédictions incorrectes

### 2. Entraînement (`model/xgboost/train_xgboost.py`)

```python
def train_xgboost(frequency: str):
    # 1. Charger données brutes
    df_raw = pd.read_csv(RAW_DIR / f"train_{frequency}.csv")

    # 2. Transformer avec scaler pré-fitted
    df = transform_regression_and_xgb(
        df_raw,
        frequency=frequency,
        fit_scaler=False,  # ← Utilise scaler existant
        save=False
    )

    y = df[["conso_elec_mw", "conso_gaz_mw"]]
    X = df.drop(columns=["conso_elec_mw", "conso_gaz_mw"])

    # 3. Charger meilleurs hyperparamètres
    with open(MODELS_DIR / f"best_params_{frequency}.json", "r") as f:
        best_params = json.load(f)

    # 4. Entraîner modèle
    base_model = XGBRegressor(**best_params, n_jobs=-1, random_state=42)
    model = MultiOutputRegressor(base_model)
    model.fit(X, y)

    # 5. SAUVEGARDER POIDS ✅
    joblib.dump(model, MODELS_DIR / f"xgb_{frequency}.pkl")
    print(f"Model saved: {MODELS_DIR / f'xgb_{frequency}.pkl'}")

    # 6. SAUVEGARDER ORDRE DES FEATURES ✅
    features = list(X.columns)
    with open(MODELS_DIR / f"features_{frequency}.json", "w") as f:
        json.dump(features, f, indent=4)
```

**Ce qui est sauvegardé :**
- `models/xgboost/xgb_daily.pkl` → Poids du modèle complet (MultiOutputRegressor avec XGBRegressor)
- `models/xgboost/features_daily.json` → Liste ordonnée des features
- `models/scalers/scaler_daily_reglin_xgboost.pkl` → StandardScaler fitted

### 3. Inférence (`src/api/main.py` ou `model/predict_future.py`)

```python
# 1. Charger modèle ✅
model = joblib.load("models/xgboost/xgb_daily.pkl")

# 2. Charger ordre des features ✅
with open("models/xgboost/features_daily.json", "r") as f:
    feature_order = json.load(f)

# 3. Transformer nouvelles données avec scaler existant ✅
df_new = transform_regression_and_xgb(
    df_raw_new,
    frequency="daily",
    fit_scaler=False,  # ← CHARGE le scaler existant
    save=False
)

# 4. S'assurer que l'ordre des features est correct
X_new = df_new[feature_order]

# 5. Prédire
predictions = model.predict(X_new)
```

### Schéma récapitulatif : Training vs Inference

| Étape | Training | Inference |
|-------|----------|-----------|
| **Données** | `data/raw_data/train_daily.csv` | Nouvelles données (API météo) |
| **Transformation** | `fit_scaler=True` → Fit + Save | `fit_scaler=False` → Load + Transform |
| **Scaler** | `scaler.fit_transform()` + `joblib.dump()` | `joblib.load()` + `scaler.transform()` |
| **Modèle** | `model.fit(X, y)` + `joblib.dump()` | `joblib.load()` + `model.predict(X)` |
| **Output** | Poids + Scaler + Features.json | Prédictions |

---

## 🤖 Modèles disponibles

### 1. XGBoost (Gradient Boosting)

**Fichiers** :
- Entraînement : `model/xgboost/train_xgboost.py`
- Optimisation : `model/xgboost/optim_xgboost.py`
- Prédiction : `model/xgboost/predict_xgboost.py`

**Utilisation** :
```bash
# Optimiser hyperparamètres avec Optuna
make optim-xgboost

# Entraîner avec meilleurs params
make train-xgboost

# Prédire
make predict-xgboost
```

**Sorties** :
- `models/xgboost/xgb_daily.pkl` (poids)
- `models/xgboost/features_daily.json` (features)
- `models/xgboost/best_params_daily.json` (hyperparams)

### 2. LightGBM Quantile (Prévisions probabilistes)

**Fichiers** :
- Entraînement : `model/Quantile/train_lightgbm_quantile.py`
- Optimisation : `model/Quantile/optim_lightgbm_quantile.py`

**Particularité** : Prédictions avec quantiles (5%, 50%, 95%) pour incertitude

**Utilisation** :
```bash
make train-lightgbm
```

**Transformation spéciale** :
```python
# Utilise transform_lightgbm_quantile() au lieu de transform_regression_and_xgb()
# Ajoute lags et rolling windows
df = transform_lightgbm_quantile(df, frequency="daily", lags=True)
```

### 3. Temporal Fusion Transformer (Deep Learning)

**Fichiers** :
- Entraînement : `model/DeepLearning/train_tft.py`
- Config : `model/DeepLearning/tft_config.yaml`

**Particularité** : Modèle attention-based pour séries temporelles, gère contexte long

**Utilisation** :
```bash
make train-tft
```

**Transformation spéciale** :
```python
# Utilise transform_dl() avec time_idx
df = transform_dl(df, seq_len=24, filter_too_short=True)
```

### 4. Régression Linéaire (Baseline)

**Fichiers** :
- Entraînement : `model/reg_lin/train_reg_lin.py`
- Variantes : Ridge, Lasso

**Utilisation** :
```bash
python model/reg_lin/train_reg_lin.py --frequency daily --model ridge
```

---

## 📁 Structure des données

```
data/
├── raw_data/                       # Données brutes collectées
│   ├── energy/                    # Consommation ODRE
│   │   ├── conso_elec_daily.csv
│   │   └── conso_gaz_daily.csv
│   ├── weather/                   # Météo Open-Meteo
│   │   ├── weather_daily_region11.csv
│   │   └── ...
│   ├── market_prices/             # Prix ENTSO-E
│   │   └── prices_FR_2020_2024.csv
│   └── fundamentals/              # Production, load
│       ├── generation_FR.csv
│       └── load_FR.csv
│
├── modified_data/                 # Données fusionnées et nettoyées
│   ├── train_daily.csv           # Dataset d'entraînement
│   ├── test_daily.csv            # Dataset de test
│   └── train_hourly.csv
│
└── transformed_data/              # Features engineered
    ├── train_daily_reglin_xgboost.csv
    └── train_daily_lightgbm_quantile_withlags.csv

models/
├── xgboost/
│   ├── xgb_daily.pkl             # ✅ Poids modèle
│   ├── features_daily.json        # ✅ Ordre features
│   └── best_params_daily.json     # Hyperparamètres Optuna
│
├── scalers/
│   └── scaler_daily_reglin_xgboost.pkl  # ✅ StandardScaler fitted
│
├── Quantile/
│   ├── lgb_quantile_daily.pkl
│   └── best_params_daily.json
│
└── DeepLearning/
    └── tft_daily/
        └── checkpoints/
            └── best_model.ckpt
```

---

## 🌐 API et Dashboard

### API FastAPI

```bash
# Lancer API
make api
# ou
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Accéder à la doc interactive
http://localhost:8000/docs
```

**Endpoints principaux** :
- `POST /predict` : Prédiction pour nouvelles données
- `GET /health` : Health check
- `GET /models` : Liste des modèles disponibles

**Exemple d'utilisation** :
```python
import requests

data = {
    "date": "2024-11-15",
    "temperature_2m_max": 15.5,
    "temperature_2m_min": 8.2,
    "rain_sum": 2.3,
    # ... autres features météo
    "insee_region": "11"
}

response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())
# {"conso_elec_mw": 4523.45, "conso_gaz_mw": 1234.56}
```

### Dashboard Streamlit

```bash
# Lancer dashboard
make dashboard
# ou
streamlit run src/dashboard/app.py

# Accéder au dashboard
http://localhost:8501
```

**Fonctionnalités** :
- Visualisation des prédictions historiques
- Comparaison des modèles
- Analyse des erreurs
- Exploration des features importantes

---

## 🔬 MLOps et Expérimentation

### MLflow Tracking

```python
# Activation dans train_xgboost.py
from src.ml.mlflow_tracker import MLflowTracker

tracker = MLflowTracker(experiment_name="xgboost_daily")
tracker.log_params(best_params)
tracker.log_metrics({"rmse_elec": rmse_elec, "rmse_gaz": rmse_gaz})
tracker.log_model(model, "xgboost")
```

```bash
# Visualiser expériences
mlflow ui
# Accéder à http://localhost:5000
```

### Optuna Hyperparameter Tuning

```python
# Exemple dans optim_xgboost.py
from src.ml.optuna_tuner import OptunaTuner

tuner = OptunaTuner(n_trials=100)
best_params = tuner.optimize(
    objective_func=objective,
    study_name="xgboost_daily"
)
```

---

## 🧪 Tests et CI/CD

### Tests locaux

```bash
# Tous les tests
make test

# Tests spécifiques
pytest tests/test_transformation.py -v

# Avec coverage
pytest --cov=data_processing --cov-report=html
```

### CI/CD GitHub Actions

Le projet utilise GitHub Actions pour CI/CD automatique :

**3 jobs automatiques sur chaque push** :

1. **`test`** : Tests + Linting + Formatting
   - `pytest` : Tests unitaires
   - `flake8` : Linting
   - `black --check` : Vérification formatage

2. **`docker-build`** : Construction image Docker
   - Build de l'image API
   - Vérification qu'elle démarre correctement

3. **`security-scan`** : Scan de sécurité
   - Trivy : Scan vulnérabilités dépendances
   - Upload résultats vers GitHub Security

**Fichier** : `.github/workflows/ci.yml`

---

## 🐳 Docker

### Build et run API

```bash
# Build image
docker build -t energy-forecast-api .

# Run conteneur
docker run -p 8000:8000 energy-forecast-api

# Ou avec Make
make api-docker
```

### Docker Compose (API + Dashboard)

```bash
docker-compose up

# Services disponibles :
# - API : http://localhost:8000
# - Dashboard : http://localhost:8501
```

---

## 📊 Dataset Kaggle

**France Energy and Weather Data – Daily & Hourly (2013–2024)**

🔗 [Kaggle Dataset](https://www.kaggle.com/datasets/ravvvvvvvvvvvv/france-energy-weather-hourly)

**Contenu** :
- 13 régions françaises (INSEE codes)
- Données quotidiennes et horaires
- Variables météo : température, vent, radiation solaire, précipitations
- Consommation électricité et gaz

**Utilisation** :
```bash
# Télécharger depuis Kaggle
kaggle datasets download -d ravvvvvvvvvvvv/france-energy-weather-hourly

# Extraire dans data/raw_data/
unzip france-energy-weather-hourly.zip -d data/raw_data/
```

---

## 🚨 Troubleshooting

### Problème : Erreur "scaler not found"

```python
FileNotFoundError: models/scalers/scaler_daily_reglin_xgboost.pkl not found
```

**Solution** : Fit le scaler d'abord
```bash
python data_processing/transformation.py --frequency daily --fit-scaler
```

### Problème : Prédictions incorrectes

**Cause possible** : Ordre des features incorrect

**Solution** : Vérifier que l'ordre correspond à `features_daily.json`
```python
with open("models/xgboost/features_daily.json") as f:
    feature_order = json.load(f)
X = df[feature_order]  # ✅ Réorganiser colonnes
```

### Problème : API key ENTSO-E invalide

```python
requests.exceptions.HTTPError: 401 Unauthorized
```

**Solution** : Vérifier `.env`
```bash
cat .env
# ENTSOE_API_KEY doit être défini
```

---

## 📚 Ressources

### APIs et données
- [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/) - Données marché électricité
- [ODRE (data.gouv.fr)](https://odre.opendatasoft.com/) - Consommation énergie France
- [Open-Meteo](https://open-meteo.com/) - API météo gratuite

### Documentation technique
- [XGBoost Docs](https://xgboost.readthedocs.io/)
- [LightGBM Docs](https://lightgbm.readthedocs.io/)
- [PyTorch Forecasting](https://pytorch-forecasting.readthedocs.io/) - TFT
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [Streamlit Docs](https://docs.streamlit.io/)

### Papers
- [Temporal Fusion Transformers (2021)](https://arxiv.org/abs/1912.09363)
- [Electricity Price Forecasting Review](https://doi.org/10.1016/j.apenergy.2020.114983)

---

## 🤝 Contributing

Contributions bienvenues ! Domaines d'amélioration :

- Nouveaux modèles (Prophet, N-BEATS, etc.)
- Sources de données supplémentaires
- Optimisations performance
- Tests unitaires
- Documentation

---

## 📝 License

Projet éducatif et de recherche.

---

## 👤 Auteur

Créé par [@rav-lad](https://github.com/rav-lad)

**Contact** : [Créer une issue](https://github.com/rav-lad/energy-demand-forecast/issues)

---

<p align="center">
  <b>⚡ Prévision de la demande énergétique avec ML ⚡</b>
</p>

<p align="center">
  Made with ❤️ for energy forecasting research
</p>
