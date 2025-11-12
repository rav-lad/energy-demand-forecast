# 🚀 Plan d'Amélioration du Projet

## Table des matières

- [Priorité 1 : Critique (Faire maintenant)](#priorité-1--critique)
- [Priorité 2 : Important (1-2 semaines)](#priorité-2--important)
- [Priorité 3 : Nice to have (1-2 mois)](#priorité-3--nice-to-have)
- [Priorité 4 : Long terme (3+ mois)](#priorité-4--long-terme)

---

## Priorité 1 : Critique

### ❌ 1. Tests Automatisés (IMPACT: ⭐⭐⭐⭐⭐)

**Problème** : Aucun test = risque de régressions
**Solution** :

```bash
# Structure créée
tests/
├── unit/           # Tests unitaires (fonctions isolées)
├── integration/    # Tests d'intégration (pipelines)
├── e2e/           # Tests end-to-end (workflows complets)
├── conftest.py    # Fixtures pytest
└── test_suite.py  # Suite de tests ✅ CRÉÉ

# Implementation
pip install pytest pytest-cov pytest-mock

# Commandes
make test          # Run all tests
make test-unit     # Unit tests only
make test-cov      # With coverage report
```

**Tests à implémenter** :

```python
# Unit tests
✓ test_data_loading()
✓ test_feature_engineering()
✓ test_model_prediction_shape()
✓ test_model_prediction_no_nan()
✓ test_backtest_engine_logic()
✓ test_signal_generation()

# Integration tests
✓ test_full_training_pipeline()
✓ test_data_to_prediction()
✓ test_benchmark_pipeline()

# E2E tests
✓ test_complete_workflow()
✓ test_docker_services()
```

**Ajouter au Makefile** :
```makefile
test:
    pytest tests/ -v

test-cov:
    pytest tests/ --cov=. --cov-report=html

test-unit:
    pytest tests/unit/ -v

test-integration:
    pytest tests/integration/ -v
```

---

### ❌ 2. CI/CD Pipeline (IMPACT: ⭐⭐⭐⭐⭐)

**Problème** : Build/tests manuels, pas de validation automatique
**Solution** : GitHub Actions

```yaml
# .github/workflows/ci.yml
name: CI Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov

      - name: Run tests
        run: pytest tests/ --cov

      - name: Upload coverage
        uses: codecov/codecov-action@v3

  docker:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build Docker image
        run: docker build -t energy-trading:test .

      - name: Test Docker image
        run: |
          docker run energy-trading:test python --version
          docker run energy-trading:test pip list

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Lint with flake8
        run: |
          pip install flake8
          flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Security scan
        uses: aquasecurity/trivy-action@master
        with:
          scan-type: 'fs'
          scan-ref: '.'
```

**Avantages** :
- ✅ Tests automatiques à chaque commit
- ✅ Validation avant merge
- ✅ Coverage tracking
- ✅ Security scanning
- ✅ Docker image validation

---

### ❌ 3. Configuration Management (IMPACT: ⭐⭐⭐⭐)

**Problème** : Chemins hardcodés dans le code
**Solution** : Centraliser avec Pydantic

```python
# src/config/settings.py
from pydantic import BaseSettings, Field
from pathlib import Path
from typing import List, Optional

class PathsConfig(BaseSettings):
    """Path configuration."""
    project_root: Path = Path(__file__).parent.parent.parent
    data_root: Path = project_root / "data"
    models_root: Path = project_root / "models"
    outputs_root: Path = project_root / "outputs"

    class Config:
        env_prefix = "PATH_"

class DataConfig(BaseSettings):
    """Data collection configuration."""
    entsoe_api_key: str = Field(..., env="ENTSOE_API_KEY")
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"
    countries: List[str] = ["FR", "DE", "ES"]

    class Config:
        env_prefix = "DATA_"

class ModelConfig(BaseSettings):
    """Model training configuration."""
    frequency: str = "daily"
    test_size: float = 0.2
    random_state: int = 42

    # XGBoost
    xgb_n_estimators: int = 1000
    xgb_learning_rate: float = 0.01

    # TFT
    tft_max_epochs: int = 30
    tft_batch_size: int = 128

    class Config:
        env_prefix = "MODEL_"

class TradingConfig(BaseSettings):
    """Trading configuration."""
    initial_capital: float = 100000
    transaction_cost: float = 0.001
    max_position_size: float = 10000

    class Config:
        env_prefix = "TRADING_"

class Settings(BaseSettings):
    """Global settings."""
    paths: PathsConfig = PathsConfig()
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    trading: TradingConfig = TradingConfig()

    debug: bool = False
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Global instance
settings = Settings()
```

**Usage** :
```python
from src.config.settings import settings

# Utiliser partout
data_path = settings.paths.data_root / "raw_data"
api_key = settings.data.entsoe_api_key
capital = settings.trading.initial_capital
```

---

### ❌ 4. Logging Structuré (IMPACT: ⭐⭐⭐⭐)

**Problème** : Logging inconsistant, difficile à débuguer
**Solution** : Logging centralisé avec rotation

```python
# src/utils/logger.py
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys

def setup_logger(
    name: str,
    log_file: str = None,
    level: int = logging.INFO,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup structured logger with rotation.

    Args:
        name: Logger name
        log_file: Path to log file
        level: Logging level
        max_bytes: Max size before rotation
        backup_count: Number of backup files

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler with rotation
    if log_file:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# Usage
from src.utils.logger import setup_logger

logger = setup_logger(__name__, log_file="outputs/logs/training.log")
logger.info("Starting training...")
logger.error("Failed with error", exc_info=True)
```

---

## Priorité 2 : Important

### ⚠️ 5. Data Validation avec Great Expectations (IMPACT: ⭐⭐⭐⭐)

**Problème** : Pas de validation de qualité des données
**Solution** : Great Expectations pour data quality

```python
# data_validation/validate_data.py
import great_expectations as ge

def validate_weather_data(df):
    """Validate weather data quality."""
    # Convert to GE dataset
    gdf = ge.from_pandas(df)

    # Expectations
    gdf.expect_column_values_to_be_between('temperature_2m_max', -50, 50)
    gdf.expect_column_values_to_be_between('temperature_2m_min', -60, 40)
    gdf.expect_column_values_to_not_be_null('date')
    gdf.expect_column_values_to_be_unique('date')
    gdf.expect_column_values_to_be_in_set('insee_region', [11, 24, 27, 28, 32, 44, 52, 53, 75, 76, 84, 93, 94])

    # Check min < max
    gdf.expect_column_pair_values_A_to_be_greater_than_B('temperature_2m_max', 'temperature_2m_min')

    # Validate
    results = gdf.validate()

    if not results['success']:
        print("⚠️ Data validation failed!")
        print(results)

    return results['success']
```

---

### ⚠️ 6. Model Versioning avec MLflow (IMPACT: ⭐⭐⭐⭐)

**Problème** : Pas de tracking des expériences
**Solution** : MLflow pour versioning et tracking

```python
# src/ml/mlflow_tracker.py
import mlflow
import mlflow.sklearn
import mlflow.pytorch

class ModelTracker:
    """Track experiments with MLflow."""

    def __init__(self, experiment_name: str):
        mlflow.set_experiment(experiment_name)

    def log_training(self, model, params, metrics, artifacts=None):
        """Log training run."""
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params(params)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log model
            if hasattr(model, 'sklearn'):
                mlflow.sklearn.log_model(model, "model")
            elif hasattr(model, 'pytorch'):
                mlflow.pytorch.log_model(model, "model")

            # Log artifacts
            if artifacts:
                for name, path in artifacts.items():
                    mlflow.log_artifact(path, artifact_path=name)

# Usage
tracker = ModelTracker("energy_demand_forecast")
tracker.log_training(
    model=xgb_model,
    params={'n_estimators': 1000, 'learning_rate': 0.01},
    metrics={'rmse': 650.2, 'r2': 0.935},
    artifacts={'feature_importance': 'figures/feature_importance.png'}
)
```

**Dashboard** :
```bash
mlflow ui
# → http://localhost:5000
```

---

### ⚠️ 7. Feature Store (IMPACT: ⭐⭐⭐)

**Problème** : Features recalculées à chaque fois
**Solution** : Feature store pour réutilisation

```python
# src/features/feature_store.py
import pandas as pd
from pathlib import Path
import hashlib

class FeatureStore:
    """Simple file-based feature store."""

    def __init__(self, store_path: str = "data/feature_store"):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(self, feature_name, params):
        """Generate cache key from feature name and params."""
        param_str = str(sorted(params.items()))
        hash_str = hashlib.md5(param_str.encode()).hexdigest()
        return f"{feature_name}_{hash_str}.parquet"

    def get(self, feature_name, params=None):
        """Get cached features."""
        params = params or {}
        cache_key = self._get_cache_key(feature_name, params)
        cache_path = self.store_path / cache_key

        if cache_path.exists():
            return pd.read_parquet(cache_path)
        return None

    def put(self, feature_name, df, params=None):
        """Save features to cache."""
        params = params or {}
        cache_key = self._get_cache_key(feature_name, params)
        cache_path = self.store_path / cache_key

        df.to_parquet(cache_path)

# Usage
feature_store = FeatureStore()

# Try to get cached
features = feature_store.get('weather_features', {'frequency': 'daily'})

if features is None:
    # Compute features
    features = compute_weather_features(data, frequency='daily')
    # Cache for next time
    feature_store.put('weather_features', features, {'frequency': 'daily'})
```

---

### ⚠️ 8. API REST avec FastAPI (IMPACT: ⭐⭐⭐⭐)

**Problème** : Pas d'interface pour prédictions en temps réel
**Solution** : API REST

```python
# src/api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd

app = FastAPI(title="Energy Trading API", version="2.0.0")

# Load models at startup
models = {}

@app.on_event("startup")
async def load_models():
    """Load models on startup."""
    models['xgboost'] = joblib.load('models/xgboost/xgb_daily.pkl')
    models['scaler'] = joblib.load('models/scalers/scaler_daily_reglin_xgboost.pkl')

class PredictionRequest(BaseModel):
    """Prediction request schema."""
    temperature_max: float
    temperature_min: float
    precipitation: float
    wind_speed: float
    solar_radiation: float
    region: int

class PredictionResponse(BaseModel):
    """Prediction response schema."""
    electricity_mw: float
    gas_mw: float
    model: str
    confidence: float

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make prediction."""
    try:
        # Prepare features
        features = pd.DataFrame([{
            'temperature_2m_max': request.temperature_max,
            'temperature_2m_min': request.temperature_min,
            'precipitation_sum': request.precipitation,
            'wind_speed_10m_max': request.wind_speed,
            'shortwave_radiation_sum': request.solar_radiation,
            f'insee_region_{request.region}': 1  # One-hot encoding
        }])

        # Scale
        features_scaled = models['scaler'].transform(features)

        # Predict
        prediction = models['xgboost'].predict(features_scaled)[0]

        return PredictionResponse(
            electricity_mw=float(prediction[0]),
            gas_mw=float(prediction[1]),
            model='xgboost',
            confidence=0.85
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy", "models_loaded": len(models)}

# Run with: uvicorn src.api.main:app --reload
```

**Add to docker-compose.yml** :
```yaml
api:
  build: .
  ports:
    - "8000:8000"
  command: uvicorn src.api.main:app --host 0.0.0.0 --port 8000
  volumes:
    - ./models:/app/models
```

---

## Priorité 3 : Nice to have

### 📊 9. Dashboard Streamlit (IMPACT: ⭐⭐⭐)

**Problème** : Pas de visualisation interactive
**Solution** : Dashboard Streamlit

```python
# src/dashboard/app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="Energy Trading Dashboard", layout="wide")

st.title("🔋 Energy Trading Research Dashboard")

# Sidebar
st.sidebar.header("Configuration")
model = st.sidebar.selectbox("Model", ["XGBoost", "LightGBM", "TFT"])
frequency = st.sidebar.radio("Frequency", ["Daily", "Hourly"])

# Main content
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Best Model", "TFT", "+5% accuracy")

with col2:
    st.metric("RMSE", "480.5 MW", "-10% vs baseline")

with col3:
    st.metric("Sharpe Ratio", "1.75", "+0.3")

# Predictions chart
st.subheader("Predictions vs Actual")
# Load data and create chart...

# Trading signals
st.subheader("Trading Signals")
# Show signal history...

# Performance metrics
st.subheader("Model Performance")
# Show benchmark table...
```

**Run** :
```bash
streamlit run src/dashboard/app.py
# → http://localhost:8501
```

---

### 📊 10. Monitoring avec Prometheus + Grafana (IMPACT: ⭐⭐⭐)

**Problème** : Pas de monitoring en production
**Solution** : Stack de monitoring

```yaml
# docker-compose.monitoring.yml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus-data:/prometheus
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana
    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin

  node-exporter:
    image: prom/node-exporter
    ports:
      - "9100:9100"

volumes:
  prometheus-data:
  grafana-data:
```

```python
# src/monitoring/metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Metrics
PREDICTIONS_TOTAL = Counter('predictions_total', 'Total predictions made')
PREDICTION_LATENCY = Histogram('prediction_latency_seconds', 'Prediction latency')
MODEL_RMSE = Gauge('model_rmse', 'Model RMSE', ['model_name'])

# Start metrics server
start_http_server(8001)

# Use in code
@PREDICTION_LATENCY.time()
def predict(data):
    PREDICTIONS_TOTAL.inc()
    result = model.predict(data)
    return result
```

---

### 🔄 11. Data Pipeline avec Airflow (IMPACT: ⭐⭐⭐)

**Problème** : Pas d'orchestration de workflows
**Solution** : Apache Airflow

```python
# dags/daily_forecast_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'energy-trading',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'daily_energy_forecast',
    default_args=default_args,
    description='Daily energy demand forecast pipeline',
    schedule_interval='0 6 * * *',  # Every day at 6 AM
    start_date=datetime(2024, 1, 1),
    catchup=False,
)

def collect_weather_data():
    """Collect latest weather data."""
    # Implementation
    pass

def collect_market_data():
    """Collect market prices."""
    pass

def train_models():
    """Retrain models if needed."""
    pass

def make_predictions():
    """Make predictions for next day."""
    pass

def send_report():
    """Send daily report."""
    pass

# Tasks
t1 = PythonOperator(task_id='collect_weather', python_callable=collect_weather_data, dag=dag)
t2 = PythonOperator(task_id='collect_market', python_callable=collect_market_data, dag=dag)
t3 = PythonOperator(task_id='train_models', python_callable=train_models, dag=dag)
t4 = PythonOperator(task_id='make_predictions', python_callable=make_predictions, dag=dag)
t5 = PythonOperator(task_id='send_report', python_callable=send_report, dag=dag)

# Dependencies
[t1, t2] >> t3 >> t4 >> t5
```

---

## Priorité 4 : Long terme

### 🧠 12. AutoML avec Optuna (IMPACT: ⭐⭐⭐)

**Problème** : Hyperparameter tuning manuel
**Solution** : AutoML automatique

```python
# src/ml/automl.py
import optuna
from sklearn.model_selection import cross_val_score

def objective(trial):
    """Optuna objective function."""
    # Suggest hyperparameters
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
    }

    # Train model
    model = XGBRegressor(**params)

    # Cross-validate
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

    return -scores.mean()

# Run optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=100)

print("Best params:", study.best_params)
```

---

### 🌍 13. Multi-Region Support (IMPACT: ⭐⭐⭐⭐)

**Problème** : Focus France uniquement
**Solution** : Expansion Europe

```python
# config/regions.py
REGIONS = {
    'FR': {
        'name': 'France',
        'subregions': [11, 24, 27, ...],
        'entsoe_code': '10YFR-RTE------C',
        'timezone': 'Europe/Paris'
    },
    'DE': {
        'name': 'Germany',
        'subregions': ['DE-LU', 'DE-AT-LU'],
        'entsoe_code': '10Y1001A1001A83F',
        'timezone': 'Europe/Berlin'
    },
    'ES': {
        'name': 'Spain',
        'subregions': [...],
        'entsoe_code': '10YES-REE------0',
        'timezone': 'Europe/Madrid'
    },
    # Add more countries...
}
```

---

### 🤖 14. Deep Learning Avancé (IMPACT: ⭐⭐⭐)

**Problème** : Modèles simples
**Solution** : Architectures avancées

```python
# Nouveaux modèles à explorer :

# 1. N-BEATS (Neural Basis Expansion Analysis)
from pytorch_forecasting import NBeats

# 2. DeepAR (Probabilistic forecasting)
from gluonts.model.deepar import DeepAREstimator

# 3. Transformer natif
import torch.nn as nn

class EnergyTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.transformer = nn.TransformerEncoder(...)

    def forward(self, x):
        return self.transformer(x)

# 4. Graph Neural Networks (pour relations régionales)
from torch_geometric.nn import GCNConv

class RegionalGNN(nn.Module):
    """Model regional dependencies with GNN."""
    pass
```

---

### 📈 15. Advanced Trading Strategies (IMPACT: ⭐⭐⭐⭐)

**Problème** : Une seule stratégie simple
**Solution** : Stratégies avancées

```python
# 1. Reinforcement Learning pour trading
from stable_baselines3 import PPO

class TradingEnv(gym.Env):
    """Custom trading environment."""
    pass

agent = PPO("MlpPolicy", env=TradingEnv())
agent.learn(total_timesteps=100000)

# 2. Portfolio Optimization
from scipy.optimize import minimize

def optimize_portfolio(predictions, prices, constraints):
    """Markowitz portfolio optimization."""
    pass

# 3. Risk Parity
def risk_parity_allocation(covariance_matrix):
    """Equal risk contribution allocation."""
    pass

# 4. Market Regime Detection
from hmmlearn import hmm

model = hmm.GaussianHMM(n_components=3)  # Bull, Bear, Sideways
model.fit(returns)
regimes = model.predict(returns)
```

---

## 🎯 Roadmap Suggéré

### Phase 1 : Stabilisation (Semaine 1-2)
- [x] Docker containerization
- [x] Unified pipeline
- [x] Documentation
- [ ] **Tests automatisés** ⭐
- [ ] **CI/CD** ⭐
- [ ] **Configuration management** ⭐
- [ ] **Logging structuré** ⭐

### Phase 2 : Qualité (Semaine 3-4)
- [ ] Data validation (Great Expectations)
- [ ] Model versioning (MLflow)
- [ ] Feature store
- [ ] API REST (FastAPI)

### Phase 3 : Production (Semaine 5-8)
- [ ] Dashboard (Streamlit)
- [ ] Monitoring (Prometheus/Grafana)
- [ ] Orchestration (Airflow)
- [ ] GraphCast integration

### Phase 4 : Avancé (Mois 3+)
- [ ] AutoML (Optuna)
- [ ] Multi-region support
- [ ] Advanced ML (N-BEATS, DeepAR, Transformers)
- [ ] Advanced trading (RL, portfolio optimization)

---

## 📊 Matrice Impact/Effort

```
Impact ↑
  ⭐⭐⭐⭐⭐ │ CI/CD           │ API REST         │ Multi-region
            │ Tests           │ Monitoring       │
  ⭐⭐⭐⭐   │ Logging         │ MLflow          │ Advanced ML
            │ Config          │ Feature Store    │
  ⭐⭐⭐     │ Dashboard       │ Airflow         │ AutoML
            │                 │                  │
            └─────────────────┴──────────────────┴──────→
              Facile          Moyen              Difficile
                                             Effort →
```

---

## 🚀 Quick Wins (Faire en premier)

1. **Tests** (2 jours) - Essentiel pour stabilité
2. **CI/CD** (1 jour) - Automatisation
3. **Logging** (1 jour) - Debug plus facile
4. **Config management** (1 jour) - Clean code

**Total : 5 jours pour bases solides** ⭐

---

## 💡 Recommendations

### Pour la production
1. **Priorité absolue** : Tests + CI/CD
2. **Ensuite** : Monitoring + API
3. **Puis** : Feature store + MLflow

### Pour la recherche
1. **Dashboard** Streamlit pour visualisation
2. **MLflow** pour tracking experiments
3. **AutoML** pour optimization

### Pour le trading
1. **API REST** pour signaux temps réel
2. **Monitoring** pour alertes
3. **RL** pour stratégies avancées

---

## 📚 Resources

### Testing
- Pytest: https://docs.pytest.org/
- Great Expectations: https://docs.greatexpectations.io/

### MLOps
- MLflow: https://mlflow.org/
- DVC: https://dvc.org/

### API
- FastAPI: https://fastapi.tiangolo.com/
- Streamlit: https://streamlit.io/

### Monitoring
- Prometheus: https://prometheus.io/
- Grafana: https://grafana.com/

### Orchestration
- Airflow: https://airflow.apache.org/
- Prefect: https://www.prefect.io/

---

**Prochaine étape recommandée** : Commencer par les Quick Wins (Tests + CI/CD)

```bash
# 1. Setup tests
make test-init

# 2. Run tests
make test

# 3. Setup CI/CD
# Create .github/workflows/ci.yml

# 4. Add logging
# Implement src/utils/logger.py
```
