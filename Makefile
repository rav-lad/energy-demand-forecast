.PHONY: help build up down shell jupyter train benchmark test clean docker-clean install

# Variables
PROJECT_NAME = energy-trading
DOCKER_IMAGE = energy-trading:latest
DOCKER_COMPOSE = docker-compose

##@ General

help: ## Display this help
	@awk 'BEGIN {FS = ":.*##"; printf "\n\033[1m%s\033[0m\n", "Usage: make <target>"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Setup

install: ## Install Python dependencies locally
	pip install -r requirements.txt

setup: ## Initial setup (copy .env, create directories)
	@echo "🔧 Setting up project..."
	@if [ ! -f .env ]; then cp .env.example .env && echo "✓ Created .env from .env.example"; fi
	@mkdir -p data/raw_data/{energy,weather,market_prices,fundamentals}
	@mkdir -p data/modified_data data/transformed_data
	@mkdir -p models/{xgboost,reg_lin,Quantile/lightgbm_quantile,tft/checkpoints,scalers}
	@mkdir -p outputs/{figures,reports,logs}
	@mkdir -p trading_system/backtests
	@mkdir -p research/{notebooks/market_research,reports}
	@echo "✓ Created directory structure"
	@echo ""
	@echo "⚠️  Don't forget to add your ENTSO-E API key in .env!"
	@echo "   Get it from: https://transparency.entsoe.eu/"

##@ Docker

build: ## Build Docker image
	@echo "🐳 Building Docker image..."
	$(DOCKER_COMPOSE) build app

build-dev: ## Build development image (with Jupyter)
	@echo "🐳 Building development image..."
	$(DOCKER_COMPOSE) build jupyter

build-all: ## Build all Docker images
	@echo "🐳 Building all images..."
	$(DOCKER_COMPOSE) build

up: ## Start all services
	@echo "🚀 Starting services..."
	$(DOCKER_COMPOSE) up -d

down: ## Stop all services
	@echo "🛑 Stopping services..."
	$(DOCKER_COMPOSE) down

restart: ## Restart all services
	@echo "🔄 Restarting services..."
	$(DOCKER_COMPOSE) down
	$(DOCKER_COMPOSE) up -d

shell: ## Open bash shell in app container
	@echo "💻 Opening shell..."
	$(DOCKER_COMPOSE) run --rm app bash

jupyter: ## Start Jupyter Lab
	@echo "📊 Starting Jupyter Lab on http://localhost:8888"
	$(DOCKER_COMPOSE) up jupyter

jupyter-stop: ## Stop Jupyter Lab
	@echo "🛑 Stopping Jupyter Lab..."
	$(DOCKER_COMPOSE) stop jupyter

##@ Training

train-xgboost: ## Train XGBoost model (daily)
	@echo "🎓 Training XGBoost..."
	$(DOCKER_COMPOSE) run --rm train python scripts/train_pipeline.py --model xgboost --frequency daily

train-lightgbm: ## Train LightGBM model (daily)
	@echo "🎓 Training LightGBM..."
	$(DOCKER_COMPOSE) run --rm train python scripts/train_pipeline.py --model lightgbm --frequency daily --lags with

train-tft: ## Train TFT model (daily, GPU if available)
	@echo "🎓 Training TFT..."
	$(DOCKER_COMPOSE) run --rm train python scripts/train_pipeline.py --model tft --frequency daily --max_epochs 30

train-ridge: ## Train Ridge regression (daily)
	@echo "🎓 Training Ridge..."
	$(DOCKER_COMPOSE) run --rm train python scripts/train_pipeline.py --model ridge --frequency daily

train-lasso: ## Train Lasso regression (daily)
	@echo "🎓 Training Lasso..."
	$(DOCKER_COMPOSE) run --rm train python scripts/train_pipeline.py --model lasso --frequency daily

train-all: ## Train all models (daily)
	@echo "🎓 Training ALL models..."
	$(DOCKER_COMPOSE) run --rm train python scripts/train_pipeline.py --model all --frequency daily

##@ Evaluation

benchmark: ## Run benchmark on all trained models
	@echo "📊 Running benchmark..."
	$(DOCKER_COMPOSE) run --rm benchmark

benchmark-fast: ## Benchmark specific models (xgboost, lightgbm)
	@echo "📊 Running fast benchmark..."
	$(DOCKER_COMPOSE) run --rm app python scripts/benchmark_models.py --frequency daily --models xgboost lightgbm

backtest: ## Run backtest example
	@echo "💹 Running backtest..."
	$(DOCKER_COMPOSE) run --rm backtest

##@ Data

data-prices: ## Collect market prices (FR, DE, ES) 2020-2024
	@echo "📥 Collecting market prices..."
	$(DOCKER_COMPOSE) run --rm data-collector python data_recuperation/data_market_prices.py --start_date 2020-01-01 --end_date 2024-12-31 --countries FR DE ES

data-fundamentals: ## Collect fundamental data (FR) 2020-2024
	@echo "📥 Collecting fundamental data..."
	$(DOCKER_COMPOSE) run --rm data-collector python data_recuperation/data_fundamentals.py --country FR --data_type all --start_date 2020-01-01 --end_date 2024-12-31

data-all: data-prices data-fundamentals ## Collect all market data

##@ Testing

test: ## Run all tests with pytest
	@echo "🧪 Running tests..."
	pytest tests/ -v --cov=. --cov-report=html --cov-report=term

test-unit: ## Run unit tests only
	@echo "🧪 Running unit tests..."
	pytest tests/ -v -k "test_" --ignore=tests/integration --ignore=tests/e2e

test-integration: ## Run integration tests
	@echo "🧪 Running integration tests..."
	pytest tests/integration/ -v

test-coverage: ## Run tests with coverage report
	@echo "🧪 Running tests with coverage..."
	pytest tests/ --cov=. --cov-report=html --cov-report=term-missing
	@echo "📊 Coverage report generated in htmlcov/index.html"

test-pipeline: ## Test training pipeline with XGBoost
	@echo "🧪 Testing pipeline..."
	$(DOCKER_COMPOSE) run --rm app python scripts/train_pipeline.py --model xgboost --frequency daily

##@ Cleanup

clean: ## Clean outputs and temporary files
	@echo "🧹 Cleaning..."
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@find . -type f -name "*.pyo" -delete 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ Cleaned Python cache files"

clean-outputs: ## Clean output directories (models, logs, figures)
	@echo "⚠️  This will delete trained models and outputs!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf outputs/* models/* && echo "✓ Cleaned outputs"; \
	else \
		echo "Cancelled"; \
	fi

clean-data: ## Clean all data (WARNING: deletes everything)
	@echo "⚠️⚠️⚠️  This will DELETE ALL DATA!"
	@read -p "Are you ABSOLUTELY sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf data/* && echo "✓ Cleaned data"; \
	else \
		echo "Cancelled"; \
	fi

docker-clean: ## Clean Docker images and containers
	@echo "🐳 Cleaning Docker..."
	$(DOCKER_COMPOSE) down -v
	docker system prune -f
	@echo "✓ Cleaned Docker resources"

##@ Development

logs: ## Show logs from all services
	$(DOCKER_COMPOSE) logs -f

logs-train: ## Show training logs
	$(DOCKER_COMPOSE) logs -f train

logs-jupyter: ## Show Jupyter logs
	$(DOCKER_COMPOSE) logs -f jupyter

ps: ## Show running containers
	$(DOCKER_COMPOSE) ps

##@ Workflows

workflow-full: ## Full workflow: setup → data → train → benchmark
	@echo "🚀 Starting FULL workflow..."
	@echo ""
	@echo "Step 1/4: Setup"
	@make setup
	@echo ""
	@echo "Step 2/4: Collect data"
	@make data-all
	@echo ""
	@echo "Step 3/4: Train models"
	@make train-all
	@echo ""
	@echo "Step 4/4: Benchmark"
	@make benchmark
	@echo ""
	@echo "✅ Full workflow completed!"

workflow-quick: ## Quick workflow: train XGBoost → benchmark
	@echo "⚡ Starting QUICK workflow..."
	@make train-xgboost
	@make benchmark-fast
	@echo "✅ Quick workflow completed!"

##@ Documentation

docs: ## Open documentation in browser
	@echo "📚 Opening documentation..."
	@if command -v open > /dev/null; then \
		open README.md; \
	elif command -v xdg-open > /dev/null; then \
		xdg-open README.md; \
	else \
		echo "README.md"; \
	fi

docs-docker: ## Open Docker documentation
	@echo "📚 Opening Docker documentation..."
	@if command -v open > /dev/null; then \
		open DOCKER.md; \
	elif command -v xdg-open > /dev/null; then \
		xdg-open DOCKER.md; \
	else \
		echo "DOCKER.md"; \
	fi

docs-models: ## Open models documentation
	@echo "📚 Opening models documentation..."
	@if command -v open > /dev/null; then \
		open MODELS.md; \
	elif command -v xdg-open > /dev/null; then \
		xdg-open MODELS.md; \
	else \
		echo "MODELS.md"; \
	fi

##@ API & Services

api: ## Start FastAPI server
	@echo "🚀 Starting FastAPI server on http://localhost:8000"
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

api-docker: ## Start FastAPI in Docker
	@echo "🚀 Starting FastAPI in Docker..."
	$(DOCKER_COMPOSE) up -d api

dashboard: ## Start Streamlit dashboard
	@echo "📊 Starting Streamlit dashboard on http://localhost:8501"
	streamlit run src/dashboard/app.py

dashboard-docker: ## Start Streamlit in Docker
	@echo "📊 Starting Streamlit in Docker..."
	$(DOCKER_COMPOSE) up -d dashboard

##@ MLflow

mlflow-ui: ## Start MLflow tracking UI
	@echo "📊 Starting MLflow UI on http://localhost:5000"
	mlflow ui --backend-store-uri file:./outputs/mlruns

mlflow-server: ## Start MLflow tracking server
	@echo "🚀 Starting MLflow server..."
	mlflow server --backend-store-uri file:./outputs/mlruns --default-artifact-root ./outputs/mlartifacts --host 0.0.0.0


airflow-init: ## Initialize Airflow
	@echo "🔧 Initializing Airflow..."
	@mkdir -p airflow/logs airflow/plugins airflow/dags
	@echo "AIRFLOW_UID=$$(id -u)" > airflow/.env
	docker-compose -f docker-compose.airflow.yml up airflow-init

airflow-up: ## Start Airflow
	@echo "🚀 Starting Airflow..."
	docker-compose -f docker-compose.airflow.yml up -d
	@echo "✓ Airflow UI: http://localhost:8080 (airflow/airflow)"

airflow-down: ## Stop Airflow
	@echo "🛑 Stopping Airflow..."
	docker-compose -f docker-compose.airflow.yml down

##@ Optimization

optimize-xgboost: ## Run Optuna hyperparameter optimization for XGBoost
	@echo "🔬 Running hyperparameter optimization for XGBoost..."
	python src/ml/optuna_tuner.py --model xgboost --n-trials 100

optimize-lightgbm: ## Run Optuna optimization for LightGBM
	@echo "🔬 Running hyperparameter optimization for LightGBM..."
	python src/ml/optuna_tuner.py --model lightgbm --n-trials 100

##@ Info

version: ## Show project version
	@echo "$(PROJECT_NAME) version 2.0.0"

status: ## Show project status
	@echo "📊 Project Status:"
	@echo ""
	@echo "Docker:"
	@$(DOCKER_COMPOSE) ps
	@echo ""
	@echo "Data directories:"
	@du -sh data/* 2>/dev/null || echo "  No data yet"
	@echo ""
	@echo "Models:"
	@ls -lh models/*/*.pkl 2>/dev/null | wc -l | xargs echo "  Trained models:"
	@echo ""
	@echo "Outputs:"
	@du -sh outputs/* 2>/dev/null || echo "  No outputs yet"
