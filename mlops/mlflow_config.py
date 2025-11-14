"""MLflow configuration for energy trading project.

This module provides centralized configuration for MLflow experiment tracking,
including tracking URI, experiment naming, and default parameters.
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# MLflow tracking configuration
MLFLOW_TRACKING_URI = f"file://{PROJECT_ROOT}/mlruns"
MLFLOW_ARTIFACT_ROOT = f"{PROJECT_ROOT}/mlartifacts"

# Experiment names for different components
EXPERIMENTS = {
    "price_forecasting": "energy-price-forecasting",
    "load_forecasting": "energy-load-forecasting",
    "renewable_forecasting": "renewable-generation-forecasting",
    "trading_strategies": "trading-strategies-backtest",
    "model_comparison": "model-performance-comparison",
}

# Default tags for all runs
DEFAULT_TAGS = {
    "project": "energy-demand-forecast",
    "framework": "quantitative-trading",
    "author": "research-team",
}

# Model registry settings
MODEL_REGISTRY_PATH = f"{PROJECT_ROOT}/models/registry"

# Tracking settings
ENABLE_AUTOLOG = True
LOG_MODELS = True
LOG_ARTIFACTS = True


def get_tracking_uri():
    """Get MLflow tracking URI.

    Returns:
        str: MLflow tracking URI.
    """
    return MLFLOW_TRACKING_URI


def get_experiment_name(category):
    """Get experiment name for a given category.

    Args:
        category: Experiment category (e.g., 'price_forecasting').

    Returns:
        str: Experiment name.
    """
    return EXPERIMENTS.get(category, "default-experiment")


def setup_mlflow_dirs():
    """Create necessary MLflow directories."""
    os.makedirs(f"{PROJECT_ROOT}/mlruns", exist_ok=True)
    os.makedirs(f"{PROJECT_ROOT}/mlartifacts", exist_ok=True)
    os.makedirs(MODEL_REGISTRY_PATH, exist_ok=True)
