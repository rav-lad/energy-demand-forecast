"""MLflow utilities for experiment tracking and model management.

This module provides convenient wrapper functions for MLflow operations,
including experiment tracking, model logging, and metric visualization.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

from mlops.mlflow_config import (
    DEFAULT_TAGS,
    ENABLE_AUTOLOG,
    get_experiment_name,
    get_tracking_uri,
    setup_mlflow_dirs,
)


class MLflowTracker:
    """Wrapper class for MLflow experiment tracking.

    This class provides a convenient interface for tracking machine learning
    experiments, logging parameters, metrics, and artifacts.

    Attributes:
        experiment_name: Name of the MLflow experiment.
        run_name: Optional name for the current run.
        tags: Additional tags for the experiment run.
    """

    def __init__(
        self,
        experiment_category: str,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        """Initialize MLflow tracker.

        Args:
            experiment_category: Category of experiment (e.g., 'price_forecasting').
            run_name: Optional name for this run.
            tags: Additional tags to add to default tags.
        """
        setup_mlflow_dirs()
        mlflow.set_tracking_uri(get_tracking_uri())

        self.experiment_name = get_experiment_name(experiment_category)
        self.run_name = run_name
        self.tags = {**DEFAULT_TAGS, **(tags or {})}

        # Set or create experiment
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            mlflow.create_experiment(self.experiment_name)

        mlflow.set_experiment(self.experiment_name)

        self.client = MlflowClient()
        self.run = None

    def __enter__(self):
        """Start MLflow run context."""
        self.run = mlflow.start_run(run_name=self.run_name, tags=self.tags)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End MLflow run context."""
        mlflow.end_run()

    def log_params(self, params: Dict[str, Any]):
        """Log parameters to MLflow.

        Args:
            params: Dictionary of parameter names and values.
        """
        mlflow.log_params(params)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to MLflow.

        Args:
            metrics: Dictionary of metric names and values.
            step: Optional step number for time-series metrics.
        """
        mlflow.log_metrics(metrics, step=step)

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a single metric to MLflow.

        Args:
            key: Metric name.
            value: Metric value.
            step: Optional step number.
        """
        mlflow.log_metric(key, value, step=step)

    def log_model(
        self, model, artifact_path: str, registered_model_name: Optional[str] = None
    ):
        """Log model to MLflow.

        Args:
            model: Trained model object.
            artifact_path: Path within run artifacts to save model.
            registered_model_name: Optional name for model registry.
        """
        mlflow.sklearn.log_model(
            model, artifact_path, registered_model_name=registered_model_name
        )

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log artifact file to MLflow.

        Args:
            local_path: Local path to artifact file.
            artifact_path: Optional subdirectory in artifact store.
        """
        mlflow.log_artifact(local_path, artifact_path)

    def log_dict(self, dictionary: Dict, filename: str):
        """Log dictionary as JSON artifact.

        Args:
            dictionary: Dictionary to log.
            filename: Name of JSON file.
        """
        mlflow.log_dict(dictionary, filename)

    def log_figure(self, figure, filename: str):
        """Log matplotlib figure as artifact.

        Args:
            figure: Matplotlib figure object.
            filename: Name of image file (should end in .png).
        """
        mlflow.log_figure(figure, filename)

    def log_dataframe(self, df: pd.DataFrame, filename: str):
        """Log pandas DataFrame as CSV artifact.

        Args:
            df: DataFrame to log.
            filename: Name of CSV file.
        """
        temp_path = f"/tmp/{filename}"
        df.to_csv(temp_path, index=False)
        mlflow.log_artifact(temp_path)
        os.remove(temp_path)

    def log_forecast_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, prefix: str = ""
    ):
        """Log comprehensive forecasting metrics.

        Args:
            y_true: True values.
            y_pred: Predicted values.
            prefix: Optional prefix for metric names (e.g., 'train_', 'test_').
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        metrics = {
            f"{prefix}rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            f"{prefix}mae": mean_absolute_error(y_true, y_pred),
            f"{prefix}r2": r2_score(y_true, y_pred),
            f"{prefix}mape": np.mean(
                np.abs((y_true - y_pred) / (y_true + 1e-8))
            ) * 100,
        }

        # Directional accuracy
        if len(y_true) > 1:
            y_true_diff = np.diff(y_true)
            y_pred_diff = np.diff(y_pred)
            metrics[f"{prefix}directional_accuracy"] = (
                np.mean(np.sign(y_true_diff) == np.sign(y_pred_diff)) * 100
            )

        self.log_metrics(metrics)
        return metrics

    def log_trading_metrics(self, backtest_results: Dict[str, float], prefix: str = ""):
        """Log trading strategy performance metrics.

        Args:
            backtest_results: Dictionary of backtest metrics.
            prefix: Optional prefix for metric names.
        """
        metrics = {}
        for key, value in backtest_results.items():
            if isinstance(value, (int, float)):
                metrics[f"{prefix}{key}"] = value

        self.log_metrics(metrics)
        return metrics


def log_model_comparison(
    models_results: Dict[str, Dict[str, float]], experiment_category: str = "model_comparison"
):
    """Log and compare multiple models.

    Args:
        models_results: Dictionary mapping model names to their metrics.
        experiment_category: Experiment category name.

    Returns:
        pd.DataFrame: Comparison table of all models.
    """
    comparison_data = []

    for model_name, metrics in models_results.items():
        with MLflowTracker(experiment_category, run_name=model_name) as tracker:
            tracker.log_metrics(metrics)
            comparison_data.append({"model": model_name, **metrics})

    comparison_df = pd.DataFrame(comparison_data)
    return comparison_df


def get_best_run(experiment_name: str, metric: str, ascending: bool = False):
    """Get the best run from an experiment based on a metric.

    Args:
        experiment_name: Name of the experiment.
        metric: Metric to optimize.
        ascending: If True, lower is better. If False, higher is better.

    Returns:
        mlflow.entities.Run: Best run object.
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
        max_results=1,
    )

    if not runs:
        raise ValueError(f"No runs found in experiment '{experiment_name}'")

    return runs[0]


def load_best_model(experiment_name: str, metric: str, ascending: bool = False):
    """Load the best model from an experiment.

    Args:
        experiment_name: Name of the experiment.
        metric: Metric to optimize.
        ascending: If True, lower is better. If False, higher is better.

    Returns:
        Loaded model object.
    """
    best_run = get_best_run(experiment_name, metric, ascending)
    model_uri = f"runs:/{best_run.info.run_id}/model"
    return mlflow.sklearn.load_model(model_uri)


def create_experiment_summary(experiment_name: str) -> pd.DataFrame:
    """Create a summary DataFrame of all runs in an experiment.

    Args:
        experiment_name: Name of the experiment.

    Returns:
        pd.DataFrame: Summary of all runs with metrics and parameters.
    """
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found")

    runs = client.search_runs(experiment_ids=[experiment.experiment_id])

    data = []
    for run in runs:
        row = {
            "run_id": run.info.run_id,
            "run_name": run.data.tags.get("mlflow.runName", ""),
            "start_time": pd.to_datetime(run.info.start_time, unit="ms"),
            "status": run.info.status,
        }
        row.update(run.data.metrics)
        row.update(run.data.params)
        data.append(row)

    return pd.DataFrame(data)
