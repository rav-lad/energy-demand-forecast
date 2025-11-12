"""
MLflow Integration for Model Tracking and Versioning

Features:
- Experiment tracking
- Model versioning
- Metrics logging
- Artifact storage
- Model registry
"""

import mlflow
import mlflow.sklearn
import mlflow.pytorch
from mlflow.tracking import MlflowClient
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)


class MLflowTracker:
    """
    MLflow experiment tracker for energy trading models.

    Features:
    - Track experiments and runs
    - Log parameters, metrics, and artifacts
    - Version and register models
    - Compare model performance
    """

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: str = "energy-trading",
        artifact_location: Optional[str] = None
    ):
        """
        Initialize MLflow tracker.

        Args:
            tracking_uri: MLflow tracking server URI (default: local)
            experiment_name: Name of the experiment
            artifact_location: Path for storing artifacts
        """
        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # Use local directory
            mlflow.set_tracking_uri("file:./outputs/mlruns")

        # Set experiment
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)

        # Get experiment info
        self.experiment = mlflow.get_experiment_by_name(experiment_name)
        self.client = MlflowClient()

        logger.info(f"MLflow tracker initialized")
        logger.info(f"  Experiment: {experiment_name}")
        logger.info(f"  Tracking URI: {mlflow.get_tracking_uri()}")

        self.current_run = None

    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """
        Start a new MLflow run.

        Args:
            run_name: Name for this run
            tags: Dictionary of tags to add
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = run_name or f"run_{timestamp}"

        self.current_run = mlflow.start_run(run_name=run_name, tags=tags or {})

        logger.info(f"Started MLflow run: {run_name}")
        logger.info(f"  Run ID: {self.current_run.info.run_id}")

        return self.current_run

    def end_run(self):
        """End the current MLflow run."""
        if self.current_run:
            mlflow.end_run()
            logger.info("Ended MLflow run")
            self.current_run = None

    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters.

        Args:
            params: Dictionary of parameters
        """
        mlflow.log_params(params)
        logger.debug(f"Logged {len(params)} parameters")

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics.

        Args:
            metrics: Dictionary of metrics
            step: Step number (for time series)
        """
        mlflow.log_metrics(metrics, step=step)
        logger.debug(f"Logged {len(metrics)} metrics")

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """
        Log a single metric.

        Args:
            key: Metric name
            value: Metric value
            step: Step number
        """
        mlflow.log_metric(key, value, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log an artifact (file).

        Args:
            local_path: Path to local file
            artifact_path: Destination path in artifact store
        """
        mlflow.log_artifact(local_path, artifact_path)
        logger.debug(f"Logged artifact: {local_path}")

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None,
        **kwargs
    ):
        """
        Log a model.

        Args:
            model: Model object
            artifact_path: Path in artifact store
            registered_model_name: Name for model registry
            **kwargs: Additional arguments for log_model
        """
        # Determine model type and log accordingly
        if hasattr(model, 'fit') and hasattr(model, 'predict'):
            # Scikit-learn style model
            mlflow.sklearn.log_model(
                model,
                artifact_path,
                registered_model_name=registered_model_name,
                **kwargs
            )
        else:
            # Generic pickle
            mlflow.log_artifact(str(model), artifact_path)

        logger.info(f"Logged model to {artifact_path}")

        if registered_model_name:
            logger.info(f"Registered model: {registered_model_name}")

    def log_figure(self, figure, filename: str):
        """
        Log a matplotlib figure.

        Args:
            figure: Matplotlib figure
            filename: Filename for the figure
        """
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / filename
            figure.savefig(filepath, dpi=150, bbox_inches='tight')
            mlflow.log_artifact(str(filepath))

        logger.debug(f"Logged figure: {filename}")

    def log_dataframe(self, df: pd.DataFrame, filename: str):
        """
        Log a pandas DataFrame as CSV.

        Args:
            df: DataFrame to log
            filename: Filename for the CSV
        """
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / filename
            df.to_csv(filepath, index=False)
            mlflow.log_artifact(str(filepath))

        logger.debug(f"Logged DataFrame: {filename}")

    def log_dict(self, data: Dict, filename: str):
        """
        Log a dictionary as JSON.

        Args:
            data: Dictionary to log
            filename: Filename for the JSON
        """
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / filename
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            mlflow.log_artifact(str(filepath))

        logger.debug(f"Logged dict: {filename}")

    def log_training_session(
        self,
        model_name: str,
        model: Any,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        artifacts: Optional[Dict[str, str]] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Log a complete training session.

        Args:
            model_name: Name of the model
            model: Trained model object
            params: Training parameters
            metrics: Evaluation metrics
            artifacts: Dictionary of artifact paths
            tags: Tags for the run

        Returns:
            Run ID
        """
        # Prepare tags
        run_tags = {
            'model_type': model_name,
            'framework': 'sklearn',  # Default
            **(tags or {})
        }

        # Start run
        run = self.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}", tags=run_tags)

        try:
            # Log parameters
            self.log_params(params)

            # Log metrics
            self.log_metrics(metrics)

            # Log model
            self.log_model(model, registered_model_name=f"{self.experiment_name}_{model_name}")

            # Log artifacts
            if artifacts:
                for name, path in artifacts.items():
                    if Path(path).exists():
                        self.log_artifact(path)

            logger.info(f"Logged complete training session for {model_name}")

        finally:
            self.end_run()

        return run.info.run_id

    def compare_models(self, run_ids: List[str]) -> pd.DataFrame:
        """
        Compare multiple model runs.

        Args:
            run_ids: List of run IDs to compare

        Returns:
            DataFrame with comparison
        """
        runs_data = []

        for run_id in run_ids:
            run = self.client.get_run(run_id)

            run_info = {
                'run_id': run_id,
                'run_name': run.data.tags.get('mlflow.runName', 'N/A'),
                'model_type': run.data.tags.get('model_type', 'N/A'),
                'start_time': pd.to_datetime(run.info.start_time, unit='ms'),
                **run.data.params,
                **run.data.metrics
            }

            runs_data.append(run_info)

        comparison_df = pd.DataFrame(runs_data)

        logger.info(f"Compared {len(run_ids)} runs")

        return comparison_df

    def get_best_run(self, metric_name: str, ascending: bool = True) -> Dict:
        """
        Get the best run based on a metric.

        Args:
            metric_name: Metric to optimize
            ascending: If True, lower is better

        Returns:
            Dictionary with best run info
        """
        experiment_id = self.experiment.experiment_id

        runs = self.client.search_runs(
            experiment_ids=[experiment_id],
            order_by=[f"metrics.{metric_name} {'ASC' if ascending else 'DESC'}"],
            max_results=1
        )

        if not runs:
            logger.warning("No runs found")
            return {}

        best_run = runs[0]

        best_info = {
            'run_id': best_run.info.run_id,
            'run_name': best_run.data.tags.get('mlflow.runName', 'N/A'),
            'model_type': best_run.data.tags.get('model_type', 'N/A'),
            f'best_{metric_name}': best_run.data.metrics.get(metric_name),
            'params': best_run.data.params,
            'metrics': best_run.data.metrics
        }

        logger.info(f"Best run by {metric_name}: {best_info['run_name']}")
        logger.info(f"  {metric_name}: {best_info[f'best_{metric_name}']:.4f}")

        return best_info

    def load_model(self, run_id: str, artifact_path: str = "model") -> Any:
        """
        Load a model from a run.

        Args:
            run_id: Run ID
            artifact_path: Path to model artifact

        Returns:
            Loaded model
        """
        model_uri = f"runs:/{run_id}/{artifact_path}"
        model = mlflow.sklearn.load_model(model_uri)

        logger.info(f"Loaded model from run {run_id}")

        return model

    def register_model(
        self,
        run_id: str,
        model_name: str,
        artifact_path: str = "model",
        description: Optional[str] = None
    ) -> str:
        """
        Register a model in the model registry.

        Args:
            run_id: Run ID
            model_name: Name for the registered model
            artifact_path: Path to model artifact
            description: Model description

        Returns:
            Model version
        """
        model_uri = f"runs:/{run_id}/{artifact_path}"

        result = mlflow.register_model(model_uri, model_name)

        # Add description if provided
        if description:
            self.client.update_model_version(
                name=model_name,
                version=result.version,
                description=description
            )

        logger.info(f"Registered model: {model_name} (version {result.version})")

        return result.version


# Context manager for convenient run tracking
class mlflow_run:
    """Context manager for MLflow runs."""

    def __init__(self, tracker: MLflowTracker, run_name: Optional[str] = None, tags: Optional[Dict] = None):
        self.tracker = tracker
        self.run_name = run_name
        self.tags = tags

    def __enter__(self):
        self.tracker.start_run(run_name=self.run_name, tags=self.tags)
        return self.tracker

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tracker.end_run()


# Example usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Initialize tracker
    tracker = MLflowTracker(experiment_name="energy-trading-demo")

    # Example: Log a training session
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score

    # Generate sample data
    X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Log with context manager
    with mlflow_run(tracker, run_name="rf_demo"):
        tracker.log_params({
            'n_estimators': 100,
            'random_state': 42,
            'max_depth': None
        })

        tracker.log_metrics({
            'rmse': rmse,
            'r2': r2
        })

        tracker.log_model(model, registered_model_name="energy_demo_model")

    print(f"\nTraining logged successfully!")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
