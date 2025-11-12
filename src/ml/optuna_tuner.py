"""
Optuna Integration for Hyperparameter Optimization

Features:
- Automated hyperparameter tuning
- Multi-objective optimization
- Pruning for early stopping
- Integration with MLflow
"""

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from typing import Dict, Any, Callable, Optional, List
import logging
from pathlib import Path
import mlflow

logger = logging.getLogger(__name__)


class OptunaHyperparameterTuner:
    """
    Hyperparameter tuner using Optuna.

    Features:
    - Bayesian optimization
    - Early stopping with pruning
    - MLflow integration
    - Multi-objective optimization
    """

    def __init__(
        self,
        study_name: str = "energy-trading-optimization",
        storage: Optional[str] = None,
        direction: str = "minimize",
        n_trials: int = 100,
        pruner: Optional[optuna.pruners.BasePruner] = None,
        sampler: Optional[optuna.samplers.BaseSampler] = None
    ):
        """
        Initialize Optuna tuner.

        Args:
            study_name: Name of the optimization study
            storage: Database URL for storing study (optional)
            direction: 'minimize' or 'maximize'
            n_trials: Number of optimization trials
            pruner: Pruner for early stopping
            sampler: Sampling algorithm
        """
        self.study_name = study_name
        self.n_trials = n_trials
        self.direction = direction

        # Default pruner and sampler
        self.pruner = pruner or MedianPruner(n_startup_trials=10, n_warmup_steps=5)
        self.sampler = sampler or TPESampler(seed=42)

        # Storage (use SQLite if not specified)
        self.storage = storage or f"sqlite:///outputs/optuna/{study_name}.db"

        # Create study
        self.study = optuna.create_study(
            study_name=study_name,
            storage=self.storage,
            direction=direction,
            pruner=self.pruner,
            sampler=self.sampler,
            load_if_exists=True
        )

        logger.info(f"Initialized Optuna tuner: {study_name}")
        logger.info(f"  Direction: {direction}")
        logger.info(f"  Trials: {n_trials}")

    def optimize_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        multioutput: bool = True
    ) -> Dict[str, Any]:
        """
        Optimize XGBoost hyperparameters.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            multioutput: If True, use MultiOutputRegressor

        Returns:
            Best hyperparameters
        """
        from xgboost import XGBRegressor
        from sklearn.multioutput import MultiOutputRegressor

        def objective(trial):
            # Suggest hyperparameters
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'gamma': trial.suggest_float('gamma', 0.0, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                'random_state': 42
            }

            # Create model
            base_model = XGBRegressor(**params)

            if multioutput:
                model = MultiOutputRegressor(base_model)
            else:
                model = base_model

            # Train
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))

            # Report intermediate value (for pruning)
            trial.report(rmse, step=0)

            # Check if should prune
            if trial.should_prune():
                raise optuna.TrialPruned()

            return rmse

        # Run optimization
        logger.info("Starting XGBoost optimization...")

        self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        logger.info(f"Best RMSE: {self.study.best_value:.2f}")
        logger.info(f"Best params: {self.study.best_params}")

        return self.study.best_params

    def optimize_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        quantile: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Optimize LightGBM hyperparameters.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            quantile: Quantile for quantile regression (optional)

        Returns:
            Best hyperparameters
        """
        import lightgbm as lgb

        def objective(trial):
            # Suggest hyperparameters
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 12),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 20, 200),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
                'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
                'random_state': 42,
                'verbose': -1
            }

            # Add quantile objective if specified
            if quantile is not None:
                params['objective'] = 'quantile'
                params['alpha'] = quantile

            # Create model
            model = lgb.LGBMRegressor(**params)

            # Train
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))

            return rmse

        # Run optimization
        logger.info("Starting LightGBM optimization...")

        self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        logger.info(f"Best RMSE: {self.study.best_value:.2f}")
        logger.info(f"Best params: {self.study.best_params}")

        return self.study.best_params

    def optimize_ridge(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ) -> Dict[str, Any]:
        """
        Optimize Ridge regression hyperparameters.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets

        Returns:
            Best hyperparameters
        """
        from sklearn.linear_model import Ridge
        from sklearn.multioutput import MultiOutputRegressor

        def objective(trial):
            # Suggest hyperparameters
            params = {
                'alpha': trial.suggest_float('alpha', 0.001, 100.0, log=True),
                'solver': trial.suggest_categorical('solver', ['auto', 'svd', 'cholesky', 'lsqr']),
                'random_state': 42
            }

            # Create model
            model = MultiOutputRegressor(Ridge(**params))

            # Train
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))

            return rmse

        # Run optimization
        logger.info("Starting Ridge optimization...")

        self.study.optimize(objective, n_trials=self.n_trials, show_progress_bar=True)

        logger.info(f"Best RMSE: {self.study.best_value:.2f}")
        logger.info(f"Best params: {self.study.best_params}")

        return self.study.best_params

    def get_optimization_history(self) -> pd.DataFrame:
        """
        Get optimization history as DataFrame.

        Returns:
            DataFrame with trial results
        """
        df = self.study.trials_dataframe()

        logger.info(f"Retrieved {len(df)} trials")

        return df

    def plot_optimization_history(self, save_path: Optional[Path] = None):
        """
        Plot optimization history.

        Args:
            save_path: Path to save figure (optional)
        """
        import plotly

        # Optimization history
        fig1 = optuna.visualization.plot_optimization_history(self.study)

        if save_path:
            fig1.write_html(save_path / "optimization_history.html")
        else:
            fig1.show()

        # Parallel coordinate plot
        fig2 = optuna.visualization.plot_parallel_coordinate(self.study)

        if save_path:
            fig2.write_html(save_path / "parallel_coordinate.html")
        else:
            fig2.show()

        # Param importances
        try:
            fig3 = optuna.visualization.plot_param_importances(self.study)

            if save_path:
                fig3.write_html(save_path / "param_importances.html")
            else:
                fig3.show()
        except:
            logger.warning("Could not plot param importances (need > 1 trial)")

    def export_best_model_config(self, output_path: Path):
        """
        Export best hyperparameters to JSON.

        Args:
            output_path: Path to save config
        """
        import json

        config = {
            'study_name': self.study_name,
            'best_value': self.study.best_value,
            'best_params': self.study.best_params,
            'n_trials': len(self.study.trials),
            'optimization_date': str(pd.Timestamp.now())
        }

        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Exported best config to {output_path}")


class MultiObjectiveOptimizer:
    """
    Multi-objective optimization (e.g., optimize both accuracy and speed).
    """

    def __init__(self, study_name: str = "multi-objective"):
        self.study = optuna.create_study(
            study_name=study_name,
            directions=["minimize", "minimize"],  # RMSE, inference time
            load_if_exists=True
        )

        logger.info(f"Initialized multi-objective optimizer: {study_name}")

    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_trials: int = 100
    ):
        """
        Optimize for both accuracy and inference speed.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            n_trials: Number of trials
        """
        from xgboost import XGBRegressor
        from sklearn.multioutput import MultiOutputRegressor
        import time

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            }

            model = MultiOutputRegressor(XGBRegressor(**params, random_state=42))

            # Train
            model.fit(X_train, y_train)

            # Evaluate accuracy
            y_pred = model.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))

            # Evaluate speed
            start_time = time.time()
            _ = model.predict(X_val[:100])
            inference_time = (time.time() - start_time) * 1000  # ms

            return rmse, inference_time

        self.study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        logger.info("Multi-objective optimization completed")

        # Show Pareto front
        pareto_front = self.study.best_trials

        logger.info(f"Pareto front has {len(pareto_front)} solutions")

        for trial in pareto_front[:5]:  # Show top 5
            logger.info(f"  RMSE={trial.values[0]:.2f}, Time={trial.values[1]:.2f}ms")


# Example usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # Generate sample data
    from sklearn.datasets import make_regression

    X, y = make_regression(n_samples=1000, n_features=10, n_targets=2, random_state=42)

    # Split
    split = int(0.8 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # Optimize XGBoost
    tuner = OptunaHyperparameterTuner(
        study_name="xgboost_test",
        n_trials=20
    )

    best_params = tuner.optimize_xgboost(X_train, y_train, X_val, y_val)

    print(f"\nBest parameters: {best_params}")

    # Plot results
    output_path = Path("outputs/optuna")
    output_path.mkdir(parents=True, exist_ok=True)

    tuner.plot_optimization_history(save_path=output_path)
    tuner.export_best_model_config(output_path / "best_config.json")

    print("\nOptimization complete!")
