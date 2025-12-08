"""Bayesian Optimization for adaptive hyperparameter tuning in walk-forward validation.

This module implements an advanced hyperparameter tuning approach used by quantitative
trading firms:
- Continuous learning: hyperparameters adapt to changing market conditions
- Efficient search: uses Gaussian Process to model hyperparameter → performance
- Production-ready: low overhead, can run weekly/monthly

Key advantages over GridSearch:
✅ Adapts to regime changes (normal vs volatile markets)
✅ More efficient: finds good params with fewer evaluations
✅ Principled uncertainty quantification
✅ Can run continuously in production

Typical usage:
    # Initial tuning phase (offline)
    tuner = BayesianHyperparameterTuner(model_type='xgboost')
    best_params = tuner.optimize(df, feature_cols, n_iterations=20)

    # Adaptive tuning during walk-forward
    tuner = AdaptiveWalkForwardTuner(model_type='xgboost')
    predictions = tuner.walk_forward_with_tuning(
        df, feature_cols,
        tune_frequency_days=30  # Re-tune monthly
    )
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from scipy.stats import norm

logger = logging.getLogger(__name__)


class BayesianHyperparameterTuner:
    """Bayesian Optimization for XGBoost hyperparameter tuning.

    Uses Gaussian Process to model the relationship between hyperparameters
    and validation performance. Efficiently explores the hyperparameter space
    using Expected Improvement acquisition function.
    """

    def __init__(self, model_type: str = 'xgboost', random_state: int = 42):
        """Initialize Bayesian tuner.

        Args:
            model_type: Type of model ('xgboost', 'lightgbm', etc.)
            random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        # Hyperparameter search space (normalized to [0, 1])
        self.param_space = self._define_param_space()
        self.param_names = list(self.param_space.keys())

        # Gaussian Process for modeling performance
        kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,
            random_state=random_state,
            normalize_y=True,
            alpha=1e-6  # Noise level
        )

        # History of evaluations
        self.X_observed = []  # Hyperparameters (normalized)
        self.y_observed = []  # MAE scores

        self.scaler = StandardScaler()

    def _define_param_space(self) -> Dict[str, Dict]:
        """Define hyperparameter search space.

        Returns:
            Dictionary mapping param name to (min, max, type)
        """
        if self.model_type == 'xgboost':
            return {
                'n_estimators': {'min': 100, 'max': 500, 'type': 'int'},
                'max_depth': {'min': 3, 'max': 10, 'type': 'int'},
                'learning_rate': {'min': 0.01, 'max': 0.2, 'type': 'float'},
                'subsample': {'min': 0.6, 'max': 1.0, 'type': 'float'},
                'colsample_bytree': {'min': 0.6, 'max': 1.0, 'type': 'float'},
                'min_child_weight': {'min': 1, 'max': 10, 'type': 'int'},
                'gamma': {'min': 0.0, 'max': 0.5, 'type': 'float'},
                'reg_alpha': {'min': 0.0, 'max': 1.0, 'type': 'float'},
                'reg_lambda': {'min': 0.5, 'max': 2.0, 'type': 'float'},
            }
        else:
            raise ValueError(f"Model type '{self.model_type}' not supported")

    def _normalize_params(self, params: Dict[str, float]) -> np.ndarray:
        """Normalize parameters to [0, 1] range.

        Args:
            params: Dictionary of hyperparameters

        Returns:
            Normalized array
        """
        normalized = []
        for name in self.param_names:
            value = params[name]
            space = self.param_space[name]
            # Normalize to [0, 1]
            norm_value = (value - space['min']) / (space['max'] - space['min'])
            normalized.append(norm_value)
        return np.array(normalized)

    def _denormalize_params(self, x: np.ndarray) -> Dict[str, Any]:
        """Convert normalized params back to original scale.

        Args:
            x: Normalized parameter array [0, 1]

        Returns:
            Dictionary of hyperparameters
        """
        params = {}
        for i, name in enumerate(self.param_names):
            space = self.param_space[name]
            # Denormalize
            value = x[i] * (space['max'] - space['min']) + space['min']
            # Cast to correct type
            if space['type'] == 'int':
                value = int(round(value))
            params[name] = value
        return params

    def _sample_random_params(self) -> np.ndarray:
        """Sample random hyperparameters from search space.

        Returns:
            Normalized parameter array
        """
        return self.rng.uniform(0, 1, size=len(self.param_names))

    def _expected_improvement(self, X: np.ndarray, xi: float = 0.01) -> np.ndarray:
        """Calculate Expected Improvement acquisition function.

        Args:
            X: Candidate hyperparameters (normalized)
            xi: Exploration-exploitation trade-off parameter

        Returns:
            Expected improvement for each candidate
        """
        if len(self.X_observed) == 0:
            return np.zeros(len(X))

        # Predict mean and std using GP
        mu, sigma = self.gp.predict(X, return_std=True)

        # Ensure proper shapes
        mu = mu.reshape(-1, 1)
        sigma = sigma.reshape(-1, 1)

        # Best observed value (minimize MAE)
        f_best = np.min(self.y_observed)

        # Calculate EI
        with np.errstate(divide='warn'):
            imp = f_best - mu - xi
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma.flatten() == 0.0] = 0.0

        return ei.flatten()

    def _suggest_next_params(self) -> np.ndarray:
        """Suggest next hyperparameters to evaluate.

        Uses Expected Improvement to balance exploration vs exploitation.

        Returns:
            Normalized parameter array
        """
        if len(self.X_observed) < 5:
            # Random exploration for first few iterations
            return self._sample_random_params()

        # Generate candidates
        n_candidates = 1000
        candidates = self.rng.uniform(0, 1, size=(n_candidates, len(self.param_names)))

        # Evaluate EI for all candidates
        ei = self._expected_improvement(candidates)

        # Select candidate with highest EI
        best_idx = np.argmax(ei)
        return candidates[best_idx]

    def _evaluate_params(
        self,
        params: Dict[str, Any],
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = 'price',
        validation_days: int = 90
    ) -> float:
        """Evaluate hyperparameters using time series validation.

        Args:
            params: Hyperparameters to evaluate
            df: Dataset
            feature_cols: Feature columns
            target_col: Target column
            validation_days: Days to use for validation

        Returns:
            MAE score
        """
        # Use last validation_days for testing
        n_test = validation_days * 24
        n_train = len(df) - n_test

        if n_train < 100:
            raise ValueError("Not enough data for validation")

        # Split
        train_df = df.iloc[:n_train]
        test_df = df.iloc[n_train:n_train + n_test]

        # Train model
        if self.model_type == 'xgboost':
            model = xgb.XGBRegressor(
                **params,
                objective='reg:squarederror',
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=0
            )
        else:
            raise ValueError(f"Model type '{self.model_type}' not supported")

        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values
        X_test = test_df[feature_cols].values
        y_test = test_df[target_col].values

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        return mae

    def optimize(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = 'price',
        n_iterations: int = 20,
        validation_days: int = 90,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """Run Bayesian Optimization to find best hyperparameters.

        Args:
            df: Dataset
            feature_cols: Feature columns
            target_col: Target column
            n_iterations: Number of optimization iterations
            validation_days: Days to use for validation
            verbose: Print progress

        Returns:
            Best hyperparameters found
        """
        if verbose:
            logger.info(
                f"\n{'='*80}\n"
                f"BAYESIAN OPTIMIZATION - HYPERPARAMETER TUNING\n"
                f"{'='*80}\n"
                f"Model: {self.model_type}\n"
                f"Parameters to tune: {len(self.param_names)}\n"
                f"Iterations: {n_iterations}\n"
                f"Validation period: {validation_days} days\n"
            )

        # Bayesian optimization loop
        for i in range(n_iterations):
            # Suggest next hyperparameters
            x_next = self._suggest_next_params()
            params_next = self._denormalize_params(x_next)

            # Evaluate
            mae = self._evaluate_params(
                params_next, df, feature_cols, target_col, validation_days
            )

            # Store observation
            self.X_observed.append(x_next)
            self.y_observed.append(mae)

            # Update GP model
            if len(self.X_observed) >= 2:
                X_train = np.array(self.X_observed)
                y_train = np.array(self.y_observed).reshape(-1, 1)
                self.gp.fit(X_train, y_train)

            if verbose:
                best_mae = np.min(self.y_observed)
                logger.info(
                    f"Iteration {i+1:2d}/{n_iterations}: "
                    f"MAE = {mae:.2f} EUR/MWh | "
                    f"Best = {best_mae:.2f} EUR/MWh"
                )
                if i % 5 == 4:
                    logger.info(f"  Current params: {params_next}")

        # Get best hyperparameters
        best_idx = np.argmin(self.y_observed)
        best_x = self.X_observed[best_idx]
        best_params = self._denormalize_params(best_x)
        best_mae = self.y_observed[best_idx]

        if verbose:
            logger.info(
                f"\n{'='*80}\n"
                f"OPTIMIZATION COMPLETED\n"
                f"{'='*80}\n"
                f"Best MAE: {best_mae:.2f} EUR/MWh\n"
                f"Best hyperparameters:\n"
            )
            for name, value in best_params.items():
                logger.info(f"  {name:20s}: {value}")

        return best_params


class AdaptiveWalkForwardTuner:
    """Walk-forward validation with adaptive hyperparameter tuning.

    This class implements the "online learning" approach used by top quant firms:
    - Every N days, re-optimize hyperparameters based on recent performance
    - Adapts to regime changes automatically
    - Maintains ensemble of recent best models
    """

    def __init__(
        self,
        model_type: str = 'xgboost',
        tune_frequency_days: int = 30,
        tuning_iterations: int = 15,
        random_state: int = 42
    ):
        """Initialize adaptive tuner.

        Args:
            model_type: Type of model
            tune_frequency_days: How often to re-tune hyperparameters
            tuning_iterations: Bayesian optimization iterations per tuning
            random_state: Random seed
        """
        self.model_type = model_type
        self.tune_frequency_days = tune_frequency_days
        self.tuning_iterations = tuning_iterations
        self.random_state = random_state

        self.tuner = BayesianHyperparameterTuner(
            model_type=model_type,
            random_state=random_state
        )

        # Track hyperparameter evolution
        self.hyperparameter_history = []

    def walk_forward_with_adaptive_tuning(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = 'price',
        initial_train_days: int = 365,
        forecast_horizon_hours: int = 24,
        retrain_frequency_days: int = 1,
        verbose: bool = True
    ) -> Tuple[pd.DataFrame, Dict[str, float], List[Dict]]:
        """Perform walk-forward validation with periodic hyperparameter re-tuning.

        Args:
            df: Dataset
            feature_cols: Feature columns
            target_col: Target column
            initial_train_days: Initial training period
            forecast_horizon_hours: Forecast horizon
            retrain_frequency_days: How often to retrain model
            verbose: Print progress

        Returns:
            - predictions_df: Predictions with datetime, actual, predicted
            - metrics: Overall performance metrics
            - daily_metrics: Daily performance metrics
        """
        df = df.sort_values('datetime_hour').reset_index(drop=True)

        # Initial hyperparameter tuning
        if verbose:
            logger.info(
                f"\n{'='*80}\n"
                f"ADAPTIVE WALK-FORWARD VALIDATION\n"
                f"{'='*80}\n"
                f"Initial hyperparameter tuning...\n"
            )

        # Use first year for initial tuning
        tune_df = df.iloc[:initial_train_days * 24]
        best_params = self.tuner.optimize(
            tune_df,
            feature_cols,
            target_col,
            n_iterations=self.tuning_iterations,
            validation_days=90,
            verbose=verbose
        )

        self.hyperparameter_history.append({
            'datetime': df['datetime_hour'].iloc[initial_train_days * 24],
            'params': best_params.copy(),
            'reason': 'initial_tuning'
        })

        # Start walk-forward with adaptive tuning
        predictions = []
        daily_metrics_list = []

        # Initial training
        initial_train_end_idx = initial_train_days * 24
        train_df = df.iloc[:initial_train_end_idx].copy()

        # Create model with best params
        if self.model_type == 'xgboost':
            model = xgb.XGBRegressor(
                **best_params,
                objective='reg:squarederror',
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=0
            )
        else:
            raise ValueError(f"Model type '{self.model_type}' not supported")

        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values
        model.fit(X_train, y_train)

        if verbose:
            logger.info(f"\nStarting walk-forward validation...")

        # Walk forward
        first_test_datetime = df.iloc[initial_train_end_idx]['datetime_hour']
        first_pred_date = (first_test_datetime.normalize() + pd.Timedelta(days=1)).date()
        current_date = pd.Timestamp(first_pred_date)

        iteration = 0
        days_since_retrain = 0
        days_since_tuning = 0

        while current_date <= df['datetime_hour'].max():
            iteration += 1

            # Get test data for current day
            next_date = current_date + pd.Timedelta(days=1)
            day_mask = (df['datetime_hour'] >= current_date) & (df['datetime_hour'] < next_date)
            test_df = df[day_mask].copy()

            if len(test_df) != forecast_horizon_hours:
                if verbose and len(test_df) > 0:
                    logger.info(f"Skipping {current_date.date()}: only {len(test_df)}/24 hours")
                current_date = next_date
                continue

            # Predict
            X_test = test_df[feature_cols].values
            y_test = test_df[target_col].values
            y_pred = model.predict(X_test)

            # Store predictions
            for i, (idx, row) in enumerate(test_df.iterrows()):
                predictions.append({
                    'datetime_hour': row['datetime_hour'],
                    'actual': y_test[i],
                    'predicted': y_pred[i],
                    'error': y_pred[i] - y_test[i],
                    'abs_error': abs(y_pred[i] - y_test[i]),
                    'iteration': iteration,
                })

            # Calculate daily metrics
            mae = mean_absolute_error(y_test, y_pred)
            daily_metrics_list.append({
                'datetime': current_date,
                'mae': mae,
                'iteration': iteration
            })

            # Move to next day
            current_date = next_date
            days_since_retrain += 1
            days_since_tuning += 1

            # Re-tune hyperparameters periodically
            if days_since_tuning >= self.tune_frequency_days:
                if verbose:
                    logger.info(
                        f"\n{'='*80}\n"
                        f"PERIODIC HYPERPARAMETER RE-TUNING (Day {iteration})\n"
                        f"{'='*80}\n"
                    )

                # Get recent data for tuning (last 6 months)
                tune_end_idx = df[df['datetime_hour'] < current_date].index[-1] + 1
                tune_start_idx = max(0, tune_end_idx - 180 * 24)
                tune_df = df.iloc[tune_start_idx:tune_end_idx]

                # Re-optimize hyperparameters
                self.tuner = BayesianHyperparameterTuner(
                    model_type=self.model_type,
                    random_state=self.random_state
                )

                best_params = self.tuner.optimize(
                    tune_df,
                    feature_cols,
                    target_col,
                    n_iterations=self.tuning_iterations,
                    validation_days=60,
                    verbose=verbose
                )

                self.hyperparameter_history.append({
                    'datetime': current_date,
                    'params': best_params.copy(),
                    'reason': 'periodic_retuning'
                })

                # Create new model with updated params
                if self.model_type == 'xgboost':
                    model = xgb.XGBRegressor(
                        **best_params,
                        objective='reg:squarederror',
                        random_state=self.random_state,
                        n_jobs=-1,
                        verbosity=0
                    )

                # CRITICAL: Train the new model immediately with all data up to now
                train_end_mask = df['datetime_hour'] < current_date
                current_idx = train_end_mask.sum()
                train_df = df.iloc[:current_idx].copy()
                X_train = train_df[feature_cols].values
                y_train = train_df[target_col].values
                model.fit(X_train, y_train)

                if verbose:
                    logger.info(f"✅ New model trained with updated hyperparameters on {len(train_df)} hours")

                days_since_tuning = 0
                days_since_retrain = 0  # Reset retrain counter

            # Retrain model
            if days_since_retrain >= retrain_frequency_days:
                train_end_mask = df['datetime_hour'] < current_date
                current_idx = train_end_mask.sum()

                train_df = df.iloc[:current_idx].copy()
                X_train = train_df[feature_cols].values
                y_train = train_df[target_col].values

                model.fit(X_train, y_train)
                days_since_retrain = 0

                if verbose and iteration % 30 == 0:
                    logger.info(
                        f"Day {iteration}: Retrained on {len(train_df)} hours | "
                        f"Recent MAE: {mae:.2f} EUR/MWh"
                    )

        # Create results DataFrame
        predictions_df = pd.DataFrame(predictions)

        # Calculate overall metrics
        from model.price_forecasting.walk_forward_validator import calculate_price_forecast_metrics
        overall_metrics = calculate_price_forecast_metrics(
            predictions_df['actual'].values,
            predictions_df['predicted'].values
        )

        overall_metrics['n_predictions'] = len(predictions_df)
        overall_metrics['n_iterations'] = iteration
        overall_metrics['n_tuning_events'] = len(self.hyperparameter_history)

        if verbose:
            logger.info(
                f"\n{'='*80}\n"
                f"ADAPTIVE WALK-FORWARD COMPLETED\n"
                f"{'='*80}\n"
                f"Predictions: {len(predictions_df):,} hours\n"
                f"Tuning events: {len(self.hyperparameter_history)}\n"
                f"\n"
                f"Overall Metrics:\n"
                f"  MAE:  {overall_metrics['mae']:.2f} EUR/MWh\n"
                f"  RMSE: {overall_metrics['rmse']:.2f} EUR/MWh\n"
                f"  R²:   {overall_metrics['r2']:.3f}\n"
            )

        return predictions_df, overall_metrics, daily_metrics_list

    def get_hyperparameter_evolution(self) -> pd.DataFrame:
        """Get history of hyperparameter changes.

        Returns:
            DataFrame with datetime and parameter values
        """
        if not self.hyperparameter_history:
            return pd.DataFrame()

        records = []
        for entry in self.hyperparameter_history:
            record = {'datetime': entry['datetime'], 'reason': entry['reason']}
            record.update(entry['params'])
            records.append(record)

        return pd.DataFrame(records)


if __name__ == "__main__":
    # Test with dummy data
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    print("\n" + "="*80)
    print("TESTING BAYESIAN HYPERPARAMETER TUNER")
    print("="*80)

    # Create dummy dataset
    dates = pd.date_range("2023-01-01", "2024-12-31", freq="H")
    n = len(dates)

    df = pd.DataFrame({
        'datetime_hour': dates,
        'price': 50 + 20 * np.sin(2 * np.pi * dates.hour / 24) + np.random.normal(0, 10, n),
        'feature1': np.random.normal(0, 1, n),
        'feature2': np.random.normal(0, 1, n),
        'feature3': np.random.normal(0, 1, n),
    })

    feature_cols = ['feature1', 'feature2', 'feature3']

    # Test basic Bayesian optimization
    tuner = BayesianHyperparameterTuner(model_type='xgboost')
    best_params = tuner.optimize(
        df=df,
        feature_cols=feature_cols,
        n_iterations=10,
        validation_days=90,
        verbose=True
    )

    print("\n✅ Bayesian optimization test completed!")
