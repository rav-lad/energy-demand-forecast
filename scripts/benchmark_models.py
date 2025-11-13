"""
Model Benchmark Script

Compares performance of all trained models on the same test set.

Metrics:
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
  - MAPE (Mean Absolute Percentage Error)
  - R² Score
  - Training time
  - Inference time
  - Memory usage

Usage:
    python scripts/benchmark_models.py --frequency daily
    python scripts/benchmark_models.py --frequency daily --models xgboost lightgbm tft
    python scripts/benchmark_models.py --frequency daily --output results.csv
"""

import argparse
import json
import logging
import sys
import time
import traceback
import tracemalloc
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ModelBenchmark:
    """Benchmark multiple models on the same test set."""

    def __init__(self, frequency="daily", models=None):
        """
        Initialize benchmark.

        Args:
            frequency: daily or hourly
            models: List of model names to benchmark (None = all)
        """
        self.frequency = frequency
        self.all_models = ["xgboost", "lightgbm", "tft", "ridge", "lasso"]
        self.models = models if models else self.all_models
        self.results = []

        # Load test data
        self.test_data = self._load_test_data()

    def _load_test_data(self):
        """Load test data."""
        logger.info("Loading test data...")

        test_file = project_root / "data" / "modified_data" / f"test_{self.frequency}.csv"

        if not test_file.exists():
            raise FileNotFoundError(f"Test data not found: {test_file}")

        df = pd.read_csv(test_file)
        logger.info(f"  Loaded {len(df)} test samples")

        return df

    def run(self):
        """Run benchmark for all models."""
        logger.info("=" * 70)
        logger.info("MODEL BENCHMARK")
        logger.info("=" * 70)
        logger.info(f"Frequency: {self.frequency}")
        logger.info(f"Models: {', '.join(self.models)}")
        logger.info(f"Test samples: {len(self.test_data)}")
        logger.info("=" * 70 + "\n")

        for model_name in self.models:
            logger.info(f"\n{'='*70}")
            logger.info(f"Benchmarking: {model_name.upper()}")
            logger.info(f"{'='*70}\n")

            try:
                result = self._benchmark_model(model_name)
                self.results.append(result)
                self._print_result(result)

            except Exception as e:
                logger.error(f"Failed to benchmark {model_name}: {str(e)}")
                logger.error(traceback.format_exc())
                continue

        # Generate summary
        self._print_summary()

        return self.results

    def _benchmark_model(self, model_name):
        """Benchmark a single model."""
        result = {
            "model": model_name,
            "frequency": self.frequency,
            "timestamp": datetime.now().isoformat(),
        }

        # Start memory tracking
        tracemalloc.start()
        start_mem = tracemalloc.get_traced_memory()[0] / 1024 / 1024  # MB

        # Load model
        load_start = time.time()
        model, scaler = self._load_model(model_name)
        load_time = time.time() - load_start
        result["load_time_sec"] = load_time

        # Prepare features
        prep_start = time.time()
        X_test, y_test = self._prepare_features(model_name)
        prep_time = time.time() - prep_start
        result["prep_time_sec"] = prep_time

        # Make predictions
        inference_start = time.time()
        y_pred = self._predict(model, X_test, model_name)
        inference_time = time.time() - inference_start
        result["inference_time_sec"] = inference_time
        result["inference_time_per_sample_ms"] = (inference_time / len(X_test)) * 1000

        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred)
        result.update(metrics)

        # Memory usage
        current_mem = tracemalloc.get_traced_memory()[0] / 1024 / 1024  # MB
        tracemalloc.stop()
        result["memory_usage_mb"] = current_mem - start_mem

        return result

    def _load_model(self, model_name):
        """Load trained model."""
        import joblib

        models_dir = project_root / "models"

        if model_name == "xgboost":
            model_file = models_dir / "xgboost" / f"xgb_{self.frequency}.pkl"
            scaler_file = models_dir / "scalers" / f"scaler_{self.frequency}_reglin_xgboost.pkl"

        elif model_name == "lightgbm":
            # Load all 6 quantile models
            models = {}
            for target in ["conso_elec_mw", "conso_gaz_mw"]:
                for q in [5, 50, 95]:
                    key = f"{target}_q{q}"
                    model_file = (
                        models_dir
                        / "Quantile"
                        / "lightgbm_quantile"
                        / f"{target}_q{q}_{self.frequency}_withoutlags.pkl"
                    )
                    if model_file.exists():
                        models[key] = joblib.load(model_file)
            model = models
            scaler = None

        elif model_name in ["ridge", "lasso"]:
            model_file = models_dir / "reg_lin" / f"{model_name}_classic_{self.frequency}.pkl"
            scaler_file = models_dir / "scalers" / f"scaler_{self.frequency}_reglin_xgboost.pkl"

        elif model_name == "tft":
            # TFT requires different loading
            import torch
            from pytorch_forecasting import TemporalFusionTransformer

            checkpoint_file = models_dir / "tft" / "checkpoints" / "best_tft.ckpt"
            if not checkpoint_file.exists():
                raise FileNotFoundError(f"TFT checkpoint not found: {checkpoint_file}")

            model = TemporalFusionTransformer.load_from_checkpoint(str(checkpoint_file))
            scaler = None

        else:
            raise ValueError(f"Unknown model: {model_name}")

        if model_name in ["xgboost", "ridge", "lasso"]:
            if not model_file.exists():
                raise FileNotFoundError(f"Model file not found: {model_file}")
            model = joblib.load(model_file)

            if scaler_file.exists():
                scaler = joblib.load(scaler_file)
            else:
                logger.warning(f"Scaler not found: {scaler_file}")
                scaler = None

        return model, scaler

    def _prepare_features(self, model_name):
        """Prepare features for prediction."""
        # This is a simplified version - in production, use actual transformation pipelines

        if model_name in ["xgboost", "ridge", "lasso"]:
            # Use transformed data
            transformed_file = (
                project_root
                / "data"
                / "transformed_data"
                / f"test_{self.frequency}_reglin_xgboost.csv"
            )

            if transformed_file.exists():
                df = pd.read_csv(transformed_file)
            else:
                logger.warning(f"Transformed test data not found, using raw data")
                df = self.test_data

            # Extract features and targets
            y_cols = ["conso_elec_mw", "conso_gaz_mw"]
            X = df.drop(columns=y_cols + ["date", "insee_region"], errors="ignore")
            y = df[y_cols]

        elif model_name == "lightgbm":
            # Use lightgbm transformed data
            transformed_file = (
                project_root
                / "data"
                / "transformed_data"
                / f"test_{self.frequency}_lightgbm_withoutlags.csv"
            )

            if transformed_file.exists():
                df = pd.read_csv(transformed_file)
            else:
                df = self.test_data

            y_cols = ["conso_elec_mw", "conso_gaz_mw"]
            X = df.drop(columns=y_cols + ["date", "insee_region"], errors="ignore")
            y = df[y_cols]

        elif model_name == "tft":
            # TFT requires special dataset format
            # For now, return dummy data (implement proper TFT prediction later)
            logger.warning("TFT prediction not fully implemented in benchmark")
            X = self.test_data
            y = self.test_data[["conso_elec_mw", "conso_gaz_mw"]]

        else:
            raise ValueError(f"Unknown model: {model_name}")

        return X, y

    def _predict(self, model, X, model_name):
        """Make predictions."""
        if model_name == "lightgbm":
            # Predict with all quantile models
            predictions = np.zeros((len(X), 6))  # 2 targets × 3 quantiles

            targets = ["conso_elec_mw", "conso_gaz_mw"]
            quantiles = [5, 50, 95]

            col_idx = 0
            for target in targets:
                for q in quantiles:
                    key = f"{target}_q{q}"
                    if key in model:
                        predictions[:, col_idx] = model[key].predict(X)
                    col_idx += 1

            # For benchmark, use median predictions
            y_pred = predictions[:, [1, 4]]  # q50 for elec and gaz

        elif model_name == "tft":
            # TFT prediction (simplified)
            logger.warning("Using dummy predictions for TFT (not implemented)")
            y_pred = np.zeros((len(X), 2))

        else:
            # Standard prediction
            y_pred = model.predict(X)

        return y_pred

    def _calculate_metrics(self, y_true, y_pred):
        """Calculate performance metrics."""
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        # Ensure arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        # Handle NaN
        mask = ~(np.isnan(y_true).any(axis=1) | np.isnan(y_pred).any(axis=1))
        y_true = y_true[mask]
        y_pred = y_pred[mask]

        metrics = {}

        # Overall metrics
        metrics["rmse_overall"] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics["mae_overall"] = mean_absolute_error(y_true, y_pred)
        metrics["r2_overall"] = r2_score(y_true, y_pred)

        # MAPE (avoiding division by zero)
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
        metrics["mape_overall"] = mape

        # Per-target metrics (assuming 2 targets: elec, gaz)
        if y_true.shape[1] == 2:
            metrics["rmse_elec"] = np.sqrt(mean_squared_error(y_true[:, 0], y_pred[:, 0]))
            metrics["rmse_gaz"] = np.sqrt(mean_squared_error(y_true[:, 1], y_pred[:, 1]))
            metrics["mae_elec"] = mean_absolute_error(y_true[:, 0], y_pred[:, 0])
            metrics["mae_gaz"] = mean_absolute_error(y_true[:, 1], y_pred[:, 1])
            metrics["r2_elec"] = r2_score(y_true[:, 0], y_pred[:, 0])
            metrics["r2_gaz"] = r2_score(y_true[:, 1], y_pred[:, 1])

        return metrics

    def _print_result(self, result):
        """Print benchmark result for one model."""
        logger.info(f"\n{result['model'].upper()} Results:")
        logger.info(f"  RMSE Elec:     {result.get('rmse_elec', 0):.2f} MW")
        logger.info(f"  RMSE Gaz:      {result.get('rmse_gaz', 0):.2f} MW")
        logger.info(f"  R² Elec:       {result.get('r2_elec', 0):.4f}")
        logger.info(f"  R² Gaz:        {result.get('r2_gaz', 0):.4f}")
        logger.info(f"  Inference:     {result['inference_time_sec']:.3f}s")
        logger.info(f"  Per sample:    {result['inference_time_per_sample_ms']:.3f}ms")
        logger.info(f"  Memory:        {result['memory_usage_mb']:.2f} MB")

    def _print_summary(self):
        """Print summary table."""
        if not self.results:
            logger.warning("No results to summarize")
            return

        logger.info("\n" + "=" * 70)
        logger.info("BENCHMARK SUMMARY")
        logger.info("=" * 70 + "\n")

        # Create DataFrame
        df = pd.DataFrame(self.results)

        # Sort by RMSE (electricity)
        if "rmse_elec" in df.columns:
            df = df.sort_values("rmse_elec")

        # Print table
        logger.info("Performance Ranking (by RMSE Electricity):\n")

        print(
            f"{'Model':<15} {'RMSE Elec':<12} {'RMSE Gaz':<12} {'R² Elec':<10} {'Inference (s)':<15}"
        )
        print("-" * 70)

        for _, row in df.iterrows():
            model = row["model"]
            rmse_elec = row.get("rmse_elec", float("nan"))
            rmse_gaz = row.get("rmse_gaz", float("nan"))
            r2_elec = row.get("r2_elec", float("nan"))
            inference = row.get("inference_time_sec", float("nan"))

            print(
                f"{model:<15} {rmse_elec:<12.2f} {rmse_gaz:<12.2f} {r2_elec:<10.4f} {inference:<15.3f}"
            )

        print("\n" + "=" * 70 + "\n")

        # Best model
        best_model = df.iloc[0]["model"]
        logger.info(f"🏆 Best Model: {best_model.upper()}")
        logger.info(f"   RMSE Electricity: {df.iloc[0].get('rmse_elec', 0):.2f} MW")
        logger.info(f"   R² Score: {df.iloc[0].get('r2_elec', 0):.4f}")
        logger.info("=" * 70 + "\n")

    def save_results(self, output_file):
        """Save results to CSV."""
        if not self.results:
            logger.warning("No results to save")
            return

        df = pd.DataFrame(self.results)

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False)
        logger.info(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark all models on test set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Benchmark all models
  python scripts/benchmark_models.py --frequency daily

  # Benchmark specific models
  python scripts/benchmark_models.py --frequency daily --models xgboost lightgbm

  # Save results
  python scripts/benchmark_models.py --frequency daily --output results.csv
        """,
    )

    parser.add_argument(
        "--frequency",
        type=str,
        default="daily",
        choices=["daily", "hourly"],
        help="Data frequency (default: daily)",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=["xgboost", "lightgbm", "tft", "ridge", "lasso"],
        help="Models to benchmark (default: all)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/benchmark_results.csv",
        help="Output CSV file (default: outputs/benchmark_results.csv)",
    )

    args = parser.parse_args()

    # Run benchmark
    benchmark = ModelBenchmark(args.frequency, args.models)
    results = benchmark.run()

    # Save results
    benchmark.save_results(args.output)


if __name__ == "__main__":
    main()
