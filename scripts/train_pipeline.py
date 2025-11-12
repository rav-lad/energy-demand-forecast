"""
Unified Training Pipeline for All Models

This script provides a clean pipeline for training any model:
    Data → Preprocessing → Training → Evaluation → Save

Usage:
    python scripts/train_pipeline.py --model xgboost --frequency daily
    python scripts/train_pipeline.py --model lightgbm --frequency daily --lags with
    python scripts/train_pipeline.py --model tft --frequency daily --epochs 30
    python scripts/train_pipeline.py --model ridge --frequency daily
    python scripts/train_pipeline.py --model lasso --frequency daily
    python scripts/train_pipeline.py --model all --frequency daily  # Train all models

Models:
    - xgboost: XGBoost Regressor
    - lightgbm: LightGBM Quantile Regressor
    - tft: Temporal Fusion Transformer
    - ridge: Ridge Regression
    - lasso: Lasso Regression
    - all: Train all models sequentially
"""

import sys
import argparse
import logging
from pathlib import Path
import time
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TrainingPipeline:
    """Unified pipeline for model training."""

    def __init__(self, model_type, frequency='daily', **kwargs):
        """
        Initialize training pipeline.

        Args:
            model_type: Model type (xgboost, lightgbm, tft, ridge, lasso)
            frequency: daily or hourly
            **kwargs: Additional model-specific parameters
        """
        self.model_type = model_type.lower()
        self.frequency = frequency.lower()
        self.kwargs = kwargs
        self.start_time = None
        self.end_time = None
        self.metrics = {}

    def run(self):
        """Execute the full training pipeline."""
        logger.info("="*70)
        logger.info(f"TRAINING PIPELINE: {self.model_type.upper()}")
        logger.info("="*70)

        self.start_time = time.time()

        try:
            # Step 1: Validate data
            self._validate_data()

            # Step 2: Preprocess (if needed)
            self._preprocess()

            # Step 3: Train model
            self._train()

            # Step 4: Evaluate
            self._evaluate()

            # Step 5: Save results
            self._save_results()

            self.end_time = time.time()
            duration = self.end_time - self.start_time

            logger.info("="*70)
            logger.info(f"TRAINING COMPLETED SUCCESSFULLY in {duration:.2f}s")
            logger.info("="*70)

            return True

        except Exception as e:
            logger.error(f"Training failed: {str(e)}", exc_info=True)
            return False

    def _validate_data(self):
        """Validate that required data files exist."""
        logger.info("Step 1/5: Validating data...")

        data_dir = project_root / "data" / "modified_data"
        train_file = data_dir / f"train_{self.frequency}.csv"
        test_file = data_dir / f"test_{self.frequency}.csv"

        if not train_file.exists():
            raise FileNotFoundError(
                f"Training data not found: {train_file}\n"
                f"Please run data processing pipeline first."
            )

        if not test_file.exists():
            logger.warning(f"Test data not found: {test_file}")

        logger.info(f"  ✓ Training data: {train_file}")
        logger.info(f"  ✓ Test data: {test_file if test_file.exists() else 'Not found'}")

    def _preprocess(self):
        """Run preprocessing if needed."""
        logger.info("Step 2/5: Preprocessing data...")

        # Check if transformed data exists
        transformed_dir = project_root / "data" / "transformed_data"

        if self.model_type in ['xgboost', 'ridge', 'lasso']:
            suffix = 'reglin_xgboost'
        elif self.model_type == 'lightgbm':
            lags = self.kwargs.get('lags', 'with')
            suffix = f'lightgbm_{lags}lags'
        elif self.model_type == 'tft':
            suffix = 'dl'
        else:
            suffix = 'unknown'

        train_transformed = transformed_dir / f"train_{self.frequency}_{suffix}.csv"

        if not train_transformed.exists():
            logger.warning(f"Transformed data not found: {train_transformed}")
            logger.info("  Running transformation...")
            self._run_transformation()
        else:
            logger.info(f"  ✓ Using existing transformed data: {train_transformed}")

    def _run_transformation(self):
        """Run data transformation."""
        from data_processing.transformation import (
            transform_regression_and_xgb,
            transform_lightgbm_quantile,
            transform_dl
        )

        if self.model_type in ['xgboost', 'ridge', 'lasso']:
            logger.info("  Running transform_regression_and_xgb...")
            transform_regression_and_xgb(self.frequency)

        elif self.model_type == 'lightgbm':
            logger.info("  Running transform_lightgbm_quantile...")
            lags = self.kwargs.get('lags', 'with')
            transform_lightgbm_quantile(self.frequency, lags=lags)

        elif self.model_type == 'tft':
            logger.info("  Running transform_dl...")
            transform_dl(self.frequency)

    def _train(self):
        """Train the model."""
        logger.info(f"Step 3/5: Training {self.model_type} model...")

        if self.model_type == 'xgboost':
            self._train_xgboost()
        elif self.model_type == 'lightgbm':
            self._train_lightgbm()
        elif self.model_type == 'tft':
            self._train_tft()
        elif self.model_type == 'ridge':
            self._train_linear('ridge')
        elif self.model_type == 'lasso':
            self._train_linear('lasso')
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _train_xgboost(self):
        """Train XGBoost model."""
        import subprocess

        script = project_root / "model" / "xgboost" / "train_xgboost.py"
        cmd = ["python", str(script), "--frequency", self.frequency]

        logger.info(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"XGBoost training failed:\n{result.stderr}")

        logger.info(result.stdout)

    def _train_lightgbm(self):
        """Train LightGBM model."""
        import subprocess

        script = project_root / "model" / "Quantile" / "train_lightgbm_quantile.py"
        lags = self.kwargs.get('lags', 'with')

        cmd = ["python", str(script), "--frequency", self.frequency, "--lags", lags]

        logger.info(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"LightGBM training failed:\n{result.stderr}")

        logger.info(result.stdout)

    def _train_tft(self):
        """Train TFT model."""
        import subprocess

        script = project_root / "model" / "DeepLearning" / "train_tft.py"

        cmd = [
            "python", str(script),
            "--frequency", self.frequency,
            "--max_epochs", str(self.kwargs.get('max_epochs', 30)),
            "--batch_size", str(self.kwargs.get('batch_size', 128)),
            "--gpus", str(self.kwargs.get('gpus', 0))
        ]

        logger.info(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"TFT training failed:\n{result.stderr}")

        logger.info(result.stdout)

    def _train_linear(self, model_type):
        """Train linear model (Ridge or Lasso)."""
        import subprocess

        script = project_root / "model" / "reg_lin" / "train_reg_lin.py"

        cmd = [
            "python", str(script),
            "--model", model_type,
            "--frequency", self.frequency
        ]

        logger.info(f"  Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(f"{model_type.capitalize()} training failed:\n{result.stderr}")

        logger.info(result.stdout)

    def _evaluate(self):
        """Evaluate the trained model."""
        logger.info("Step 4/5: Evaluating model...")

        # TODO: Add evaluation code
        # For now, just log that evaluation would happen
        logger.info("  Evaluation metrics will be calculated from training output")

    def _save_results(self):
        """Save training results and metadata."""
        logger.info("Step 5/5: Saving results...")

        results_dir = project_root / "outputs" / "training_results"
        results_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = results_dir / f"{self.model_type}_{self.frequency}_{timestamp}.json"

        results = {
            'model_type': self.model_type,
            'frequency': self.frequency,
            'timestamp': timestamp,
            'duration_seconds': self.end_time - self.start_time if self.end_time else None,
            'parameters': self.kwargs,
            'metrics': self.metrics
        }

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info(f"  ✓ Results saved to: {results_file}")


def train_all_models(frequency='daily', **kwargs):
    """Train all models sequentially."""
    models = ['xgboost', 'lightgbm', 'ridge', 'lasso', 'tft']

    results = {}

    for model in models:
        logger.info("\n" + "="*70)
        logger.info(f"TRAINING MODEL {models.index(model) + 1}/{len(models)}: {model.upper()}")
        logger.info("="*70 + "\n")

        pipeline = TrainingPipeline(model, frequency, **kwargs)
        success = pipeline.run()
        results[model] = success

        if not success:
            logger.warning(f"Model {model} failed, continuing with next model...")

        # Short pause between models
        time.sleep(2)

    # Summary
    logger.info("\n" + "="*70)
    logger.info("TRAINING SUMMARY")
    logger.info("="*70)

    for model, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        logger.info(f"  {model:15s}: {status}")

    logger.info("="*70 + "\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Unified training pipeline for all models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train XGBoost on daily data
  python scripts/train_pipeline.py --model xgboost --frequency daily

  # Train LightGBM with lags
  python scripts/train_pipeline.py --model lightgbm --frequency daily --lags with

  # Train TFT with custom parameters
  python scripts/train_pipeline.py --model tft --frequency daily --max_epochs 50 --batch_size 64

  # Train all models
  python scripts/train_pipeline.py --model all --frequency daily

Available models: xgboost, lightgbm, tft, ridge, lasso, all
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices=['xgboost', 'lightgbm', 'tft', 'ridge', 'lasso', 'all'],
        help='Model type to train'
    )

    parser.add_argument(
        '--frequency',
        type=str,
        default='daily',
        choices=['daily', 'hourly'],
        help='Data frequency (default: daily)'
    )

    # LightGBM specific
    parser.add_argument(
        '--lags',
        type=str,
        default='with',
        choices=['with', 'without'],
        help='Use lags for LightGBM (default: with)'
    )

    # TFT specific
    parser.add_argument(
        '--max_epochs',
        type=int,
        default=30,
        help='Max epochs for TFT (default: 30)'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='Batch size for TFT (default: 128)'
    )

    parser.add_argument(
        '--gpus',
        type=int,
        default=0,
        help='Number of GPUs for TFT (default: 0, CPU only)'
    )

    args = parser.parse_args()

    # Extract kwargs
    kwargs = {
        'lags': args.lags,
        'max_epochs': args.max_epochs,
        'batch_size': args.batch_size,
        'gpus': args.gpus
    }

    # Train
    if args.model == 'all':
        results = train_all_models(args.frequency, **kwargs)
        # Exit with error if any model failed
        if not all(results.values()):
            sys.exit(1)
    else:
        pipeline = TrainingPipeline(args.model, args.frequency, **kwargs)
        success = pipeline.run()
        if not success:
            sys.exit(1)


if __name__ == '__main__':
    main()
