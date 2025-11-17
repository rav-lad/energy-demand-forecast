"""
Drift Detection for Production Monitoring

Detects distribution shifts between training data and production data
to trigger model retraining when needed.

Usage:
    detector = DriftDetector(reference_data=train_df)
    report = detector.detect_drift(current_data=prod_df)
    if report.has_drift:
        logger.warning(f"Drift detected in {report.drifted_features}")
        trigger_retraining()
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class DriftReport:
    """Report containing drift detection results."""

    has_drift: bool
    drifted_features: List[str]
    drift_scores: Dict[str, float]
    p_values: Dict[str, float]
    threshold: float
    timestamp: str

    def __str__(self) -> str:
        """String representation of drift report."""
        if not self.has_drift:
            return f"✅ No drift detected (threshold={self.threshold})"

        return (
            f"⚠️ DRIFT DETECTED in {len(self.drifted_features)} features\n"
            f"Features: {', '.join(self.drifted_features)}\n"
            f"Timestamp: {self.timestamp}"
        )


class DriftDetector:
    """
    Statistical drift detector using Kolmogorov-Smirnov test.

    Features:
    - KS test for numerical features
    - Chi-square test for categorical features
    - Population Stability Index (PSI) calculation
    - Automatic alert generation
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        features: Optional[List[str]] = None,
        threshold: float = 0.05,
        categorical_features: Optional[List[str]] = None,
    ):
        """
        Initialize drift detector.

        Args:
            reference_data: Training data as reference
            features: List of features to monitor (None = all numeric)
            threshold: P-value threshold for drift detection (default: 0.05)
            categorical_features: List of categorical feature names
        """
        self.reference_data = reference_data.copy()
        self.threshold = threshold
        self.categorical_features = categorical_features or []

        # Auto-detect features to monitor
        if features is None:
            self.features = [
                col
                for col in reference_data.select_dtypes(include=[np.number]).columns
                if col not in ["datetime_hour", "date"]
            ]
        else:
            self.features = features

        logger.info(f"Initialized DriftDetector with {len(self.features)} features")

    def detect_drift(
        self, current_data: pd.DataFrame, save_report: bool = True
    ) -> DriftReport:
        """
        Detect drift between reference and current data.

        Args:
            current_data: Current/production data
            save_report: Whether to save HTML report

        Returns:
            DriftReport with detection results
        """
        drifted_features = []
        drift_scores = {}
        p_values = {}

        for feature in self.features:
            if feature not in current_data.columns:
                logger.warning(f"Feature {feature} not in current data, skipping")
                continue

            # Get reference and current distributions
            ref_values = self.reference_data[feature].dropna()
            curr_values = current_data[feature].dropna()

            if len(ref_values) == 0 or len(curr_values) == 0:
                logger.warning(f"Empty data for feature {feature}, skipping")
                continue

            # Run statistical test
            if feature in self.categorical_features:
                statistic, p_value = self._chi_square_test(ref_values, curr_values)
            else:
                statistic, p_value = self._ks_test(ref_values, curr_values)

            drift_scores[feature] = statistic
            p_values[feature] = p_value

            # Check if drift detected
            if p_value < self.threshold:
                drifted_features.append(feature)
                logger.warning(
                    f"Drift detected in '{feature}': "
                    f"statistic={statistic:.4f}, p-value={p_value:.4f}"
                )

        # Create report
        report = DriftReport(
            has_drift=len(drifted_features) > 0,
            drifted_features=drifted_features,
            drift_scores=drift_scores,
            p_values=p_values,
            threshold=self.threshold,
            timestamp=pd.Timestamp.now().isoformat(),
        )

        logger.info(str(report))

        # Save detailed report if requested
        if save_report:
            self._save_detailed_report(current_data, report)

        return report

    def _ks_test(
        self, ref_values: pd.Series, curr_values: pd.Series
    ) -> Tuple[float, float]:
        """
        Kolmogorov-Smirnov test for numerical features.

        Returns:
            (statistic, p_value)
        """
        statistic, p_value = stats.ks_2samp(ref_values, curr_values)
        return statistic, p_value

    def _chi_square_test(
        self, ref_values: pd.Series, curr_values: pd.Series
    ) -> Tuple[float, float]:
        """
        Chi-square test for categorical features.

        Returns:
            (statistic, p_value)
        """
        # Get unique categories
        categories = list(set(ref_values.unique()) | set(curr_values.unique()))

        # Count frequencies
        ref_counts = ref_values.value_counts().reindex(categories, fill_value=0)
        curr_counts = curr_values.value_counts().reindex(categories, fill_value=0)

        # Normalize to probabilities
        ref_freq = ref_counts / ref_counts.sum()
        curr_freq = curr_counts / curr_counts.sum()

        # Expected frequencies (assuming no drift)
        total = ref_counts.sum() + curr_counts.sum()
        expected = (ref_counts + curr_counts) / 2

        # Chi-square test
        observed = np.concatenate([ref_counts, curr_counts])
        expected_full = np.concatenate([expected, expected])

        # Avoid division by zero
        expected_full = np.where(expected_full == 0, 1e-10, expected_full)

        chi2 = np.sum((observed - expected_full) ** 2 / expected_full)
        df = len(categories) - 1
        p_value = 1 - stats.chi2.cdf(chi2, df)

        return chi2, p_value

    def calculate_psi(
        self, feature: str, current_data: pd.DataFrame, n_bins: int = 10
    ) -> float:
        """
        Calculate Population Stability Index (PSI) for a feature.

        PSI < 0.1: No significant change
        PSI 0.1-0.2: Moderate change
        PSI > 0.2: Significant change (retraining recommended)

        Args:
            feature: Feature name
            current_data: Current data
            n_bins: Number of bins for discretization

        Returns:
            PSI score
        """
        ref_values = self.reference_data[feature].dropna()
        curr_values = current_data[feature].dropna()

        # Create bins based on reference data
        _, bins = pd.cut(ref_values, bins=n_bins, retbins=True, duplicates="drop")

        # Bin both distributions
        ref_binned = pd.cut(ref_values, bins=bins, include_lowest=True)
        curr_binned = pd.cut(curr_values, bins=bins, include_lowest=True)

        # Calculate frequencies
        ref_freq = ref_binned.value_counts(normalize=True, sort=False)
        curr_freq = curr_binned.value_counts(normalize=True, sort=False)

        # Align indices
        ref_freq = ref_freq.reindex(curr_freq.index, fill_value=1e-10)
        curr_freq = curr_freq.fillna(1e-10)
        ref_freq = ref_freq.fillna(1e-10)

        # Calculate PSI
        psi = np.sum((curr_freq - ref_freq) * np.log(curr_freq / ref_freq))

        return psi

    def _save_detailed_report(
        self, current_data: pd.DataFrame, report: DriftReport
    ) -> None:
        """Save detailed drift report to file."""
        # Create reports directory
        report_dir = Path("outputs/drift_reports")
        report_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON report
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"drift_report_{timestamp}.json"

        report_data = {
            "timestamp": report.timestamp,
            "has_drift": report.has_drift,
            "threshold": report.threshold,
            "n_drifted_features": len(report.drifted_features),
            "drifted_features": report.drifted_features,
            "all_features": {
                feature: {
                    "drift_score": float(report.drift_scores[feature]),
                    "p_value": float(report.p_values[feature]),
                    "has_drift": feature in report.drifted_features,
                }
                for feature in report.drift_scores.keys()
            },
        }

        import json

        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=2)

        logger.info(f"Drift report saved to {report_path}")

    def monitor_continuous(
        self,
        data_stream: pd.DataFrame,
        window_size: int = 1000,
        alert_callback: Optional[callable] = None,
    ) -> List[DriftReport]:
        """
        Monitor drift continuously on streaming data.

        Args:
            data_stream: Streaming data
            window_size: Size of rolling window
            alert_callback: Function to call when drift detected

        Returns:
            List of drift reports
        """
        reports = []

        for i in range(window_size, len(data_stream), window_size // 2):
            window = data_stream.iloc[i - window_size : i]

            report = self.detect_drift(window, save_report=False)
            reports.append(report)

            if report.has_drift and alert_callback:
                alert_callback(report)

        return reports


# Example usage
if __name__ == "__main__":
    # Create synthetic data
    np.random.seed(42)

    # Reference data (training)
    train_data = pd.DataFrame(
        {
            "price": np.random.normal(50, 10, 1000),
            "load": np.random.normal(50000, 10000, 1000),
            "temperature": np.random.normal(15, 5, 1000),
        }
    )

    # Current data (with drift in price)
    prod_data = pd.DataFrame(
        {
            "price": np.random.normal(60, 15, 500),  # Mean shifted
            "load": np.random.normal(50000, 10000, 500),  # No drift
            "temperature": np.random.normal(15, 5, 500),  # No drift
        }
    )

    # Initialize detector
    detector = DriftDetector(reference_data=train_data, threshold=0.05)

    # Detect drift
    report = detector.detect_drift(current_data=prod_data)

    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)

    # Calculate PSI for each feature
    print("\nPopulation Stability Index (PSI):")
    for feature in ["price", "load", "temperature"]:
        psi = detector.calculate_psi(feature, prod_data)
        status = "✅" if psi < 0.1 else "⚠️" if psi < 0.2 else "🔴"
        print(f"{status} {feature}: PSI = {psi:.4f}")
