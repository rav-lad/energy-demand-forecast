"""
Data Validation with Great Expectations

This module provides data quality checks for:
- Weather data
- Energy consumption data
- Market prices
- Model predictions
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a validation check."""

    passed: bool
    metric_name: str
    expected: str
    actual: str
    severity: str = "ERROR"  # ERROR, WARNING, INFO

    def __str__(self):
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return f"{status} [{self.severity}] {self.metric_name}: expected {self.expected}, got {self.actual}"


class DataValidator:
    """
    Validate data quality for energy trading platform.

    Features:
    - Schema validation
    - Range checks
    - Missing value detection
    - Outlier detection
    - Statistical validation
    """

    def __init__(self, strict_mode: bool = True):
        """
        Initialize data validator.

        Args:
            strict_mode: If True, fail on any validation error
        """
        self.strict_mode = strict_mode
        self.validation_results: List[ValidationResult] = []

    def validate_weather_data(self, df: pd.DataFrame) -> bool:
        """
        Validate weather data quality.

        Args:
            df: Weather DataFrame

        Returns:
            True if all validations pass
        """
        logger.info("Validating weather data...")
        self.validation_results = []

        # Check required columns
        required_cols = [
            "date",
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_sum",
            "wind_speed_10m_max",
            "shortwave_radiation_sum",
        ]

        for col in required_cols:
            result = ValidationResult(
                passed=col in df.columns,
                metric_name=f"Column '{col}' exists",
                expected="Column present",
                actual=f"{'Present' if col in df.columns else 'Missing'}",
            )
            self.validation_results.append(result)

        if not all(r.passed for r in self.validation_results):
            logger.error("Missing required columns")
            return False

        # Temperature range checks
        temp_max = df["temperature_2m_max"]
        temp_min = df["temperature_2m_min"]

        self.validation_results.extend(
            [
                ValidationResult(
                    passed=(temp_max >= -50).all(),
                    metric_name="Temperature max >= -50°C",
                    expected=">= -50",
                    actual=f"min={temp_max.min():.1f}",
                ),
                ValidationResult(
                    passed=(temp_max <= 50).all(),
                    metric_name="Temperature max <= 50°C",
                    expected="<= 50",
                    actual=f"max={temp_max.max():.1f}",
                ),
                ValidationResult(
                    passed=(temp_min >= -50).all(),
                    metric_name="Temperature min >= -50°C",
                    expected=">= -50",
                    actual=f"min={temp_min.min():.1f}",
                ),
                ValidationResult(
                    passed=(temp_min <= temp_max).all(),
                    metric_name="Temperature min <= max",
                    expected="min <= max",
                    actual=f"violations={(temp_min > temp_max).sum()}",
                ),
            ]
        )

        # Precipitation checks
        precip = df["precipitation_sum"]
        self.validation_results.extend(
            [
                ValidationResult(
                    passed=(precip >= 0).all(),
                    metric_name="Precipitation >= 0",
                    expected=">= 0",
                    actual=f"min={precip.min():.1f}",
                ),
                ValidationResult(
                    passed=(precip <= 500).all(),
                    metric_name="Precipitation <= 500mm (sanity)",
                    expected="<= 500",
                    actual=f"max={precip.max():.1f}",
                    severity="WARNING",
                ),
            ]
        )

        # Wind speed checks
        wind = df["wind_speed_10m_max"]
        self.validation_results.extend(
            [
                ValidationResult(
                    passed=(wind >= 0).all(),
                    metric_name="Wind speed >= 0",
                    expected=">= 0",
                    actual=f"min={wind.min():.1f}",
                ),
                ValidationResult(
                    passed=(wind <= 200).all(),
                    metric_name="Wind speed <= 200 km/h (sanity)",
                    expected="<= 200",
                    actual=f"max={wind.max():.1f}",
                    severity="WARNING",
                ),
            ]
        )

        # Radiation checks
        radiation = df["shortwave_radiation_sum"]
        self.validation_results.extend(
            [
                ValidationResult(
                    passed=(radiation >= 0).all(),
                    metric_name="Radiation >= 0",
                    expected=">= 0",
                    actual=f"min={radiation.min():.1f}",
                ),
            ]
        )

        # Missing values check
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
        self.validation_results.append(
            ValidationResult(
                passed=missing_pct < 5.0,
                metric_name="Missing values < 5%",
                expected="< 5%",
                actual=f"{missing_pct:.2f}%",
                severity="WARNING",
            )
        )

        return self._report_results()

    def validate_energy_data(self, df: pd.DataFrame) -> bool:
        """
        Validate energy consumption data.

        Args:
            df: Energy consumption DataFrame

        Returns:
            True if all validations pass
        """
        logger.info("Validating energy consumption data...")
        self.validation_results = []

        # Check required columns
        required_cols = ["date", "conso_elec_mw", "conso_gaz_mw"]

        for col in required_cols:
            result = ValidationResult(
                passed=col in df.columns,
                metric_name=f"Column '{col}' exists",
                expected="Column present",
                actual=f"{'Present' if col in df.columns else 'Missing'}",
            )
            self.validation_results.append(result)

        if not all(r.passed for r in self.validation_results):
            logger.error("Missing required columns")
            return False

        # Electricity consumption checks
        elec = df["conso_elec_mw"]
        self.validation_results.extend(
            [
                ValidationResult(
                    passed=(elec >= 0).all(),
                    metric_name="Electricity consumption >= 0",
                    expected=">= 0",
                    actual=f"min={elec.min():.1f}",
                ),
                ValidationResult(
                    passed=(elec <= 100000).all(),
                    metric_name="Electricity consumption <= 100000 MW",
                    expected="<= 100000",
                    actual=f"max={elec.max():.1f}",
                    severity="WARNING",
                ),
            ]
        )

        # Gas consumption checks
        gaz = df["conso_gaz_mw"]
        self.validation_results.extend(
            [
                ValidationResult(
                    passed=(gaz >= 0).all(),
                    metric_name="Gas consumption >= 0",
                    expected=">= 0",
                    actual=f"min={gaz.min():.1f}",
                ),
                ValidationResult(
                    passed=(gaz <= 50000).all(),
                    metric_name="Gas consumption <= 50000 MW",
                    expected="<= 50000",
                    actual=f"max={gaz.max():.1f}",
                    severity="WARNING",
                ),
            ]
        )

        # Check for suspicious patterns
        # Sudden jumps (> 50% change)
        elec_pct_change = elec.pct_change().abs()
        suspicious_jumps = (elec_pct_change > 0.5).sum()

        self.validation_results.append(
            ValidationResult(
                passed=suspicious_jumps < len(df) * 0.01,  # < 1% of data
                metric_name="Suspicious consumption jumps < 1%",
                expected="< 1%",
                actual=f"{suspicious_jumps/len(df)*100:.2f}%",
                severity="WARNING",
            )
        )

        return self._report_results()

    def validate_market_prices(self, df: pd.DataFrame) -> bool:
        """
        Validate market price data.

        Args:
            df: Market prices DataFrame

        Returns:
            True if all validations pass
        """
        logger.info("Validating market prices...")
        self.validation_results = []

        # Check date column
        self.validation_results.append(
            ValidationResult(
                passed="date" in df.columns,
                metric_name="Column 'date' exists",
                expected="Column present",
                actual=f"{'Present' if 'date' in df.columns else 'Missing'}",
            )
        )

        # Check price columns
        price_cols = [col for col in df.columns if col.startswith("price_")]

        self.validation_results.append(
            ValidationResult(
                passed=len(price_cols) > 0,
                metric_name="At least one price column exists",
                expected=">= 1",
                actual=f"{len(price_cols)}",
            )
        )

        # Validate each price column
        for col in price_cols:
            prices = df[col]

            self.validation_results.extend(
                [
                    ValidationResult(
                        passed=(prices >= -500).all(),
                        metric_name=f"{col} >= -500 EUR/MWh",
                        expected=">= -500",
                        actual=f"min={prices.min():.2f}",
                        severity="WARNING",  # Negative prices can happen
                    ),
                    ValidationResult(
                        passed=(prices <= 3000).all(),
                        metric_name=f"{col} <= 3000 EUR/MWh",
                        expected="<= 3000",
                        actual=f"max={prices.max():.2f}",
                        severity="WARNING",
                    ),
                ]
            )

            # Check for suspiciously constant prices
            unique_ratio = len(prices.unique()) / len(prices)
            self.validation_results.append(
                ValidationResult(
                    passed=unique_ratio > 0.1,
                    metric_name=f"{col} has sufficient variance",
                    expected="> 10% unique values",
                    actual=f"{unique_ratio*100:.1f}%",
                    severity="WARNING",
                )
            )

        return self._report_results()

    def validate_predictions(
        self,
        y_pred: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        target_names: Optional[List[str]] = None,
    ) -> bool:
        """
        Validate model predictions.

        Args:
            y_pred: Predicted values
            y_true: Actual values (optional)
            target_names: Names of targets

        Returns:
            True if all validations pass
        """
        logger.info("Validating predictions...")
        self.validation_results = []

        # Check for NaN/Inf
        self.validation_results.extend(
            [
                ValidationResult(
                    passed=not np.isnan(y_pred).any(),
                    metric_name="No NaN in predictions",
                    expected="No NaN",
                    actual=f"{np.isnan(y_pred).sum()} NaN values",
                ),
                ValidationResult(
                    passed=not np.isinf(y_pred).any(),
                    metric_name="No Inf in predictions",
                    expected="No Inf",
                    actual=f"{np.isinf(y_pred).sum()} Inf values",
                ),
            ]
        )

        # Check reasonable ranges (assuming energy predictions)
        self.validation_results.extend(
            [
                ValidationResult(
                    passed=(y_pred >= 0).all(),
                    metric_name="Predictions >= 0",
                    expected=">= 0",
                    actual=f"min={y_pred.min():.2f}",
                ),
                ValidationResult(
                    passed=(y_pred <= 200000).all(),
                    metric_name="Predictions <= 200000 MW",
                    expected="<= 200000",
                    actual=f"max={y_pred.max():.2f}",
                    severity="WARNING",
                ),
            ]
        )

        # If ground truth available, check prediction errors
        if y_true is not None:
            errors = np.abs(y_pred - y_true)
            mape = np.mean(errors / (y_true + 1e-10)) * 100

            self.validation_results.append(
                ValidationResult(
                    passed=mape < 50.0,  # MAPE should be reasonable
                    metric_name="MAPE < 50%",
                    expected="< 50%",
                    actual=f"{mape:.2f}%",
                    severity="WARNING",
                )
            )

        return self._report_results()

    def _report_results(self) -> bool:
        """
        Report validation results.

        Returns:
            True if all critical validations passed
        """
        errors = [
            r for r in self.validation_results if not r.passed and r.severity == "ERROR"
        ]
        warnings = [
            r
            for r in self.validation_results
            if not r.passed and r.severity == "WARNING"
        ]

        logger.info(f"\nValidation Results: {len(self.validation_results)} checks")

        for result in self.validation_results:
            if not result.passed:
                if result.severity == "ERROR":
                    logger.error(str(result))
                else:
                    logger.warning(str(result))

        logger.info(
            f"✓ Passed: {len([r for r in self.validation_results if r.passed])}"
        )
        logger.info(f"✗ Failed: {len(errors)} errors, {len(warnings)} warnings")

        # In strict mode, fail if any error
        if self.strict_mode and len(errors) > 0:
            logger.error("Validation failed in strict mode")
            return False

        return len(errors) == 0


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test weather data validation
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    weather_data = pd.DataFrame(
        {
            "date": dates,
            "temperature_2m_max": np.random.normal(15, 10, len(dates)),
            "temperature_2m_min": np.random.normal(5, 8, len(dates)),
            "precipitation_sum": np.random.exponential(2, len(dates)),
            "wind_speed_10m_max": np.random.exponential(15, len(dates)),
            "shortwave_radiation_sum": np.random.normal(5000, 2000, len(dates)),
        }
    )

    validator = DataValidator(strict_mode=True)
    result = validator.validate_weather_data(weather_data)

    print(f"\nValidation result: {'PASSED' if result else 'FAILED'}")
