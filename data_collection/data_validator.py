"""
Data Quality Validation for ENTSO-E Market Data

This module validates price and load data from ENTSO-E API to ensure:
- Data completeness (no missing timestamps)
- Realistic value ranges
- Outlier detection
- Data consistency

Usage:
    from data_collection.data_validator import validate_entsoe_data

    validator = DataValidator()
    is_valid, report = validator.validate_prices(df)
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ValidationReport:
    """Container for validation results."""

    is_valid: bool
    total_records: int
    missing_values: dict
    outliers: dict
    warnings: list
    errors: list
    statistics: dict


class DataValidator:
    """Validator for ENTSO-E market data."""

    # Reasonable price bounds (EUR/MWh)
    PRICE_MIN_NORMAL = -50.0  # Negative prices possible during renewable surplus
    PRICE_MAX_NORMAL = 500.0  # Typical maximum
    PRICE_EXTREME_MIN = -500.0  # Absolute minimum (seen during COVID-19)
    PRICE_EXTREME_MAX = 3000.0  # Absolute maximum (seen during energy crisis)

    # Load bounds (MW) - adjust based on country
    LOAD_MIN_NORMAL = 20000  # Minimum France load
    LOAD_MAX_NORMAL = 100000  # Maximum France load

    def __init__(self, strict: bool = False):
        """Initialize validator.

        Args:
            strict: If True, use stricter validation rules (warnings become errors)
        """
        self.strict = strict

    def validate_prices(
        self, df: pd.DataFrame, datetime_col: str = "datetime"
    ) -> Tuple[bool, ValidationReport]:
        """Validate day-ahead price data.

        Args:
            df: DataFrame with price data
            datetime_col: Name of datetime column

        Returns:
            tuple: (is_valid, validation_report)
        """
        warnings = []
        errors = []
        missing_values = {}
        outliers = {}

        # Check required columns
        price_col = None
        for col in ["price_eur_mwh", "price", "price_FR"]:
            if col in df.columns:
                price_col = col
                break

        if price_col is None:
            errors.append(f"No price column found. Available: {df.columns.tolist()}")
            return False, ValidationReport(
                is_valid=False,
                total_records=len(df),
                missing_values={},
                outliers={},
                warnings=[],
                errors=errors,
                statistics={},
            )

        if datetime_col not in df.columns:
            errors.append(f"Datetime column '{datetime_col}' not found")

        # Check for missing values
        missing_prices = df[price_col].isna().sum()
        if missing_prices > 0:
            pct = (missing_prices / len(df)) * 100
            msg = f"Missing prices: {missing_prices} ({pct:.1f}%)"
            if pct > 5:
                errors.append(msg)
            else:
                warnings.append(msg)
            missing_values["price"] = missing_prices

        # Check datetime continuity (hourly data should have no gaps)
        if datetime_col in df.columns and not df[datetime_col].isna().all():
            df_sorted = df.sort_values(datetime_col)
            datetime_series = pd.to_datetime(df_sorted[datetime_col])

            # Expected hourly frequency
            expected_freq = pd.Timedelta(hours=1)
            time_diffs = datetime_series.diff()

            # Find gaps (more than 2 hours to account for DST transitions)
            gaps = time_diffs[time_diffs > pd.Timedelta(hours=2)]

            if len(gaps) > 0:
                msg = f"Found {len(gaps)} time gaps in data (missing hours)"
                if len(gaps) > 10:
                    errors.append(msg)
                else:
                    warnings.append(msg)

            # Check for duplicate timestamps
            duplicates = datetime_series.duplicated().sum()
            if duplicates > 0:
                msg = f"Found {duplicates} duplicate timestamps"
                if duplicates > 5:
                    errors.append(msg)
                else:
                    warnings.append(msg)

        # Price range validation
        prices = df[price_col].dropna()

        if len(prices) > 0:
            min_price = prices.min()
            max_price = prices.max()
            mean_price = prices.mean()
            std_price = prices.std()

            # Check for extreme values
            extreme_low = prices[prices < self.PRICE_EXTREME_MIN]
            extreme_high = prices[prices > self.PRICE_EXTREME_MAX]

            if len(extreme_low) > 0:
                errors.append(
                    f"Found {len(extreme_low)} prices below absolute minimum "
                    f"({self.PRICE_EXTREME_MIN} EUR/MWh): min={extreme_low.min():.2f}"
                )

            if len(extreme_high) > 0:
                errors.append(
                    f"Found {len(extreme_high)} prices above absolute maximum "
                    f"({self.PRICE_EXTREME_MAX} EUR/MWh): max={extreme_high.max():.2f}"
                )

            # Check for unusual values (outside normal range)
            abnormal_low = prices[
                (prices >= self.PRICE_EXTREME_MIN) & (prices < self.PRICE_MIN_NORMAL)
            ]
            abnormal_high = prices[
                (prices > self.PRICE_MAX_NORMAL) & (prices <= self.PRICE_EXTREME_MAX)
            ]

            if len(abnormal_low) > 0:
                pct = (len(abnormal_low) / len(prices)) * 100
                msg = (
                    f"Found {len(abnormal_low)} ({pct:.1f}%) very low prices "
                    f"({self.PRICE_MIN_NORMAL} to {self.PRICE_EXTREME_MIN} EUR/MWh). "
                    f"This may be normal during high renewable production."
                )
                warnings.append(msg)
                outliers["very_low"] = len(abnormal_low)

            if len(abnormal_high) > 0:
                pct = (len(abnormal_high) / len(prices)) * 100
                msg = (
                    f"Found {len(abnormal_high)} ({pct:.1f}%) very high prices "
                    f"({self.PRICE_MAX_NORMAL} to {self.PRICE_EXTREME_MAX} EUR/MWh). "
                    f"This may be normal during scarcity events."
                )
                warnings.append(msg)
                outliers["very_high"] = len(abnormal_high)

            # Statistical outlier detection (IQR method)
            q1 = prices.quantile(0.25)
            q3 = prices.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 3 * iqr
            upper_bound = q3 + 3 * iqr

            statistical_outliers = prices[(prices < lower_bound) | (prices > upper_bound)]
            if len(statistical_outliers) > 0:
                pct = (len(statistical_outliers) / len(prices)) * 100
                if pct > 5:
                    warnings.append(
                        f"Statistical outliers: {len(statistical_outliers)} ({pct:.1f}%). "
                        f"This may indicate data quality issues or extreme market events."
                    )
                    outliers["statistical"] = len(statistical_outliers)

            # Compile statistics
            statistics = {
                "count": len(prices),
                "mean": round(mean_price, 2),
                "std": round(std_price, 2),
                "min": round(min_price, 2),
                "q1": round(q1, 2),
                "median": round(prices.median(), 2),
                "q3": round(q3, 2),
                "max": round(max_price, 2),
                "negative_prices": int((prices < 0).sum()),
                "negative_prices_pct": round((prices < 0).sum() / len(prices) * 100, 2),
            }
        else:
            errors.append("No valid price data found")
            statistics = {}

        # Determine overall validity
        is_valid = len(errors) == 0 and (not self.strict or len(warnings) == 0)

        report = ValidationReport(
            is_valid=is_valid,
            total_records=len(df),
            missing_values=missing_values,
            outliers=outliers,
            warnings=warnings,
            errors=errors,
            statistics=statistics,
        )

        return is_valid, report

    def validate_load(
        self, df: pd.DataFrame, datetime_col: str = "datetime"
    ) -> Tuple[bool, ValidationReport]:
        """Validate load (consumption) data.

        Args:
            df: DataFrame with load data
            datetime_col: Name of datetime column

        Returns:
            tuple: (is_valid, validation_report)
        """
        warnings = []
        errors = []
        missing_values = {}
        outliers = {}

        # Find load column
        load_col = None
        for col in ["load_mw", "load", "actual_load"]:
            if col in df.columns:
                load_col = col
                break

        if load_col is None:
            errors.append(f"No load column found. Available: {df.columns.tolist()}")
            return False, ValidationReport(
                is_valid=False,
                total_records=len(df),
                missing_values={},
                outliers={},
                warnings=[],
                errors=errors,
                statistics={},
            )

        # Check for missing values
        missing_load = df[load_col].isna().sum()
        if missing_load > 0:
            pct = (missing_load / len(df)) * 100
            msg = f"Missing load values: {missing_load} ({pct:.1f}%)"
            if pct > 5:
                errors.append(msg)
            else:
                warnings.append(msg)
            missing_values["load"] = missing_load

        # Load range validation
        loads = df[load_col].dropna()

        if len(loads) > 0:
            min_load = loads.min()
            max_load = loads.max()

            # Check for impossible values
            if min_load < 0:
                errors.append(f"Negative load values found: min={min_load:.0f} MW")

            # Check for unusual values
            if min_load < self.LOAD_MIN_NORMAL:
                warnings.append(
                    f"Unusually low load: {min_load:.0f} MW (expected >{self.LOAD_MIN_NORMAL} MW)"
                )
                outliers["low_load"] = int((loads < self.LOAD_MIN_NORMAL).sum())

            if max_load > self.LOAD_MAX_NORMAL:
                warnings.append(
                    f"Unusually high load: {max_load:.0f} MW (expected <{self.LOAD_MAX_NORMAL} MW)"
                )
                outliers["high_load"] = int((loads > self.LOAD_MAX_NORMAL).sum())

            statistics = {
                "count": len(loads),
                "mean": round(loads.mean(), 0),
                "std": round(loads.std(), 0),
                "min": round(min_load, 0),
                "median": round(loads.median(), 0),
                "max": round(max_load, 0),
            }
        else:
            errors.append("No valid load data found")
            statistics = {}

        is_valid = len(errors) == 0 and (not self.strict or len(warnings) == 0)

        report = ValidationReport(
            is_valid=is_valid,
            total_records=len(df),
            missing_values=missing_values,
            outliers=outliers,
            warnings=warnings,
            errors=errors,
            statistics=statistics,
        )

        return is_valid, report

    def print_report(self, report: ValidationReport):
        """Print validation report in a readable format.

        Args:
            report: Validation report to print
        """
        print("\n" + "=" * 70)
        print("DATA VALIDATION REPORT")
        print("=" * 70)

        # Overall status
        status = "✅ VALID" if report.is_valid else "❌ INVALID"
        print(f"\nStatus: {status}")
        print(f"Total records: {report.total_records}")

        # Statistics
        if report.statistics:
            print("\n📊 Statistics:")
            for key, value in report.statistics.items():
                print(f"  {key}: {value}")

        # Missing values
        if report.missing_values:
            print("\n⚠️  Missing Values:")
            for col, count in report.missing_values.items():
                pct = (count / report.total_records) * 100
                print(f"  {col}: {count} ({pct:.1f}%)")

        # Outliers
        if report.outliers:
            print("\n🔍 Outliers:")
            for outlier_type, count in report.outliers.items():
                pct = (count / report.total_records) * 100
                print(f"  {outlier_type}: {count} ({pct:.1f}%)")

        # Warnings
        if report.warnings:
            print("\n⚠️  Warnings:")
            for i, warning in enumerate(report.warnings, 1):
                print(f"  {i}. {warning}")

        # Errors
        if report.errors:
            print("\n❌ Errors:")
            for i, error in enumerate(report.errors, 1):
                print(f"  {i}. {error}")

        print("\n" + "=" * 70)


def validate_entsoe_data(
    csv_path: str, data_type: str = "prices", strict: bool = False
) -> bool:
    """Validate ENTSO-E data from CSV file.

    Args:
        csv_path: Path to CSV file
        data_type: Type of data ('prices' or 'load')
        strict: Use strict validation

    Returns:
        bool: True if data is valid
    """
    logger.info(f"Validating {data_type} data from {csv_path}")

    df = pd.read_csv(csv_path)

    validator = DataValidator(strict=strict)

    if data_type == "prices":
        is_valid, report = validator.validate_prices(df)
    elif data_type == "load":
        is_valid, report = validator.validate_load(df)
    else:
        raise ValueError(f"Unknown data type: {data_type}")

    validator.print_report(report)

    return is_valid


def main():
    """CLI for data validation."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate ENTSO-E data quality")
    parser.add_argument("csv_path", help="Path to CSV file")
    parser.add_argument(
        "--type",
        choices=["prices", "load"],
        default="prices",
        help="Type of data to validate",
    )
    parser.add_argument("--strict", action="store_true", help="Use strict validation")

    args = parser.parse_args()

    is_valid = validate_entsoe_data(args.csv_path, args.type, args.strict)

    if not is_valid:
        exit(1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
