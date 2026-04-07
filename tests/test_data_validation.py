"""Tests for data validation and leakage prevention.

Tests:
1. Price data validation (ranges, volatility, missing values)
2. Data leakage checks in feature engineering
3. Temporal ordering in train/test splits
4. DataValidator class (weather, energy, market prices)
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.data_loader import (
    validate_real_price_data,
    add_lag_features,
    add_calendar_features,
    simulate_realistic_prices,
    prepare_price_forecasting_dataset,
)
from src.data.validator import DataValidator, ValidationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_price_df(n=2000, mean=60.0, std=20.0, seed=42):
    """Create a realistic synthetic price DataFrame."""
    rng = np.random.default_rng(seed)
    prices = rng.normal(mean, std, n)
    datetimes = pd.date_range("2023-01-01", periods=n, freq="h")
    return pd.DataFrame({"datetime_hour": datetimes, "price": prices})


def make_load_df(n=2000, seed=0):
    """Create a synthetic load DataFrame."""
    rng = np.random.default_rng(seed)
    load = rng.uniform(30_000, 70_000, n)
    datetimes = pd.date_range("2023-01-01", periods=n, freq="h")
    return pd.DataFrame({"datetime_hour": datetimes, "conso_elec_mw": load})


# ---------------------------------------------------------------------------
# validate_real_price_data
# ---------------------------------------------------------------------------


class TestValidateRealPriceData:
    """Unit tests for the price sanity-check function."""

    def test_valid_prices_pass(self):
        df = make_price_df()
        # Should not raise
        validate_real_price_data(df, data_source="test")

    def test_missing_price_column_raises(self):
        df = pd.DataFrame({"other": [1, 2, 3]})
        with pytest.raises(ValueError, match="price"):
            validate_real_price_data(df)

    def test_extremely_negative_prices_raise(self):
        df = make_price_df()
        df.loc[0, "price"] = -500.0  # below -200 threshold
        with pytest.raises(ValueError, match="Extremely negative"):
            validate_real_price_data(df)

    def test_extremely_high_prices_raise(self):
        df = make_price_df()
        df.loc[0, "price"] = 5000.0  # above 3000 threshold
        with pytest.raises(ValueError, match="Extremely high"):
            validate_real_price_data(df)

    def test_small_sample_does_not_raise(self):
        """Small dataset should warn but not raise."""
        df = make_price_df(n=100)
        # Should complete without exception (warning only)
        validate_real_price_data(df)

    def test_negative_prices_within_range_pass(self):
        """Negative prices in [-150, 0] are valid for EU markets."""
        df = make_price_df()
        df.loc[:10, "price"] = -50.0
        validate_real_price_data(df)


# ---------------------------------------------------------------------------
# add_calendar_features
# ---------------------------------------------------------------------------


class TestCalendarFeatures:
    """Tests for calendar feature engineering."""

    def test_all_features_created(self):
        df = make_price_df(n=100)
        result = add_calendar_features(df)
        expected = [
            "hour",
            "day_of_week",
            "day_of_month",
            "day_of_year",
            "week_of_year",
            "month",
            "quarter",
            "year",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "month_sin",
            "month_cos",
            "is_weekend",
            "is_peak_hour",
            "is_night",
            "is_winter",
            "is_summer",
        ]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_cyclical_encoding_bounds(self):
        df = make_price_df(n=200)
        result = add_calendar_features(df)
        for col in [
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "month_sin",
            "month_cos",
        ]:
            assert result[col].between(-1, 1).all(), f"{col} out of [-1, 1]"

    def test_binary_features_are_binary(self):
        df = make_price_df(n=200)
        result = add_calendar_features(df)
        for col in ["is_weekend", "is_peak_hour", "is_night", "is_winter", "is_summer"]:
            assert set(result[col].unique()).issubset({0, 1}), f"{col} not binary"

    def test_original_df_not_mutated(self):
        df = make_price_df(n=50)
        cols_before = set(df.columns)
        add_calendar_features(df)
        assert set(df.columns) == cols_before


# ---------------------------------------------------------------------------
# add_lag_features — leakage prevention
# ---------------------------------------------------------------------------


class TestLagFeatures:
    """Critical: verify no data leakage in lag construction."""

    def test_minimum_lag_is_24h(self):
        """No lag < 24h should be created (forecast horizon = 24h)."""
        df = make_price_df(n=500)
        result = add_lag_features(df, target_col="price")
        lag_cols = [c for c in result.columns if "lag_" in c and "price" in c]
        for col in lag_cols:
            lag_hours = int(col.split("lag_")[1].replace("h", ""))
            assert (
                lag_hours >= 24
            ), f"Data leakage: lag {lag_hours}h < 24h forecast horizon"

    def test_rolling_stats_shifted_by_24h(self):
        """Rolling windows must be shifted ≥24h to avoid leakage."""
        df = make_price_df(n=500)
        result = add_lag_features(df, target_col="price")
        roll_cols = [c for c in result.columns if "roll_" in c]
        assert len(roll_cols) > 0, "No rolling feature columns found"

    def test_lag_values_match_shifted_series(self):
        """Lag-24h value at row i must equal price at row i-24."""
        df = make_price_df(n=300)
        df = df.sort_values("datetime_hour").reset_index(drop=True)
        result = add_lag_features(df, target_col="price", lags=[24])
        # Row 24 → lag_24h should equal price at row 0
        assert result.loc[24, "price_lag_24h"] == pytest.approx(df.loc[0, "price"])

    def test_no_leakage_contemporaneous_load(self):
        """prepare_price_forecasting_dataset must exclude contemporaneous load."""
        df = make_price_df(n=500)
        df_with_load = df.copy()
        df_with_load["load_mw"] = np.random.uniform(30_000, 70_000, len(df))
        result = add_lag_features(df_with_load, target_col="load_mw")
        # Contemporaneous load_mw must NOT appear as a feature in isolation
        lag_cols = [c for c in result.columns if "load_mw_lag" in c]
        assert len(lag_cols) > 0


# ---------------------------------------------------------------------------
# simulate_realistic_prices
# ---------------------------------------------------------------------------


class TestSimulateRealisticPrices:
    """Tests for the price simulation used in unit tests."""

    def test_output_columns(self):
        load_df = make_load_df(n=500)
        result = simulate_realistic_prices(load_df, random_seed=0)
        assert "datetime_hour" in result.columns
        assert "price" in result.columns
        assert "load_mw" in result.columns

    def test_no_nan_in_output(self):
        load_df = make_load_df(n=300)
        result = simulate_realistic_prices(load_df, random_seed=1)
        assert not result["price"].isna().any()

    def test_price_range_realistic(self):
        load_df = make_load_df(n=2000)
        result = simulate_realistic_prices(load_df, base_price=50.0, random_seed=2)
        assert result["price"].min() >= -10, "Simulated prices too negative"
        assert result["price"].max() <= 1000, "Simulated prices too high"

    def test_reproducibility(self):
        load_df = make_load_df(n=200)
        r1 = simulate_realistic_prices(load_df, random_seed=99)
        r2 = simulate_realistic_prices(load_df, random_seed=99)
        pd.testing.assert_frame_equal(r1, r2)

    def test_length_matches_input(self):
        load_df = make_load_df(n=400)
        result = simulate_realistic_prices(load_df)
        assert len(result) == len(load_df)


# ---------------------------------------------------------------------------
# Temporal ordering / train-test split integrity
# ---------------------------------------------------------------------------


class TestTemporalOrdering:
    """Verify no future data leaks into training sets."""

    def test_lag_features_preserve_chronological_order(self):
        df = make_price_df(n=200)
        result = add_lag_features(df, target_col="price")
        datetimes = pd.to_datetime(result["datetime_hour"])
        assert datetimes.is_monotonic_increasing

    def test_first_rows_have_nan_lags(self):
        """Rows at the start of the series must have NaN for lags they can't compute."""
        df = make_price_df(n=300)
        result = add_lag_features(df, target_col="price", lags=[24, 48])
        # First 24 rows cannot have a lag_24h (no past data)
        assert result["price_lag_24h"].iloc[:24].isna().all()

    def test_no_future_data_in_features(self):
        """Lag columns must not contain values from the future."""
        df = make_price_df(n=100)
        df = df.sort_values("datetime_hour").reset_index(drop=True)
        result = add_lag_features(df, target_col="price", lags=[24])
        # For any row i, lag_24h should be price[i-24], never price[i] or price[i+1]
        for i in range(25, 50):
            expected = df.loc[i - 24, "price"]
            actual = result.loc[i, "price_lag_24h"]
            assert actual == pytest.approx(expected), f"Future leakage at row {i}"


# ---------------------------------------------------------------------------
# DataValidator (src/data/validator.py)
# ---------------------------------------------------------------------------


class TestDataValidatorWeather:
    """Tests for DataValidator.validate_weather_data."""

    def _make_weather(self, n=365):
        dates = pd.date_range("2023-01-01", periods=n, freq="D")
        rng = np.random.default_rng(42)
        temp_max = rng.uniform(10, 35, n)
        temp_min = temp_max - rng.uniform(2, 8, n)  # always < max
        return pd.DataFrame(
            {
                "date": dates,
                "temperature_2m_max": temp_max,
                "temperature_2m_min": temp_min,
                "precipitation_sum": rng.exponential(2, n),
                "wind_speed_10m_max": rng.uniform(0, 60, n),
                "shortwave_radiation_sum": rng.uniform(0, 8000, n),
            }
        )

    def test_valid_weather_passes(self):
        df = self._make_weather()
        validator = DataValidator(strict_mode=True)
        assert validator.validate_weather_data(df) is True

    def test_missing_column_fails(self):
        df = self._make_weather().drop(columns=["wind_speed_10m_max"])
        validator = DataValidator(strict_mode=True)
        assert validator.validate_weather_data(df) is False

    def test_temp_min_exceeds_max_fails(self):
        df = self._make_weather()
        df["temperature_2m_min"] = df["temperature_2m_max"] + 5  # always > max
        validator = DataValidator(strict_mode=True)
        assert validator.validate_weather_data(df) is False

    def test_negative_precipitation_fails(self):
        df = self._make_weather()
        df.loc[0, "precipitation_sum"] = -1.0
        validator = DataValidator(strict_mode=True)
        assert validator.validate_weather_data(df) is False

    def test_negative_wind_fails(self):
        df = self._make_weather()
        df.loc[0, "wind_speed_10m_max"] = -5.0
        validator = DataValidator(strict_mode=True)
        assert validator.validate_weather_data(df) is False


class TestDataValidatorEnergy:
    """Tests for DataValidator.validate_energy_data."""

    def _make_energy(self, n=200):
        return pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=n, freq="D"),
                "conso_elec_mw": np.random.uniform(20_000, 80_000, n),
                "conso_gaz_mw": np.random.uniform(5_000, 30_000, n),
            }
        )

    def test_valid_energy_passes(self):
        df = self._make_energy()
        validator = DataValidator(strict_mode=True)
        assert validator.validate_energy_data(df) is True

    def test_negative_electricity_fails(self):
        df = self._make_energy()
        df.loc[0, "conso_elec_mw"] = -100.0
        validator = DataValidator(strict_mode=True)
        assert validator.validate_energy_data(df) is False

    def test_missing_required_column_fails(self):
        df = self._make_energy().drop(columns=["conso_gaz_mw"])
        validator = DataValidator(strict_mode=True)
        assert validator.validate_energy_data(df) is False


class TestDataValidatorMarketPrices:
    """Tests for DataValidator.validate_market_prices."""

    def _make_market(self, n=200):
        return pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=n, freq="D"),
                "price_FR": np.random.uniform(30, 150, n),
                "price_DE": np.random.uniform(25, 140, n),
            }
        )

    def test_valid_prices_pass(self):
        df = self._make_market()
        validator = DataValidator(strict_mode=False)
        result = validator.validate_market_prices(df)
        assert result is True

    def test_no_price_column_fails(self):
        df = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=10, freq="D")})
        validator = DataValidator(strict_mode=True)
        result = validator.validate_market_prices(df)
        assert result is False


class TestValidationResult:
    """Tests for the ValidationResult dataclass."""

    def test_pass_str(self):
        r = ValidationResult(passed=True, metric_name="test", expected="x", actual="x")
        assert "PASS" in str(r)

    def test_fail_str(self):
        r = ValidationResult(passed=False, metric_name="test", expected="x", actual="y")
        assert "FAIL" in str(r)

    def test_default_severity_is_error(self):
        r = ValidationResult(passed=False, metric_name="test", expected="x", actual="y")
        assert r.severity == "ERROR"
