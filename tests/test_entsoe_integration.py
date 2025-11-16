"""
Integration Tests for ENTSO-E API

These tests verify the ENTSO-E connector, caching, and data loading functionality.
Some tests are skipped if ENTSOE_API_KEY is not configured to allow CI/CD without API access.

Run with:
    pytest tests/test_entsoe_integration.py -v
    pytest tests/test_entsoe_integration.py -v -k "not real_api"  # Skip real API tests
"""

import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pandas as pd
import pytest

from data_collection.entsoe_connector import EntsoeConnector
from data_collection.api_cache import ApiCache
from data_collection.data_validator import DataValidator


class TestEntsoeConnector:
    """Test ENTSO-E connector functionality."""

    def test_init_without_api_key(self):
        """Test that connector raises error without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="API key required"):
                EntsoeConnector()

    def test_init_with_api_key(self):
        """Test connector initialization with API key."""
        connector = EntsoeConnector(api_key="test_key_123")
        assert connector.api_key == "test_key_123"
        assert connector.rate_limit_rpm == 400
        assert connector.total_requests == 0

    def test_rate_limiter_initialization(self):
        """Test rate limiter is properly initialized."""
        connector = EntsoeConnector(api_key="test_key", rate_limit_rpm=100)
        assert connector.rate_limit_rpm == 100
        assert len(connector.request_timestamps) == 0

    def test_usage_stats(self):
        """Test usage statistics tracking."""
        connector = EntsoeConnector(api_key="test_key")
        stats = connector.get_usage_stats()

        assert stats["total_requests"] == 0
        assert stats["requests_last_minute"] == 0
        assert stats["rate_limit_rpm"] == 400
        assert stats["utilization_percent"] == 0

    def test_cache_initialization(self):
        """Test cache is initialized when enabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            connector = EntsoeConnector(
                api_key="test_key", use_cache=True, cache_dir=tmpdir
            )
            assert connector.use_cache is True
            assert connector.cache is not None

    def test_cache_disabled(self):
        """Test connector works without cache."""
        connector = EntsoeConnector(api_key="test_key", use_cache=False)
        assert connector.use_cache is False
        assert connector.cache is None

    @patch("requests.Session.get")
    def test_rate_limit_retry(self, mock_get):
        """Test that connector retries on 429 rate limit."""
        # First call: rate limited, second call: success
        mock_response_fail = Mock()
        mock_response_fail.status_code = 429
        mock_response_fail.raise_for_status.side_effect = Exception("Rate limited")

        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.content = b"""<?xml version="1.0" encoding="UTF-8"?>
        <Publication_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:0">
            <TimeSeries>
                <Period>
                    <timeInterval>
                        <start>2024-01-01T00:00:00Z</start>
                    </timeInterval>
                    <Point>
                        <position>1</position>
                        <price.amount>50.0</price.amount>
                    </Point>
                </Period>
            </TimeSeries>
        </Publication_MarketDocument>"""

        mock_get.side_effect = [mock_response_fail, mock_response_success]

        connector = EntsoeConnector(api_key="test_key", use_cache=False)

        # This should succeed after retry
        result = connector.get_day_ahead_prices("FR", "2024-01-01", "2024-01-02")

        assert len(result) > 0
        assert mock_get.call_count == 2

    def test_invalid_country_code(self):
        """Test error handling for invalid country code."""
        connector = EntsoeConnector(api_key="test_key")

        with pytest.raises(ValueError, match="Unknown country code"):
            connector.get_day_ahead_prices("XX", "2024-01-01", "2024-01-02")


class TestApiCache:
    """Test API caching functionality."""

    def test_cache_initialization(self):
        """Test cache directory creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ApiCache(cache_dir=tmpdir, ttl_days=7)
            assert cache.cache_dir.exists()
            assert cache.metadata_dir.exists()
            assert cache.ttl_days == 7

    def test_cache_key_generation(self):
        """Test cache key is consistent."""
        cache = ApiCache(cache_dir="test_cache")

        key1 = cache._get_cache_key(
            country="FR", start_date="2024-01-01", end_date="2024-01-31", data_type="prices"
        )
        key2 = cache._get_cache_key(
            country="FR", start_date="2024-01-01", end_date="2024-01-31", data_type="prices"
        )

        # Same parameters should give same key
        assert key1 == key2

        # Different parameters should give different key
        key3 = cache._get_cache_key(
            country="DE", start_date="2024-01-01", end_date="2024-01-31", data_type="prices"
        )
        assert key1 != key3

    def test_cache_miss(self):
        """Test cache returns None on miss."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ApiCache(cache_dir=tmpdir)

            result = cache.get(
                country="FR", start_date="2024-01-01", end_date="2024-01-31", data_type="prices"
            )

            assert result is None

    def test_cache_set_and_get(self):
        """Test caching and retrieval of data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ApiCache(cache_dir=tmpdir, ttl_days=7)

            # Create test data
            df = pd.DataFrame(
                {
                    "datetime": pd.date_range("2024-01-01", periods=24, freq="h"),
                    "price_eur_mwh": [50.0 + i for i in range(24)],
                    "country": ["FR"] * 24,
                }
            )

            # Cache data
            cache.set(df, country="FR", start_date="2024-01-01", end_date="2024-01-02", data_type="prices")

            # Retrieve from cache
            cached_df = cache.get(
                country="FR", start_date="2024-01-01", end_date="2024-01-02", data_type="prices"
            )

            assert cached_df is not None
            assert len(cached_df) == 24
            assert "price_eur_mwh" in cached_df.columns

    def test_cache_expiration(self):
        """Test that expired cache is not returned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ApiCache(cache_dir=tmpdir, ttl_days=0)  # 0 day TTL

            df = pd.DataFrame(
                {
                    "datetime": pd.date_range("2024-01-01", periods=24, freq="h"),
                    "price_eur_mwh": [50.0] * 24,
                }
            )

            cache.set(df, country="FR", start_date="2024-01-01", end_date="2024-01-02", data_type="prices")

            # Should be expired immediately with 0 day TTL
            result = cache.get(
                country="FR", start_date="2024-01-01", end_date="2024-01-02", data_type="prices"
            )

            assert result is None

    def test_cache_stats(self):
        """Test cache statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ApiCache(cache_dir=tmpdir)

            # Initially empty
            stats = cache.get_stats()
            assert stats["total_entries"] == 0

            # Add some data
            df = pd.DataFrame({"datetime": [datetime.now()], "price": [50.0]})
            cache.set(df, country="FR", start_date="2024-01-01", end_date="2024-01-02", data_type="prices")

            stats = cache.get_stats()
            assert stats["total_entries"] == 1
            assert stats["total_size_mb"] > 0

    def test_cache_clear(self):
        """Test cache clearing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ApiCache(cache_dir=tmpdir)

            df = pd.DataFrame({"datetime": [datetime.now()], "price": [50.0]})
            cache.set(df, country="FR", start_date="2024-01-01", end_date="2024-01-02", data_type="prices")

            # Verify cache exists
            assert cache.get_stats()["total_entries"] == 1

            # Clear cache
            cache.clear()

            # Verify cache is empty
            assert cache.get_stats()["total_entries"] == 0


class TestDataValidator:
    """Test data validation functionality."""

    def test_validate_valid_prices(self):
        """Test validation of valid price data."""
        df = pd.DataFrame(
            {
                "datetime": pd.date_range("2024-01-01", periods=100, freq="h"),
                "price_eur_mwh": [50.0 + i * 0.5 for i in range(100)],
            }
        )

        validator = DataValidator()
        is_valid, report = validator.validate_prices(df)

        assert is_valid
        assert report.total_records == 100
        assert len(report.errors) == 0
        assert "mean" in report.statistics

    def test_validate_missing_price_column(self):
        """Test validation fails without price column."""
        df = pd.DataFrame({"datetime": pd.date_range("2024-01-01", periods=10, freq="h")})

        validator = DataValidator()
        is_valid, report = validator.validate_prices(df)

        assert not is_valid
        assert len(report.errors) > 0

    def test_validate_extreme_prices(self):
        """Test validation detects extreme prices."""
        df = pd.DataFrame(
            {
                "datetime": pd.date_range("2024-01-01", periods=100, freq="h"),
                "price_eur_mwh": [50.0] * 95 + [5000.0] * 5,  # 5 extreme values
            }
        )

        validator = DataValidator()
        is_valid, report = validator.validate_prices(df)

        assert not is_valid  # Should fail due to extreme values
        assert len(report.errors) > 0

    def test_validate_negative_prices(self):
        """Test validation handles negative prices (which are valid)."""
        df = pd.DataFrame(
            {
                "datetime": pd.date_range("2024-01-01", periods=100, freq="h"),
                "price_eur_mwh": [50.0] * 90 + [-10.0] * 10,
            }
        )

        validator = DataValidator()
        is_valid, report = validator.validate_prices(df)

        # Negative prices should be flagged but not necessarily invalid
        assert report.statistics["negative_prices"] == 10

    def test_validate_missing_values(self):
        """Test validation detects missing values."""
        df = pd.DataFrame(
            {
                "datetime": pd.date_range("2024-01-01", periods=100, freq="h"),
                "price_eur_mwh": [50.0] * 90 + [None] * 10,
            }
        )

        validator = DataValidator()
        is_valid, report = validator.validate_prices(df)

        assert "price" in report.missing_values
        assert report.missing_values["price"] == 10

    def test_validate_load_data(self):
        """Test validation of load data."""
        df = pd.DataFrame(
            {
                "datetime": pd.date_range("2024-01-01", periods=100, freq="h"),
                "load_mw": [50000.0 + i * 100 for i in range(100)],
            }
        )

        validator = DataValidator()
        is_valid, report = validator.validate_load(df)

        assert is_valid
        assert report.total_records == 100

    def test_validate_negative_load(self):
        """Test validation fails for negative load."""
        df = pd.DataFrame(
            {"datetime": pd.date_range("2024-01-01", periods=10, freq="h"), "load_mw": [-1000.0] * 10}
        )

        validator = DataValidator()
        is_valid, report = validator.validate_load(df)

        assert not is_valid
        assert len(report.errors) > 0


@pytest.mark.skipif(
    not os.getenv("ENTSOE_API_KEY"),
    reason="ENTSOE_API_KEY not set - skipping real API tests",
)
class TestRealApiIntegration:
    """Integration tests with real ENTSO-E API.

    These tests are only run if ENTSOE_API_KEY environment variable is set.
    They make real API calls and verify end-to-end functionality.
    """

    def test_real_api_fetch_prices(self):
        """Test fetching real price data from API."""
        connector = EntsoeConnector()

        # Fetch just 2 days of recent data
        end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(days=2)

        prices = connector.get_day_ahead_prices("FR", start_date, end_date)

        assert len(prices) > 0
        assert "price_eur_mwh" in prices.columns
        assert "datetime" in prices.columns

        # Validate the data
        validator = DataValidator()
        is_valid, report = validator.validate_prices(prices, datetime_col="datetime")

        # Print validation report
        validator.print_report(report)

        # Should be valid or have only minor warnings
        assert len(report.errors) == 0

    def test_real_api_with_cache(self):
        """Test that caching works with real API."""
        with tempfile.TemporaryDirectory() as tmpdir:
            connector = EntsoeConnector(use_cache=True, cache_dir=tmpdir)

            end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            start_date = end_date - timedelta(days=1)

            # First call - should hit API
            prices1 = connector.get_day_ahead_prices("FR", start_date, end_date)

            # Second call - should hit cache
            prices2 = connector.get_day_ahead_prices("FR", start_date, end_date)

            # Should get same data
            assert len(prices1) == len(prices2)

            # Check cache was used
            cache_stats = connector.cache.get_stats()
            assert cache_stats["total_entries"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
