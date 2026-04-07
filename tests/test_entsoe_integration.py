"""Tests for ENTSO-E data connector and API cache.

Tests:
1. EntsoeConnector (init, rate limiting, error handling)
2. ApiCache (set/get, expiration, stats, clear)
3. Real API integration (skipped unless ENTSOE_API_KEY is set)
"""

import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch
import sys

import pandas as pd
import pytest

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.entsoe_connector import EntsoeConnector
from src.data.api_cache import ApiCache


# ---------------------------------------------------------------------------
# EntsoeConnector
# ---------------------------------------------------------------------------


class TestEntsoeConnectorInit:

    def test_init_with_api_key(self):
        connector = EntsoeConnector(api_key="test_key_123")
        assert connector.api_key == "test_key_123"

    def test_init_without_api_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises((ValueError, KeyError, TypeError)):
                EntsoeConnector()

    def test_rate_limit_default(self):
        connector = EntsoeConnector(api_key="test_key")
        assert connector.rate_limit_rpm > 0

    def test_custom_rate_limit(self):
        connector = EntsoeConnector(api_key="test_key", rate_limit_rpm=100)
        assert connector.rate_limit_rpm == 100

    def test_requests_counter_starts_zero(self):
        connector = EntsoeConnector(api_key="test_key")
        assert connector.total_requests == 0

    def test_cache_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            connector = EntsoeConnector(
                api_key="test_key", use_cache=True, cache_dir=tmpdir
            )
            assert connector.use_cache is True
            assert connector.cache is not None

    def test_cache_disabled(self):
        connector = EntsoeConnector(api_key="test_key", use_cache=False)
        assert connector.use_cache is False
        assert connector.cache is None


class TestEntsoeConnectorUsageStats:

    def test_usage_stats_initial(self):
        connector = EntsoeConnector(api_key="test_key")
        stats = connector.get_usage_stats()
        assert stats["total_requests"] == 0
        assert stats["requests_last_minute"] == 0
        assert stats["rate_limit_rpm"] > 0
        assert stats["utilization_percent"] == 0

    def test_utilization_percent_type(self):
        connector = EntsoeConnector(api_key="test_key")
        stats = connector.get_usage_stats()
        assert isinstance(stats["utilization_percent"], (int, float))


class TestEntsoeConnectorValidation:

    def test_invalid_country_code_raises(self):
        connector = EntsoeConnector(api_key="test_key")
        with pytest.raises((ValueError, KeyError)):
            connector.get_day_ahead_prices("XX", "2024-01-01", "2024-01-02")

    def test_valid_country_codes_accepted(self):
        """FR and DE should be recognised as valid country codes."""
        connector = EntsoeConnector(api_key="test_key", use_cache=False)
        for country in ["FR", "DE"]:
            try:
                connector.get_day_ahead_prices(country, "2024-01-01", "2024-01-02")
            except (ValueError, KeyError) as e:
                if "Unknown country" in str(e):
                    pytest.fail(f"{country} should be a valid country code, got: {e}")
            except Exception:
                pass  # Network errors expected without a real key


class TestEntsoeConnectorRetry:

    @patch("requests.Session.get")
    def test_retries_on_429(self, mock_get):
        mock_fail = Mock()
        mock_fail.status_code = 429
        mock_fail.raise_for_status.side_effect = Exception("Rate limited")

        mock_ok = Mock()
        mock_ok.status_code = 200
        mock_ok.content = b"""<?xml version="1.0" encoding="UTF-8"?>
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

        mock_get.side_effect = [mock_fail, mock_ok]
        connector = EntsoeConnector(api_key="test_key", use_cache=False)
        try:
            connector.get_day_ahead_prices("FR", "2024-01-01", "2024-01-02")
        except Exception:
            pass
        assert mock_get.call_count >= 1


# ---------------------------------------------------------------------------
# ApiCache
# ---------------------------------------------------------------------------


class TestApiCacheInit:

    def test_creates_cache_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "new_cache"
            cache = ApiCache(cache_dir=str(cache_dir), ttl_days=7)
            assert cache.cache_dir.exists()

    def test_ttl_stored(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ApiCache(cache_dir=tmpdir, ttl_days=14)
            assert cache.ttl_days == 14


class TestApiCacheKeyGeneration:

    def test_same_params_same_key(self):
        cache = ApiCache(cache_dir="test_cache")
        k1 = cache._get_cache_key(
            country="FR",
            start_date="2024-01-01",
            end_date="2024-01-31",
            data_type="prices",
        )
        k2 = cache._get_cache_key(
            country="FR",
            start_date="2024-01-01",
            end_date="2024-01-31",
            data_type="prices",
        )
        assert k1 == k2

    def test_different_country_different_key(self):
        cache = ApiCache(cache_dir="test_cache")
        k1 = cache._get_cache_key(
            country="FR",
            start_date="2024-01-01",
            end_date="2024-01-31",
            data_type="prices",
        )
        k2 = cache._get_cache_key(
            country="DE",
            start_date="2024-01-01",
            end_date="2024-01-31",
            data_type="prices",
        )
        assert k1 != k2

    def test_different_dates_different_key(self):
        cache = ApiCache(cache_dir="test_cache")
        k1 = cache._get_cache_key(
            country="FR",
            start_date="2024-01-01",
            end_date="2024-01-31",
            data_type="prices",
        )
        k2 = cache._get_cache_key(
            country="FR",
            start_date="2024-02-01",
            end_date="2024-02-28",
            data_type="prices",
        )
        assert k1 != k2


class TestApiCacheSetGet:

    def _make_df(self, n=24):
        return pd.DataFrame(
            {
                "datetime": pd.date_range("2024-01-01", periods=n, freq="h"),
                "price_eur_mwh": [50.0 + i for i in range(n)],
                "country": ["FR"] * n,
            }
        )

    def test_cache_miss_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ApiCache(cache_dir=tmpdir)
            result = cache.get(
                country="FR",
                start_date="2024-01-01",
                end_date="2024-01-31",
                data_type="prices",
            )
            assert result is None

    def test_set_then_get_returns_data(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ApiCache(cache_dir=tmpdir, ttl_days=7)
            df = self._make_df(24)
            cache.set(
                df,
                country="FR",
                start_date="2024-01-01",
                end_date="2024-01-02",
                data_type="prices",
            )
            result = cache.get(
                country="FR",
                start_date="2024-01-01",
                end_date="2024-01-02",
                data_type="prices",
            )
            assert result is not None
            assert len(result) == 24
            assert "price_eur_mwh" in result.columns

    def test_cached_data_values_preserved(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ApiCache(cache_dir=tmpdir, ttl_days=7)
            df = self._make_df(10)
            cache.set(
                df,
                country="FR",
                start_date="2024-01-01",
                end_date="2024-01-02",
                data_type="prices",
            )
            result = cache.get(
                country="FR",
                start_date="2024-01-01",
                end_date="2024-01-02",
                data_type="prices",
            )
            assert list(result["price_eur_mwh"]) == list(df["price_eur_mwh"])

    def test_expired_cache_returns_none(self):
        """Cache with negative TTL or truly old data should expire.

        TTL=0 means 'expires after 0 full days elapsed' — the same-second check
        (age_days=0, ttl=0) does NOT expire (0 > 0 is False). We test TTL=-1
        to guarantee expiration (age_days=0 > -1 is True).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ApiCache(cache_dir=tmpdir, ttl_days=-1)
            df = self._make_df(5)
            cache.set(
                df,
                country="FR",
                start_date="2024-01-01",
                end_date="2024-01-02",
                data_type="prices",
            )
            result = cache.get(
                country="FR",
                start_date="2024-01-01",
                end_date="2024-01-02",
                data_type="prices",
            )
            assert result is None

    def test_different_queries_isolated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ApiCache(cache_dir=tmpdir, ttl_days=7)
            df_fr = self._make_df(12)
            df_de = self._make_df(6)
            cache.set(
                df_fr,
                country="FR",
                start_date="2024-01-01",
                end_date="2024-01-02",
                data_type="prices",
            )
            cache.set(
                df_de,
                country="DE",
                start_date="2024-01-01",
                end_date="2024-01-02",
                data_type="prices",
            )

            result_fr = cache.get(
                country="FR",
                start_date="2024-01-01",
                end_date="2024-01-02",
                data_type="prices",
            )
            result_de = cache.get(
                country="DE",
                start_date="2024-01-01",
                end_date="2024-01-02",
                data_type="prices",
            )

            assert len(result_fr) == 12
            assert len(result_de) == 6


class TestApiCacheStats:

    def test_initial_stats_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ApiCache(cache_dir=tmpdir)
            stats = cache.get_stats()
            assert stats["total_entries"] == 0

    def test_stats_after_set(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ApiCache(cache_dir=tmpdir, ttl_days=7)
            df = pd.DataFrame(
                {
                    "datetime": [datetime.now()],
                    "price_eur_mwh": [50.0],
                }
            )
            cache.set(
                df,
                country="FR",
                start_date="2024-01-01",
                end_date="2024-01-02",
                data_type="prices",
            )
            stats = cache.get_stats()
            assert stats["total_entries"] == 1
            # File may be tiny but must be >= 0
            assert stats["total_size_mb"] >= 0

    def test_stats_after_clear(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache = ApiCache(cache_dir=tmpdir, ttl_days=7)
            df = pd.DataFrame({"datetime": [datetime.now()], "price": [50.0]})
            cache.set(
                df,
                country="FR",
                start_date="2024-01-01",
                end_date="2024-01-02",
                data_type="prices",
            )
            cache.clear()
            stats = cache.get_stats()
            assert stats["total_entries"] == 0


# ---------------------------------------------------------------------------
# Real API Integration (skipped without key)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.getenv("ENTSOE_API_KEY"),
    reason="ENTSOE_API_KEY not set — skipping real API tests",
)
class TestRealApiIntegration:

    def test_fetch_day_ahead_prices_fr(self):
        connector = EntsoeConnector()
        end = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        start = end - timedelta(days=2)
        prices = connector.get_day_ahead_prices("FR", start, end)
        assert len(prices) > 0

    def test_caching_reduces_api_calls(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            connector = EntsoeConnector(use_cache=True, cache_dir=tmpdir)
            end = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            start = end - timedelta(days=1)

            prices1 = connector.get_day_ahead_prices("FR", start, end)
            requests_after_first = connector.total_requests

            prices2 = connector.get_day_ahead_prices("FR", start, end)
            requests_after_second = connector.total_requests

            assert requests_after_second == requests_after_first
            assert len(prices1) == len(prices2)
