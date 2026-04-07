"""ENTSO-E Transparency Platform API Connector

This module provides a clean interface to fetch real electricity market data from
ENTSO-E Transparency Platform (https://transparency.entsoe.eu/).

Key Features:
- Day-ahead prices (EUR/MWh) for European markets
- Actual load data
- Generation by fuel type
- Cross-border flows
- Retry logic with exponential backoff

API Documentation: https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html

Requirements:
    pip install entsoe-py requests pandas

Usage:
    connector = EntsoeConnector(api_key="YOUR_API_KEY")
    prices = connector.get_day_ahead_prices(
        country="FR",
        start="2022-01-01",
        end="2022-12-31"
    )
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Optional
from collections import deque

import pandas as pd
import requests

logger = logging.getLogger(__name__)


class EntsoeConnector:
    """Connector for ENTSO-E Transparency Platform API.

    API Registration: https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html#_authentication_and_authorisation

    Free tier limitations:
    - 400 requests per minute
    - Rate limiting: use retry logic
    """

    BASE_URL = "https://web-api.tp.entsoe.eu/api"

    # EIC (Energy Identification Code) for European countries
    COUNTRY_CODES = {
        "FR": "10YFR-RTE------C",  # France
        "DE": "10Y1001A1001A83F",  # Germany
        "ES": "10YES-REE------0",  # Spain
        "IT": "10YIT-GRTN-----B",  # Italy
        "GB": "10YGB----------A",  # Great Britain
        "NL": "10YNL----------L",  # Netherlands
        "BE": "10YBE----------2",  # Belgium
        "NO": "10YNO-0--------C",  # Norway
        "SE": "10YSE-1--------K",  # Sweden
        "DK": "10Y1001A1001A65H",  # Denmark
    }

    # Document types
    DOC_TYPE_DAY_AHEAD_PRICES = "A44"  # Day-ahead prices
    DOC_TYPE_ACTUAL_LOAD = "A65"  # Actual total load
    DOC_TYPE_GENERATION = "A75"  # Actual generation per type

    def __init__(
        self,
        api_key: Optional[str] = None,
        rate_limit_rpm: int = 400,
        use_cache: bool = True,
        cache_dir: str = "data/cache/entsoe",
        cache_ttl_days: int = 7,
    ):
        """Initialize ENTSO-E connector.

        Args:
            api_key: ENTSO-E API key. If None, will try to read from environment variable ENTSOE_API_KEY.
            rate_limit_rpm: Maximum requests per minute (default: 400, ENTSO-E free tier limit).
            use_cache: Whether to use caching for API responses.
            cache_dir: Directory for cache storage.
            cache_ttl_days: Time-to-live for cached data in days.

        Raises:
            ValueError: If no API key is provided or found in environment.
        """
        import os

        self.api_key = api_key or os.getenv("ENTSOE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ENTSO-E API key required. Either pass api_key parameter or set ENTSOE_API_KEY environment variable.\n"
                "Get your key at: https://transparency.entsoe.eu/content/static_content/Static%20content/web%20api/Guide.html"
            )

        self.session = requests.Session()
        self.session.params = {"securityToken": self.api_key}  # type: ignore

        # Rate limiting (token bucket algorithm)
        self.rate_limit_rpm = rate_limit_rpm
        self.request_timestamps = deque()  # Track request timestamps
        self.total_requests = 0  # Total requests counter

        # Caching
        self.use_cache = use_cache
        self.cache = None
        if use_cache:
            from src.data.api_cache import ApiCache

            self.cache = ApiCache(cache_dir=cache_dir, ttl_days=cache_ttl_days)
            logger.info("API caching enabled")

    def _enforce_rate_limit(self):
        """Enforce rate limiting using token bucket algorithm.

        Ensures we don't exceed rate_limit_rpm requests per minute.
        Sleeps if necessary to comply with rate limits.
        """
        now = time.time()

        # Remove timestamps older than 1 minute
        while self.request_timestamps and now - self.request_timestamps[0] > 60:
            self.request_timestamps.popleft()

        # Check if we've hit the rate limit
        if len(self.request_timestamps) >= self.rate_limit_rpm:
            # Calculate how long to wait
            oldest_timestamp = self.request_timestamps[0]
            wait_time = 60 - (now - oldest_timestamp) + 0.1  # Add 100ms buffer

            if wait_time > 0:
                logger.warning(
                    f"Rate limit reached ({self.rate_limit_rpm} req/min). "
                    f"Waiting {wait_time:.1f}s..."
                )
                time.sleep(wait_time)
                # Clean up old timestamps after waiting
                now = time.time()
                while self.request_timestamps and now - self.request_timestamps[0] > 60:
                    self.request_timestamps.popleft()

        # Record this request
        self.request_timestamps.append(now)
        self.total_requests += 1

        # Log progress periodically
        if self.total_requests % 50 == 0:
            logger.info(
                f"API requests: {self.total_requests} total, "
                f"{len(self.request_timestamps)} in last minute"
            )

    def _make_request(
        self,
        params: dict,
        max_retries: int = 3,
        initial_delay: float = 1.0,
    ) -> requests.Response:
        """Make API request with retry logic and rate limiting.

        Args:
            params: Query parameters.
            max_retries: Maximum retry attempts.
            initial_delay: Initial delay before retry (exponential backoff).

        Returns:
            Response object.

        Raises:
            requests.exceptions.RequestException: If all retries fail.
        """
        # Enforce rate limiting before making request
        self._enforce_rate_limit()

        delay = initial_delay

        for attempt in range(max_retries + 1):
            try:
                response = self.session.get(self.BASE_URL, params=params, timeout=30)
                response.raise_for_status()
                return response

            except requests.exceptions.HTTPError as e:
                if response.status_code == 429:  # Rate limit
                    logger.warning(f"Rate limited. Retrying in {delay}s...")
                elif response.status_code == 401:
                    raise ValueError(
                        "Invalid API key. Check your ENTSOE_API_KEY."
                    ) from e
                elif attempt < max_retries:
                    logger.warning(
                        f"Request failed (attempt {attempt + 1}/{max_retries + 1}): {e}. Retrying in {delay}s..."
                    )
                else:
                    raise

                if attempt < max_retries:
                    time.sleep(delay)
                    delay *= 2

            except requests.exceptions.RequestException as e:
                if attempt < max_retries:
                    logger.warning(f"Request failed: {e}. Retrying in {delay}s...")
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise

        raise requests.exceptions.RequestException("All retries exhausted")

    def get_usage_stats(self) -> dict:
        """Get API usage statistics.

        Returns:
            dict: Usage statistics including total requests and current rate.
        """
        now = time.time()
        # Clean up old timestamps
        while self.request_timestamps and now - self.request_timestamps[0] > 60:
            self.request_timestamps.popleft()

        return {
            "total_requests": self.total_requests,
            "requests_last_minute": len(self.request_timestamps),
            "rate_limit_rpm": self.rate_limit_rpm,
            "utilization_percent": (
                (len(self.request_timestamps) / self.rate_limit_rpm * 100)
                if self.rate_limit_rpm > 0
                else 0
            ),
        }

    def _parse_datetime(self, dt):
        """Convert datetime to ENTSO-E format (YYYYMMDDHHmm).

        Args:
            dt: Datetime string (YYYY-MM-DD) or datetime object.

        Returns:
            Formatted datetime string.
        """
        if isinstance(dt, str):
            from datetime import datetime as dt_class

            dt = dt_class.fromisoformat(dt)
        return dt.strftime("%Y%m%d%H%M")

    def get_day_ahead_prices(
        self,
        country: str,
        start,
        end,
    ) -> pd.DataFrame:
        """Fetch day-ahead electricity prices.

        Args:
            country: Country code (e.g., "FR", "DE", "ES").
            start: Start date (YYYY-MM-DD or datetime).
            end: End date (YYYY-MM-DD or datetime).

        Returns:
            DataFrame with columns: datetime, price_eur_mwh, country

        Example:
            >>> connector = EntsoeConnector(api_key="YOUR_KEY")
            >>> prices = connector.get_day_ahead_prices("FR", "2022-01-01", "2022-01-31")
            >>> print(prices.head())
                         datetime  price_eur_mwh country
            0 2022-01-01 00:00:00          68.50      FR
            1 2022-01-01 01:00:00          62.30      FR
        """
        if country not in self.COUNTRY_CODES:
            raise ValueError(
                f"Unknown country code: {country}. "
                f"Available: {', '.join(self.COUNTRY_CODES.keys())}"
            )

        # Convert dates to strings for caching
        from datetime import datetime as dt_class

        start_str = start.strftime("%Y-%m-%d") if isinstance(start, dt_class) else start
        end_str = end.strftime("%Y-%m-%d") if isinstance(end, dt_class) else end

        # Try cache first
        if self.use_cache and self.cache:
            cached_data = self.cache.get(
                country=country,
                start_date=start_str,
                end_date=end_str,
                data_type="prices",
            )
            if cached_data is not None:
                return cached_data

        params = {
            "documentType": self.DOC_TYPE_DAY_AHEAD_PRICES,
            "in_Domain": self.COUNTRY_CODES[country],
            "out_Domain": self.COUNTRY_CODES[country],
            "periodStart": self._parse_datetime(start),
            "periodEnd": self._parse_datetime(end),
        }

        logger.info(f"Fetching day-ahead prices for {country} from {start} to {end}")

        try:
            response = self._make_request(params)

            # Parse XML response
            import xml.etree.ElementTree as ET

            root = ET.fromstring(response.content)

            # Debug: log the root tag and namespace
            logger.debug(f"XML root tag: {root.tag}")

            # Extract namespace from root tag (handles different ENTSO-E API versions)
            import re

            namespace_match = re.match(r"\{(.*)\}", root.tag)
            if namespace_match:
                namespace = namespace_match.group(1)
                logger.debug(f"Detected namespace: {namespace}")
            else:
                # Fallback to common namespace
                namespace = "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3"

            ns = {"ns": namespace}

            data = []
            for timeseries in root.findall(".//ns:TimeSeries", ns):
                for period in timeseries.findall(".//ns:Period", ns):
                    start_time = period.find("ns:timeInterval/ns:start", ns)
                    if start_time is None:
                        continue

                    period_start = pd.to_datetime(start_time.text)

                    for point in period.findall("ns:Point", ns):
                        position_elem = point.find("ns:position", ns)
                        price_elem = point.find("ns:price.amount", ns)

                        if position_elem is not None and price_elem is not None:
                            position = int(position_elem.text)
                            price = float(price_elem.text)

                            # Position 1 = first hour of period
                            timestamp = period_start + timedelta(hours=position - 1)

                            data.append(
                                {
                                    "datetime": timestamp,
                                    "price_eur_mwh": price,
                                    "country": country,
                                }
                            )

            df = pd.DataFrame(data)

            if df.empty:
                logger.warning(f"No data found for {country} from {start} to {end}")
                return df

            df = df.sort_values("datetime").reset_index(drop=True)

            logger.info(f"Successfully fetched {len(df)} hourly prices for {country}")

            # Cache the result
            if self.use_cache and self.cache:
                self.cache.set(
                    data=df,
                    country=country,
                    start_date=start_str,
                    end_date=end_str,
                    data_type="prices",
                )

            return df

        except Exception as e:
            logger.error(f"Failed to fetch day-ahead prices: {e}")
            raise

    def get_actual_load(
        self,
        country: str,
        start,
        end,
    ) -> pd.DataFrame:
        """Fetch actual total load (electricity consumption).

        Args:
            country: Country code.
            start: Start date.
            end: End date.

        Returns:
            DataFrame with columns: datetime, load_mw, country
        """
        if country not in self.COUNTRY_CODES:
            raise ValueError(f"Unknown country code: {country}")

        params = {
            "documentType": self.DOC_TYPE_ACTUAL_LOAD,
            "processType": "A16",  # Realised
            "outBiddingZone_Domain": self.COUNTRY_CODES[country],
            "periodStart": self._parse_datetime(start),
            "periodEnd": self._parse_datetime(end),
        }

        logger.info(f"Fetching actual load for {country} from {start} to {end}")

        try:
            response = self._make_request(params)

            import xml.etree.ElementTree as ET
            import re

            root = ET.fromstring(response.content)

            # Extract namespace from root tag
            namespace_match = re.match(r"\{(.*)\}", root.tag)
            if namespace_match:
                namespace = namespace_match.group(1)
            else:
                namespace = "urn:iec62325.351:tc57wg16:451-6:publicationdocument:7:3"

            ns = {"ns": namespace}

            data = []
            for timeseries in root.findall(".//ns:TimeSeries", ns):
                for period in timeseries.findall(".//ns:Period", ns):
                    start_time = period.find("ns:timeInterval/ns:start", ns)
                    if start_time is None:
                        continue

                    period_start = pd.to_datetime(start_time.text)

                    for point in period.findall("ns:Point", ns):
                        position_elem = point.find("ns:position", ns)
                        quantity_elem = point.find("ns:quantity", ns)

                        if position_elem is not None and quantity_elem is not None:
                            position = int(position_elem.text)
                            load = float(quantity_elem.text)

                            timestamp = period_start + timedelta(hours=position - 1)

                            data.append(
                                {
                                    "datetime": timestamp,
                                    "load_mw": load,
                                    "country": country,
                                }
                            )

            df = pd.DataFrame(data)
            df = df.sort_values("datetime").reset_index(drop=True)

            logger.info(
                f"Successfully fetched {len(df)} hourly load values for {country}"
            )
            return df

        except Exception as e:
            logger.error(f"Failed to fetch actual load: {e}")
            raise

    def get_market_data(
        self,
        country: str,
        start,
        end,
    ) -> pd.DataFrame:
        """Fetch both prices and load, merged on datetime.

        Args:
            country: Country code.
            start: Start date.
            end: End date.

        Returns:
            DataFrame with columns: datetime, price_eur_mwh, load_mw, country

        Example:
            >>> connector = EntsoeConnector(api_key="YOUR_KEY")
            >>> data = connector.get_market_data("FR", "2022-01-01", "2022-01-31")
            >>> print(data.head())
        """
        logger.info(f"Fetching complete market data for {country}")

        prices = self.get_day_ahead_prices(country, start, end)
        load = self.get_actual_load(country, start, end)

        # Merge on datetime
        df = pd.merge(prices, load, on=["datetime", "country"], how="inner")

        logger.info(f"Merged dataset: {len(df)} records with both prices and load")

        return df


def main():
    """Example usage - download 1 month of French market data."""
    import os

    # Check if API key exists
    api_key = os.getenv("ENTSOE_API_KEY")
    if not api_key:
        print("[ERROR] ENTSO-E API key not found.")
        print("\nTo use this connector:")
        print("1. Register at https://transparency.entsoe.eu/")
        print("2. Request API key (free, takes ~24h)")
        print("3. Set environment variable: export ENTSOE_API_KEY='your-key-here'")
        print("\nFor now, returning mock example...")
        return

    try:
        connector = EntsoeConnector(api_key=api_key)

        # Download 1 month of data
        print("\n[SYNC] Downloading French market data (Jan 2022)...")
        data = connector.get_market_data(
            country="FR",
            start="2022-01-01",
            end="2022-02-01",
        )

        print(f"\n[OK] Successfully downloaded {len(data)} hourly records")
        print("\nSample data:")
        print(data.head(10))

        print("\nBasic statistics:")
        print(data[["price_eur_mwh", "load_mw"]].describe())

        # Save to CSV
        output_path = "data/raw_data/entsoe_france_jan2022.csv"
        data.to_csv(output_path, index=False)
        print(f"\n[SAVE] Saved to {output_path}")

    except ValueError as e:
        print(f"[ERROR] Configuration error: {e}")
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        logger.exception("Failed to fetch data")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    main()
