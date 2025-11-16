"""
API Data Caching System

This module provides a caching layer for ENTSO-E API data to:
- Reduce API calls during development/testing
- Speed up data loading
- Avoid hitting rate limits
- Enable offline development

The cache stores data in CSV files with metadata tracking.
"""

import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class ApiCache:
    """Simple file-based cache for API data."""

    def __init__(self, cache_dir: str = "data/cache/entsoe", ttl_days: int = 7):
        """Initialize cache.

        Args:
            cache_dir: Directory to store cached data
            ttl_days: Time-to-live for cached data in days
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_dir = self.cache_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)

        self.ttl_days = ttl_days

        logger.info(f"Cache initialized at {self.cache_dir} (TTL: {ttl_days} days)")

    def _get_cache_key(self, **kwargs) -> str:
        """Generate cache key from parameters.

        Args:
            **kwargs: Parameters to hash (e.g., country, start_date, end_date, data_type)

        Returns:
            str: Cache key (MD5 hash of sorted parameters)
        """
        # Sort parameters for consistent hashing
        sorted_params = sorted(kwargs.items())
        param_string = json.dumps(sorted_params, default=str)
        return hashlib.md5(param_string.encode()).hexdigest()

    def _get_cache_paths(self, cache_key: str) -> tuple[Path, Path]:
        """Get paths for data and metadata files.

        Args:
            cache_key: Cache key

        Returns:
            tuple: (data_path, metadata_path)
        """
        data_path = self.cache_dir / f"{cache_key}.csv"
        metadata_path = self.metadata_dir / f"{cache_key}.json"
        return data_path, metadata_path

    def get(
        self, country: str, start_date: str, end_date: str, data_type: str = "prices"
    ) -> Optional[pd.DataFrame]:
        """Retrieve data from cache if available and not expired.

        Args:
            country: Country code (e.g., 'FR')
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            data_type: Type of data ('prices', 'load', 'generation')

        Returns:
            pd.DataFrame or None: Cached data if valid, None otherwise
        """
        cache_key = self._get_cache_key(
            country=country, start_date=start_date, end_date=end_date, data_type=data_type
        )

        data_path, metadata_path = self._get_cache_paths(cache_key)

        # Check if cache exists
        if not data_path.exists() or not metadata_path.exists():
            logger.debug(f"Cache miss: {cache_key}")
            return None

        # Load metadata
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            # Check expiration
            cached_time = datetime.fromisoformat(metadata["cached_at"])
            age_days = (datetime.now() - cached_time).days

            if age_days > self.ttl_days:
                logger.info(
                    f"Cache expired: {cache_key} (age: {age_days} days, TTL: {self.ttl_days})"
                )
                return None

            # Load data
            df = pd.read_csv(data_path, parse_dates=["datetime"])
            logger.info(
                f"Cache hit: {cache_key} ({len(df)} records, age: {age_days} days)"
            )
            return df

        except Exception as e:
            logger.warning(f"Failed to load cache {cache_key}: {e}")
            return None

    def set(
        self,
        data: pd.DataFrame,
        country: str,
        start_date: str,
        end_date: str,
        data_type: str = "prices",
    ):
        """Store data in cache.

        Args:
            data: DataFrame to cache
            country: Country code
            start_date: Start date
            end_date: End date
            data_type: Type of data
        """
        cache_key = self._get_cache_key(
            country=country, start_date=start_date, end_date=end_date, data_type=data_type
        )

        data_path, metadata_path = self._get_cache_paths(cache_key)

        try:
            # Save data
            data.to_csv(data_path, index=False)

            # Save metadata
            metadata = {
                "country": country,
                "start_date": start_date,
                "end_date": end_date,
                "data_type": data_type,
                "cached_at": datetime.now().isoformat(),
                "record_count": len(data),
                "cache_key": cache_key,
            }

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Cached {len(data)} records: {cache_key}")

        except Exception as e:
            logger.error(f"Failed to cache data {cache_key}: {e}")

    def clear(self, older_than_days: Optional[int] = None):
        """Clear cache.

        Args:
            older_than_days: If specified, only clear cache older than this many days.
                           If None, clear all cache.
        """
        if older_than_days is None:
            # Clear all cache
            for path in self.cache_dir.glob("*.csv"):
                path.unlink()
            for path in self.metadata_dir.glob("*.json"):
                path.unlink()
            logger.info("Cleared all cache")
        else:
            # Clear old cache
            cutoff_time = datetime.now() - timedelta(days=older_than_days)
            cleared_count = 0

            for metadata_path in self.metadata_dir.glob("*.json"):
                try:
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)

                    cached_time = datetime.fromisoformat(metadata["cached_at"])

                    if cached_time < cutoff_time:
                        # Remove data and metadata
                        cache_key = metadata["cache_key"]
                        data_path, _ = self._get_cache_paths(cache_key)

                        if data_path.exists():
                            data_path.unlink()
                        metadata_path.unlink()

                        cleared_count += 1

                except Exception as e:
                    logger.warning(f"Error clearing cache {metadata_path}: {e}")

            logger.info(f"Cleared {cleared_count} old cache entries (>{older_than_days} days)")

    def get_stats(self) -> dict:
        """Get cache statistics.

        Returns:
            dict: Cache statistics (total entries, total size, oldest/newest)
        """
        data_files = list(self.cache_dir.glob("*.csv"))
        total_size_mb = sum(f.stat().st_size for f in data_files) / (1024 * 1024)

        metadata_files = list(self.metadata_dir.glob("*.json"))
        cached_times = []

        for metadata_path in metadata_files:
            try:
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                cached_times.append(datetime.fromisoformat(metadata["cached_at"]))
            except Exception:
                pass

        return {
            "total_entries": len(data_files),
            "total_size_mb": round(total_size_mb, 2),
            "oldest_entry": min(cached_times).isoformat() if cached_times else None,
            "newest_entry": max(cached_times).isoformat() if cached_times else None,
        }


def main():
    """Example usage and cache management."""
    import argparse

    parser = argparse.ArgumentParser(description="Manage API cache")
    parser.add_argument("--stats", action="store_true", help="Show cache statistics")
    parser.add_argument(
        "--clear", type=int, nargs="?", const=-1, help="Clear cache (optionally older than N days)"
    )
    args = parser.parse_args()

    cache = ApiCache()

    if args.stats:
        stats = cache.get_stats()
        print("\n📊 Cache Statistics:")
        print(f"  Total entries: {stats['total_entries']}")
        print(f"  Total size: {stats['total_size_mb']} MB")
        print(f"  Oldest entry: {stats['oldest_entry']}")
        print(f"  Newest entry: {stats['newest_entry']}")

    if args.clear is not None:
        if args.clear == -1:
            print("\n🧹 Clearing all cache...")
            cache.clear()
        else:
            print(f"\n🧹 Clearing cache older than {args.clear} days...")
            cache.clear(older_than_days=args.clear)

        print("✅ Cache cleared")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
