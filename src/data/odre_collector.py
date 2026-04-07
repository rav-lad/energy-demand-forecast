"""
ODRE Energy Consumption Data Collector

Collects French regional energy consumption data from ODRE
(French Energy Network Data Observatory).

Features:
- Automatic download from ODRE API (no API key needed)
- Hourly electricity and gas consumption by region
- Retry logic with exponential backoff
- Data validation and cleaning

API: https://odre.opendatasoft.com/
Dataset: consommation-quotidienne-brute-regionale

Usage:
    python data_collection/odre_collector.py \
        --start_date 2022-01-01 \
        --end_date 2024-12-31 \
        --output data/raw_data/energy/
"""

import argparse
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)


class OdreCollector:
    """Collector for ODRE energy consumption data."""

    BASE_URL = "https://odre.opendatasoft.com/api/records/1.0/search/"
    DATASET_ID = "consommation-quotidienne-brute-regionale"

    # French regions (INSEE codes)
    REGIONS = {
        11: "Île-de-France",
        24: "Centre-Val de Loire",
        27: "Bourgogne-Franche-Comté",
        28: "Normandie",
        32: "Hauts-de-France",
        44: "Grand Est",
        52: "Pays de la Loire",
        53: "Bretagne",
        75: "Nouvelle-Aquitaine",
        76: "Occitanie",
        84: "Auvergne-Rhône-Alpes",
        93: "Provence-Alpes-Côte d'Azur",
        94: "Corse",
    }

    def __init__(self):
        """Initialize ODRE collector."""
        self.session = requests.Session()

    def fetch_data(
        self,
        start_date: str,
        end_date: str,
        max_retries: int = 3,
    ) -> pd.DataFrame:
        """Fetch energy consumption data from ODRE.

        ODRE API has pagination limits, so we split large requests into monthly chunks.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            max_retries: Maximum retry attempts

        Returns:
            pd.DataFrame: Energy consumption data

        Raises:
            Exception: If all retries fail
        """
        logger.info(f"Fetching ODRE data from {start_date} to {end_date}")

        # Convert to datetime for chunking
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)

        # Calculate number of days
        num_days = (end_dt - start_dt).days

        # If period is small enough, fetch directly
        # ODRE has ~13 regions, so 10000 rows = ~769 days max
        if num_days <= 700:
            return self._fetch_data_chunk(start_date, end_date, max_retries)

        # Otherwise, split into monthly chunks
        logger.info(
            f"Large date range detected ({num_days} days). Splitting into monthly chunks..."
        )

        all_data = []
        current_start = start_dt

        while current_start < end_dt:
            # Fetch 1 month at a time
            current_end = min(current_start + pd.DateOffset(months=1), end_dt)

            chunk_start = current_start.strftime("%Y-%m-%d")
            chunk_end = current_end.strftime("%Y-%m-%d")

            logger.info(f"  Fetching {chunk_start} to {chunk_end}...")

            df_chunk = self._fetch_data_chunk(chunk_start, chunk_end, max_retries)

            if not df_chunk.empty:
                all_data.append(df_chunk)
                logger.info(f"     Collected {len(df_chunk)} records")

            current_start = current_end

        if not all_data:
            logger.warning("No data collected")
            return pd.DataFrame()

        # Combine all chunks
        df = pd.concat(all_data, ignore_index=True)
        df = df.sort_values("date").reset_index(drop=True)

        logger.info(f" Total: {len(df)} records collected")
        return df

    def _fetch_data_chunk(
        self,
        start_date: str,
        end_date: str,
        max_retries: int = 3,
    ) -> pd.DataFrame:
        """Fetch a single chunk of data (internal method).

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            max_retries: Maximum retry attempts

        Returns:
            pd.DataFrame: Energy consumption data for this chunk
        """
        # ODRE API parameters
        params = {
            "dataset": self.DATASET_ID,
            "q": f"date:[{start_date} TO {end_date}]",
            "rows": 10000,  # Max rows per request
            "sort": "date",
            "facet": ["date", "code_insee_region"],
        }

        all_records = []
        start_index = 0

        # Paginate through results (but limit to avoid API errors)
        while True:
            params["start"] = start_index

            # Fetch with retry logic
            response_data = self._fetch_with_retry(params, max_retries)

            if response_data is None:
                logger.warning(f"Failed to fetch data for {start_date} to {end_date}")
                break

            records = response_data.get("records", [])

            if not records:
                break  # No more data

            all_records.extend(records)

            # Check if we got all records
            nhits = response_data.get("nhits", 0)

            if len(all_records) >= nhits:
                break

            # Update start index for next page
            start_index += len(records)

            # Safety check: if we're about to exceed 10000 start index, stop
            # This prevents the API error we were getting
            if start_index >= 10000:
                logger.warning(
                    f"Reached pagination limit at {start_index} records. Consider splitting date range further."
                )
                break

            # Rate limiting (be nice to the API)
            time.sleep(0.5)

        logger.info(f"Successfully fetched {len(all_records)} total records")

        # Parse records into DataFrame
        df = self._parse_records(all_records)

        return df

    def _fetch_with_retry(
        self,
        params: dict,
        max_retries: int = 3,
        initial_delay: float = 1.0,
    ) -> Optional[dict]:
        """Fetch data with exponential backoff retry.

        Args:
            params: Query parameters
            max_retries: Maximum retry attempts
            initial_delay: Initial delay before retry

        Returns:
            dict: Response data or None if failed
        """
        delay = initial_delay

        for attempt in range(max_retries + 1):
            try:
                response = self.session.get(
                    self.BASE_URL,
                    params=params,
                    timeout=30,
                )
                response.raise_for_status()
                return response.json()

            except requests.exceptions.RequestException as e:
                if attempt < max_retries:
                    logger.warning(
                        f"Request failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    logger.error(f"All retries failed: {e}")
                    return None

        return None

    def _parse_records(self, records: list) -> pd.DataFrame:
        """Parse ODRE records into DataFrame.

        Args:
            records: List of ODRE records

        Returns:
            pd.DataFrame: Parsed data
        """
        data = []

        for record in records:
            fields = record.get("fields", {})

            # Extract relevant fields
            row = {
                "date": fields.get("date"),
                "datetime_heure": fields.get("date_heure"),
                "code_insee_region": fields.get("code_insee_region"),
                "region": fields.get("libelle_region"),
                # Electricity consumption (MW)
                "conso_elec_mw": fields.get("consommation_brute_electricite_rte"),
                # Gas consumption (MW PCS 0°C)
                "conso_gaz_mw": fields.get("consommation_brute_gaz_totale"),
                # Temperature (°C)
                "temperature": fields.get("tco_thermique"),
            }

            data.append(row)

        df = pd.DataFrame(data)

        # Convert to appropriate types
        df["date"] = pd.to_datetime(df["date"])
        if "datetime_heure" in df.columns and df["datetime_heure"].notna().any():
            df["datetime_heure"] = pd.to_datetime(df["datetime_heure"], errors="coerce")

        # Convert numeric columns
        numeric_cols = ["conso_elec_mw", "conso_gaz_mw", "temperature"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Convert region code to int
        df["code_insee_region"] = pd.to_numeric(
            df["code_insee_region"], errors="coerce"
        ).astype("Int64")

        # Sort by date and region
        df = df.sort_values(["date", "code_insee_region"]).reset_index(drop=True)

        logger.info(f"Parsed {len(df)} records into DataFrame")
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"Regions: {sorted(df['code_insee_region'].dropna().unique())}")

        return df

    def save_data(self, df: pd.DataFrame, output_path: str):
        """Save data to CSV.

        Args:
            df: DataFrame to save
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df.to_csv(output_path, index=False, encoding="utf-8")
        logger.info(f"Saved data to {output_path}")

        # Log statistics
        logger.info(f"\nData Statistics:")
        logger.info(f"  Total records: {len(df)}")
        logger.info(
            f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}"
        )
        logger.info(f"  Regions: {len(df['code_insee_region'].unique())}")

        if "conso_elec_mw" in df.columns:
            logger.info(
                f"  Electricity consumption (MW):"
                f"\n    Mean: {df['conso_elec_mw'].mean():.0f}"
                f"\n    Min:  {df['conso_elec_mw'].min():.0f}"
                f"\n    Max:  {df['conso_elec_mw'].max():.0f}"
            )

    def validate_data(self, df: pd.DataFrame) -> dict:
        """Validate data quality.

        Args:
            df: DataFrame to validate

        Returns:
            dict: Validation results
        """
        issues = []
        warnings = []

        # Check for missing values
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)

        for col in ["conso_elec_mw", "conso_gaz_mw"]:
            if col in df.columns and missing_pct[col] > 5:
                issues.append(f"{col}: {missing_pct[col]:.1f}% missing values")
            elif col in df.columns and missing_pct[col] > 0:
                warnings.append(f"{col}: {missing_pct[col]:.1f}% missing values")

        # Check for negative values
        for col in ["conso_elec_mw", "conso_gaz_mw"]:
            if col in df.columns:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    issues.append(f"{col}: {negative_count} negative values")

        # Check date continuity
        if "date" in df.columns:
            date_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
            expected_dates = len(date_range) * len(self.REGIONS)
            actual_dates = len(df)

            if actual_dates < expected_dates * 0.95:  # Allow 5% missing
                warnings.append(
                    f"Date continuity: {actual_dates} / {expected_dates} records "
                    f"({actual_dates/expected_dates*100:.1f}%)"
                )

        # Print validation report
        print("\n" + "=" * 70)
        print("DATA VALIDATION REPORT")
        print("=" * 70)

        if not issues and not warnings:
            print("\n[OK] All checks passed!")
        else:
            if issues:
                print("\n[ERROR] Issues:")
                for issue in issues:
                    print(f"  - {issue}")

            if warnings:
                print("\n[WARNING]  Warnings:")
                for warning in warnings:
                    print(f"  - {warning}")

        print("=" * 70 + "\n")

        return {"issues": issues, "warnings": warnings}


def main():
    """CLI for ODRE data collection."""
    parser = argparse.ArgumentParser(
        description="Collect French energy consumption data from ODRE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect 1 year of data
  python data_collection/odre_collector.py \\
    --start_date 2023-01-01 --end_date 2023-12-31

  # Collect recent data
  python data_collection/odre_collector.py \\
    --start_date 2024-01-01 --end_date 2024-12-31 \\
    --output data/raw_data/energy/conso_2024.csv

Data Source: ODRE (French Energy Network Data Observatory)
API: https://odre.opendatasoft.com/
No API key required - Free and open access
        """,
    )

    parser.add_argument(
        "--start_date",
        type=str,
        default="2022-01-01",
        help="Start date (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--end_date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="End date (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/raw_data/energy/odre_consumption.csv",
        help="Output CSV file path",
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run data validation after collection",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logger.info("=" * 70)
    logger.info("ODRE ENERGY CONSUMPTION DATA COLLECTION")
    logger.info("=" * 70)
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    logger.info(f"Output: {args.output}")
    logger.info("=" * 70)

    # Collect data
    collector = OdreCollector()

    try:
        df = collector.fetch_data(args.start_date, args.end_date)

        # Save data
        collector.save_data(df, args.output)

        # Validate if requested
        if args.validate:
            collector.validate_data(df)

        logger.info("=" * 70)
        logger.info("[OK] COLLECTION COMPLETED SUCCESSFULLY")
        logger.info("=" * 70)

    except Exception as e:
        logger.error(f"[ERROR] Collection failed: {e}")
        import traceback

        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
