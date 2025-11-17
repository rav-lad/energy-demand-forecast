"""
Timezone Handler for Energy Market Data

Handles timezone conversions and DST transitions for European electricity markets.

Key challenges:
- ENTSO-E data is in UTC
- Local market time is CET/CEST (Europe/Paris for France)
- DST transitions create 23-hour and 25-hour days
- Hour 2A (02:00→03:00 spring) and hour 2B (03:00→02:00 fall)

Usage:
    handler = TimezoneHandler(market_timezone="Europe/Paris")
    df = handler.convert_to_market_time(df_utc, datetime_col="datetime_hour")
    df = handler.handle_dst_transitions(df)
"""

import logging
from datetime import datetime
from typing import Optional

import pandas as pd
import pytz

logger = logging.getLogger(__name__)


class TimezoneHandler:
    """
    Handle timezone conversions and DST transitions for energy market data.

    Electricity markets operate in local time (CET/CEST for Europe),
    but APIs often return UTC. This handler ensures correct conversion.
    """

    def __init__(self, market_timezone: str = "Europe/Paris"):
        """
        Initialize timezone handler.

        Args:
            market_timezone: Market timezone (default: Europe/Paris for France)
        """
        self.market_timezone = market_timezone
        self.market_tz = pytz.timezone(market_timezone)
        self.utc_tz = pytz.UTC

        logger.info(f"Initialized TimezoneHandler with market TZ: {market_timezone}")

    def convert_to_utc(
        self, df: pd.DataFrame, datetime_col: str = "datetime_hour"
    ) -> pd.DataFrame:
        """
        Convert datetime column from market timezone to UTC.

        Args:
            df: DataFrame with datetime column
            datetime_col: Name of datetime column

        Returns:
            DataFrame with UTC datetimes
        """
        df = df.copy()

        # Parse datetime if not already datetime type
        if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
            df[datetime_col] = pd.to_datetime(df[datetime_col])

        # Localize to market timezone if naive
        if df[datetime_col].dt.tz is None:
            logger.debug(f"Localizing naive datetimes to {self.market_timezone}")
            df[datetime_col] = df[datetime_col].dt.tz_localize(
                self.market_tz, ambiguous="infer", nonexistent="shift_forward"
            )

        # Convert to UTC
        df[datetime_col] = df[datetime_col].dt.tz_convert(self.utc_tz)

        logger.debug(f"Converted {len(df)} timestamps to UTC")
        return df

    def convert_to_market_time(
        self, df: pd.DataFrame, datetime_col: str = "datetime_hour"
    ) -> pd.DataFrame:
        """
        Convert datetime column from UTC to market timezone.

        Args:
            df: DataFrame with UTC datetime column
            datetime_col: Name of datetime column

        Returns:
            DataFrame with market timezone datetimes
        """
        df = df.copy()

        # Parse datetime if not already datetime type
        if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
            df[datetime_col] = pd.to_datetime(df[datetime_col])

        # Ensure UTC timezone
        if df[datetime_col].dt.tz is None:
            logger.debug("Assuming naive datetimes are UTC")
            df[datetime_col] = df[datetime_col].dt.tz_localize(self.utc_tz)
        elif df[datetime_col].dt.tz != self.utc_tz:
            logger.warning(
                f"Datetimes are in {df[datetime_col].dt.tz}, converting to UTC first"
            )
            df[datetime_col] = df[datetime_col].dt.tz_convert(self.utc_tz)

        # Convert to market timezone
        df[datetime_col] = df[datetime_col].dt.tz_convert(self.market_tz)

        logger.debug(f"Converted {len(df)} timestamps to {self.market_timezone}")
        return df

    def handle_dst_transitions(
        self, df: pd.DataFrame, datetime_col: str = "datetime_hour"
    ) -> pd.DataFrame:
        """
        Handle DST transitions (23-hour and 25-hour days).

        In spring (23-hour day):
        - Hour 2A (02:00) jumps to 03:00
        - Missing hour is skipped

        In fall (25-hour day):
        - Hour 2B (02:00) repeats when clocks go back
        - Two rows for the repeated hour

        Args:
            df: DataFrame with market timezone datetimes
            datetime_col: Name of datetime column

        Returns:
            DataFrame with DST transitions handled
        """
        df = df.copy()

        # Ensure datetime type
        if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
            df[datetime_col] = pd.to_datetime(df[datetime_col])

        # Check for DST transitions
        dst_info = self._detect_dst_transitions(df, datetime_col)

        if dst_info["spring_transitions"] or dst_info["fall_transitions"]:
            logger.info(
                f"Found {len(dst_info['spring_transitions'])} spring DST transitions "
                f"and {len(dst_info['fall_transitions'])} fall DST transitions"
            )

        # Mark DST transitions
        df["is_dst_transition"] = False
        df["dst_type"] = None

        for spring_date in dst_info["spring_transitions"]:
            mask = df[datetime_col].dt.date == spring_date.date()
            df.loc[mask, "is_dst_transition"] = True
            df.loc[mask, "dst_type"] = "spring_forward"

        for fall_date in dst_info["fall_transitions"]:
            mask = df[datetime_col].dt.date == fall_date.date()
            df.loc[mask, "is_dst_transition"] = True
            df.loc[mask, "dst_type"] = "fall_back"

        return df

    def _detect_dst_transitions(
        self, df: pd.DataFrame, datetime_col: str
    ) -> dict:
        """
        Detect DST transition dates in the data.

        Returns:
            dict with 'spring_transitions' and 'fall_transitions' lists
        """
        timestamps = df[datetime_col].sort_values()

        spring_transitions = []
        fall_transitions = []

        # Look for time jumps (spring) or duplicates (fall)
        for i in range(1, len(timestamps)):
            time_diff = (timestamps.iloc[i] - timestamps.iloc[i - 1]).total_seconds()

            # Spring forward: 2-hour jump (missing hour)
            if time_diff == 7200:  # 2 hours
                spring_transitions.append(timestamps.iloc[i])

            # Fall back: 0-hour jump (repeated hour)
            elif time_diff == 0:
                fall_transitions.append(timestamps.iloc[i])

        return {
            "spring_transitions": spring_transitions,
            "fall_transitions": fall_transitions,
        }

    def create_hourly_index(
        self, start_date: str, end_date: str, include_dst: bool = True
    ) -> pd.DatetimeIndex:
        """
        Create a complete hourly index handling DST transitions.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            include_dst: Whether to include DST transition handling

        Returns:
            DatetimeIndex with hourly timestamps
        """
        # Create range in UTC first
        start = pd.Timestamp(start_date, tz=self.utc_tz)
        end = pd.Timestamp(end_date, tz=self.utc_tz)

        # Create hourly range
        hourly_index = pd.date_range(start=start, end=end, freq="H")

        if include_dst:
            # Convert to market timezone (handles DST automatically)
            hourly_index = hourly_index.tz_convert(self.market_tz)

        logger.info(
            f"Created hourly index from {start_date} to {end_date}: "
            f"{len(hourly_index)} timestamps"
        )

        return hourly_index

    def validate_hourly_data(
        self, df: pd.DataFrame, datetime_col: str = "datetime_hour"
    ) -> dict:
        """
        Validate that hourly data is complete (accounting for DST).

        Args:
            df: DataFrame with hourly data
            datetime_col: Name of datetime column

        Returns:
            Validation report dict
        """
        df = df.copy().sort_values(datetime_col)
        timestamps = df[datetime_col]

        # Calculate time differences
        time_diffs = timestamps.diff().dt.total_seconds() / 3600  # Hours

        # Expected: 1 hour (normal), 2 hours (spring DST), 0 hours (fall DST)
        normal_gaps = (time_diffs == 1.0).sum()
        spring_dst = (time_diffs == 2.0).sum()
        fall_dst = (time_diffs == 0.0).sum()
        other_gaps = (
            (time_diffs > 2.0) | ((time_diffs < 1.0) & (time_diffs > 0.0))
        ).sum()

        # Duplicates
        duplicates = timestamps.duplicated().sum()

        report = {
            "total_records": len(df),
            "normal_hourly_gaps": normal_gaps,
            "spring_dst_transitions": spring_dst,
            "fall_dst_transitions": fall_dst,
            "unexpected_gaps": other_gaps,
            "duplicates": duplicates,
            "is_valid": (other_gaps == 0) and (duplicates == 0),
        }

        # Log report
        if report["is_valid"]:
            logger.info(f"✅ Hourly data validation passed: {len(df)} records")
        else:
            logger.warning(
                f"⚠️ Hourly data validation issues: "
                f"{other_gaps} unexpected gaps, {duplicates} duplicates"
            )

        return report

    def align_to_market_hours(
        self, df: pd.DataFrame, datetime_col: str = "datetime_hour"
    ) -> pd.DataFrame:
        """
        Ensure all timestamps are aligned to market trading hours.

        For day-ahead market:
        - Prices published at 12:30 CET for next day delivery
        - Delivery starts at 00:00 CET

        Args:
            df: DataFrame with datetime column
            datetime_col: Name of datetime column

        Returns:
            DataFrame with market-aligned timestamps
        """
        df = df.copy()

        # Round to nearest hour (in case of sub-hourly data)
        df[datetime_col] = df[datetime_col].dt.floor("H")

        # Remove duplicates (keep first)
        df = df.drop_duplicates(subset=[datetime_col], keep="first")

        # Sort by datetime
        df = df.sort_values(datetime_col).reset_index(drop=True)

        logger.info(f"Aligned {len(df)} timestamps to market hours")

        return df


# Utility functions
def standardize_timezone(
    df: pd.DataFrame,
    datetime_col: str = "datetime_hour",
    source_tz: str = "UTC",
    target_tz: str = "UTC",
) -> pd.DataFrame:
    """
    Standardize timezone for a DataFrame.

    Args:
        df: DataFrame
        datetime_col: Datetime column name
        source_tz: Source timezone (if naive)
        target_tz: Target timezone

    Returns:
        DataFrame with standardized timezone
    """
    df = df.copy()

    if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        df[datetime_col] = pd.to_datetime(df[datetime_col])

    # Localize if naive
    if df[datetime_col].dt.tz is None:
        df[datetime_col] = df[datetime_col].dt.tz_localize(source_tz)

    # Convert to target timezone
    df[datetime_col] = df[datetime_col].dt.tz_convert(target_tz)

    return df


# Example usage
if __name__ == "__main__":
    # Initialize handler
    handler = TimezoneHandler(market_timezone="Europe/Paris")

    # Create sample UTC data
    dates_utc = pd.date_range("2024-03-30", periods=72, freq="H", tz="UTC")
    df_utc = pd.DataFrame(
        {"datetime_hour": dates_utc, "price": range(len(dates_utc))}
    )

    print("\n" + "=" * 80)
    print("TIMEZONE CONVERSION EXAMPLE")
    print("=" * 80)

    print(f"\n📅 Original data (UTC):")
    print(df_utc.head())

    # Convert to market time
    df_market = handler.convert_to_market_time(df_utc)
    print(f"\n📅 Converted to {handler.market_timezone}:")
    print(df_market.head())

    # Handle DST transitions
    df_dst = handler.handle_dst_transitions(df_market)
    print(f"\n📅 DST transitions detected:")
    dst_transitions = df_dst[df_dst["is_dst_transition"]]
    if len(dst_transitions) > 0:
        print(dst_transitions[["datetime_hour", "dst_type"]])
    else:
        print("No DST transitions in this period")

    # Validate
    validation = handler.validate_hourly_data(df_market)
    print(f"\n✅ Validation report:")
    for key, value in validation.items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 80)
