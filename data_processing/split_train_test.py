import pandas as pd
from pathlib import Path

# Create the modified data directory if needed
MODIFIED_DIR = Path("data/modified_data")
MODIFIED_DIR.mkdir(parents=True, exist_ok=True)

# Load the merged datasets
merged_daily_path = MODIFIED_DIR / "merged_daily_regional.csv"
merged_hourly_path = MODIFIED_DIR / "merged_hourly_regional.csv"

merged_daily = pd.read_csv(merged_daily_path)
merged_hourly = pd.read_csv(merged_hourly_path)

# Convert date columns to datetime to ensure proper sorting
merged_daily["date"] = pd.to_datetime(merged_daily["date"])
merged_hourly["datetime_hour"] = pd.to_datetime(merged_hourly["datetime_hour"])

# Chronological sorting
merged_daily = merged_daily.sort_values("date")
merged_hourly = merged_hourly.sort_values("datetime_hour")

# 80/20 time-based split without shuffling (to preserve temporal order)
split_idx_daily = int(len(merged_daily) * 0.8)
split_idx_hourly = int(len(merged_hourly) * 0.8)

train_daily = merged_daily.iloc[:split_idx_daily]
test_daily = merged_daily.iloc[split_idx_daily:]

train_hourly = merged_hourly.iloc[:split_idx_hourly]
test_hourly = merged_hourly.iloc[split_idx_hourly:]

# Save split datasets
train_daily.to_csv(MODIFIED_DIR / "train_daily.csv", index=False)
test_daily.to_csv(MODIFIED_DIR / "test_daily.csv", index=False)

train_hourly.to_csv(MODIFIED_DIR / "train_hourly.csv", index=False)
test_hourly.to_csv(MODIFIED_DIR / "test_hourly.csv", index=False)

print("Time-based train/test split completed successfully")
