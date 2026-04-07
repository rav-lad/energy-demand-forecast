"""
Market data download — delegates to scripts/download_data.py.
Do not duplicate logic here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.download_data import (
    download_all_market_data,
    print_data_summary,
    START_DATE,
    END_DATE,
    ENTSOE_API_KEY,
)

if __name__ == "__main__":
    results = download_all_market_data(START_DATE, END_DATE, ENTSOE_API_KEY)
    print_data_summary(results)
