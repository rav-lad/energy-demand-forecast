"""
Entry point for the trading backtest pipeline.

Delegates to src/backtest/trading_pipeline.py — do not duplicate logic here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.trading_pipeline import main

if __name__ == "__main__":
    sys.exit(main())
