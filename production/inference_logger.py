"""Inference logging for prediction audit trail.

Stores all predictions, signals, and positions in SQLite for:
- Post-hoc analysis and debugging
- Regulatory audit trail
- Performance attribution
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "inference_log.db"


class InferenceLogger:
    """SQLite-based inference audit trail."""

    def __init__(self, db_path: str = None):
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    prediction_date TEXT NOT NULL,
                    model_name TEXT,
                    predicted_price REAL,
                    actual_price REAL,
                    error REAL,
                    features_json TEXT,
                    metadata_json TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trading_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    signal_date TEXT NOT NULL,
                    signal_value REAL,
                    position REAL,
                    pnl_gross REAL,
                    pnl_net REAL,
                    metadata_json TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    model_name TEXT,
                    metric_name TEXT,
                    metric_value REAL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_pred_date
                ON predictions(prediction_date)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_signal_date
                ON trading_signals(signal_date)
            """)

    def log_prediction(
        self,
        prediction_date: str,
        predicted_price: float,
        actual_price: float = None,
        model_name: str = "ensemble",
        features: Dict = None,
        metadata: Dict = None,
    ):
        """Log a single prediction."""
        error = (predicted_price - actual_price) if actual_price is not None else None
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """INSERT INTO predictions
                   (timestamp, prediction_date, model_name, predicted_price,
                    actual_price, error, features_json, metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now().isoformat(),
                    prediction_date,
                    model_name,
                    predicted_price,
                    actual_price,
                    error,
                    json.dumps(features) if features else None,
                    json.dumps(metadata) if metadata else None,
                ),
            )

    def log_predictions(self, predictions_df: pd.DataFrame, metrics: Dict = None):
        """Log a batch of predictions from walk-forward output.

        Args:
            predictions_df: DataFrame with datetime_hour, actual, predicted.
            metrics: Overall model metrics to store.
        """
        with sqlite3.connect(str(self.db_path)) as conn:
            now = datetime.now().isoformat()
            rows = []
            for _, row in predictions_df.iterrows():
                rows.append((
                    now,
                    str(row.get("datetime_hour", "")),
                    "ensemble",
                    float(row.get("predicted", 0)),
                    float(row.get("actual", 0)),
                    float(row.get("predicted", 0) - row.get("actual", 0)),
                    None,
                    None,
                ))
            conn.executemany(
                """INSERT INTO predictions
                   (timestamp, prediction_date, model_name, predicted_price,
                    actual_price, error, features_json, metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )

            if metrics:
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        conn.execute(
                            """INSERT INTO model_metrics
                               (timestamp, model_name, metric_name, metric_value)
                               VALUES (?, ?, ?, ?)""",
                            (now, "ensemble", k, float(v)),
                        )

        logger.info(f"Logged {len(predictions_df)} predictions to {self.db_path}")

    def log_trading_signal(
        self,
        signal_date: str,
        signal_value: float,
        position: float,
        pnl_gross: float = None,
        pnl_net: float = None,
        metadata: Dict = None,
    ):
        """Log a trading signal and position."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """INSERT INTO trading_signals
                   (timestamp, signal_date, signal_value, position,
                    pnl_gross, pnl_net, metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now().isoformat(),
                    signal_date,
                    signal_value,
                    position,
                    pnl_gross,
                    pnl_net,
                    json.dumps(metadata) if metadata else None,
                ),
            )

    def get_recent_predictions(self, days: int = 7) -> pd.DataFrame:
        """Retrieve recent predictions for monitoring."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        with sqlite3.connect(str(self.db_path)) as conn:
            df = pd.read_sql_query(
                "SELECT * FROM predictions WHERE timestamp >= ? ORDER BY prediction_date",
                conn,
                params=(cutoff,),
            )
        return df

    def get_recent_signals(self, days: int = 7) -> pd.DataFrame:
        """Retrieve recent trading signals."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        with sqlite3.connect(str(self.db_path)) as conn:
            df = pd.read_sql_query(
                "SELECT * FROM trading_signals WHERE timestamp >= ? ORDER BY signal_date",
                conn,
                params=(cutoff,),
            )
        return df

    def get_metrics_history(self, metric_name: str = None) -> pd.DataFrame:
        """Retrieve model metrics history."""
        with sqlite3.connect(str(self.db_path)) as conn:
            if metric_name:
                df = pd.read_sql_query(
                    "SELECT * FROM model_metrics WHERE metric_name = ? ORDER BY timestamp",
                    conn,
                    params=(metric_name,),
                )
            else:
                df = pd.read_sql_query(
                    "SELECT * FROM model_metrics ORDER BY timestamp", conn
                )
        return df

    def archive_old_data(self, days_to_keep: int = 90):
        """Archive data older than N days."""
        cutoff = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
        with sqlite3.connect(str(self.db_path)) as conn:
            for table in ["predictions", "trading_signals", "model_metrics"]:
                deleted = conn.execute(
                    f"DELETE FROM {table} WHERE timestamp < ?", (cutoff,)
                ).rowcount
                if deleted > 0:
                    logger.info(f"Archived {deleted} rows from {table}")
            conn.execute("VACUUM")
