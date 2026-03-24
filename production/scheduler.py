"""Production scheduler for automated daily pipeline execution.

Uses APScheduler with cron triggers to orchestrate:
- Data fetching (06:00 CET)
- Model training + inference (07:00 CET)
- Trading signal generation (08:00 CET)
- Report generation (09:00 CET)
- Health checks (every 15 minutes)

Usage:
    python production/scheduler.py              # Run scheduler
    python production/scheduler.py --dry-run    # Test without scheduling
"""

import argparse
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

# Pipeline state
_pipeline_state = {
    "last_fetch": None,
    "last_train": None,
    "last_trade": None,
    "last_report": None,
    "errors": [],
}


def _get_alert_service():
    """Initialize alert service (lazy)."""
    from production.alert_service import AlertService
    return AlertService()


def job_fetch_data():
    """Job: Fetch latest data from all sources."""
    logger.info("=" * 60)
    logger.info("JOB: FETCH DATA")
    logger.info("=" * 60)

    try:
        from production.data_fetcher import fetch_all

        alert_svc = _get_alert_service()
        results = fetch_all(alert_service=alert_svc)

        _pipeline_state["last_fetch"] = datetime.now().isoformat()
        logger.info(f"Data fetch completed: {results}")

    except Exception as e:
        logger.error(f"Data fetch failed: {e}")
        _pipeline_state["errors"].append(
            {"job": "fetch_data", "time": datetime.now().isoformat(), "error": str(e)}
        )
        _get_alert_service().alert_pipeline_error("fetch_data", str(e))


def job_train_and_predict():
    """Job: Train models and generate predictions."""
    logger.info("=" * 60)
    logger.info("JOB: TRAIN AND PREDICT")
    logger.info("=" * 60)

    try:
        from production.daily_pipeline import load_data, train_ensemble

        df, feature_cols = load_data()
        predictions_df, metrics = train_ensemble(df, feature_cols)

        # Log to inference logger
        try:
            from production.inference_logger import InferenceLogger
            inf_logger = InferenceLogger()
            inf_logger.log_predictions(predictions_df, metrics)
        except ImportError:
            logger.debug("Inference logger not available")

        _pipeline_state["last_train"] = datetime.now().isoformat()
        logger.info(f"Training completed: MAE={metrics.get('mae', 'N/A')}")

    except Exception as e:
        logger.error(f"Train/predict failed: {e}")
        _pipeline_state["errors"].append(
            {"job": "train_predict", "time": datetime.now().isoformat(), "error": str(e)}
        )
        _get_alert_service().alert_pipeline_error("train_predict", str(e))


def job_execute_trading():
    """Job: Generate trading signals and positions."""
    logger.info("=" * 60)
    logger.info("JOB: EXECUTE TRADING")
    logger.info("=" * 60)

    try:
        from production.daily_pipeline import load_data, train_ensemble, execute_trading

        df, feature_cols = load_data()
        predictions_df, metrics = train_ensemble(df, feature_cols)
        trades_df, trading_metrics = execute_trading(predictions_df)

        # Check for drawdown alert
        max_dd = trading_metrics.get("max_drawdown", 0)
        if max_dd < -1000:
            _get_alert_service().alert_drawdown(max_dd, -1000)

        _pipeline_state["last_trade"] = datetime.now().isoformat()
        logger.info(f"Trading completed: PnL={trading_metrics.get('total_pnl', 'N/A')}")

    except Exception as e:
        logger.error(f"Trading execution failed: {e}")
        _pipeline_state["errors"].append(
            {"job": "trading", "time": datetime.now().isoformat(), "error": str(e)}
        )
        _get_alert_service().alert_pipeline_error("trading", str(e))


def job_generate_report():
    """Job: Generate daily performance report."""
    logger.info("=" * 60)
    logger.info("JOB: GENERATE REPORT")
    logger.info("=" * 60)

    try:
        from production.daily_pipeline import main as run_full_pipeline

        run_full_pipeline()
        _pipeline_state["last_report"] = datetime.now().isoformat()
        logger.info("Report generated successfully")

    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        _pipeline_state["errors"].append(
            {"job": "report", "time": datetime.now().isoformat(), "error": str(e)}
        )


def job_health_check():
    """Job: Check pipeline health and data freshness."""
    now = datetime.now()
    status = {
        "timestamp": now.isoformat(),
        "state": _pipeline_state,
    }

    # Check data freshness
    last_fetch = _pipeline_state.get("last_fetch")
    if last_fetch:
        hours_since = (now - datetime.fromisoformat(last_fetch)).total_seconds() / 3600
        if hours_since > 26:  # More than 26h since last fetch
            logger.warning(f"Data may be stale: last fetch was {hours_since:.1f}h ago")

    # Check recent errors
    recent_errors = [
        e for e in _pipeline_state.get("errors", [])
        if (now - datetime.fromisoformat(e["time"])).total_seconds() < 3600
    ]
    if len(recent_errors) >= 3:
        logger.error(f"Multiple recent errors: {len(recent_errors)} in last hour")
        _get_alert_service().alert_pipeline_error(
            "health_check",
            f"{len(recent_errors)} errors in the last hour",
        )

    logger.info(f"Health check OK: {status}")


def run_scheduler():
    """Start the APScheduler with all jobs."""
    from apscheduler.schedulers.blocking import BlockingScheduler
    from apscheduler.triggers.cron import CronTrigger

    scheduler = BlockingScheduler(timezone="Europe/Paris")

    # Daily jobs
    scheduler.add_job(
        job_fetch_data,
        CronTrigger(hour=6, minute=0),
        id="fetch_data",
        name="Fetch latest market data",
        max_instances=1,
        misfire_grace_time=1800,
    )
    scheduler.add_job(
        job_train_and_predict,
        CronTrigger(hour=7, minute=0),
        id="train_predict",
        name="Train models and predict",
        max_instances=1,
        misfire_grace_time=1800,
    )
    scheduler.add_job(
        job_execute_trading,
        CronTrigger(hour=8, minute=0),
        id="execute_trading",
        name="Execute trading strategy",
        max_instances=1,
        misfire_grace_time=1800,
    )
    scheduler.add_job(
        job_generate_report,
        CronTrigger(hour=9, minute=0),
        id="generate_report",
        name="Generate daily report",
        max_instances=1,
        misfire_grace_time=1800,
    )

    # Health check every 15 minutes
    scheduler.add_job(
        job_health_check,
        CronTrigger(minute="*/15"),
        id="health_check",
        name="Pipeline health check",
        max_instances=1,
    )

    # Graceful shutdown
    def shutdown(signum, frame):
        logger.info("Received shutdown signal, stopping scheduler...")
        scheduler.shutdown(wait=False)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    logger.info("=" * 60)
    logger.info("ENERGY TRADING SCHEDULER STARTED")
    logger.info("=" * 60)
    logger.info("Jobs scheduled:")
    for job in scheduler.get_jobs():
        logger.info(f"  {job.name}: {job.trigger}")

    scheduler.start()


def run_dry_run():
    """Execute all jobs once for testing (no scheduling)."""
    logger.info("=" * 60)
    logger.info("DRY RUN: Executing all jobs once")
    logger.info("=" * 60)

    for name, job_fn in [
        ("health_check", job_health_check),
        ("fetch_data", job_fetch_data),
        ("train_predict", job_train_and_predict),
        ("execute_trading", job_execute_trading),
        ("generate_report", job_generate_report),
    ]:
        logger.info(f"\n--- Running: {name} ---")
        try:
            job_fn()
            logger.info(f"  {name}: OK")
        except Exception as e:
            logger.error(f"  {name}: FAILED - {e}")

    logger.info("\nDry run complete. Pipeline state:")
    for k, v in _pipeline_state.items():
        logger.info(f"  {k}: {v}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="Energy Trading Pipeline Scheduler")
    parser.add_argument("--dry-run", action="store_true", help="Run all jobs once without scheduling")
    args = parser.parse_args()

    if args.dry_run:
        run_dry_run()
    else:
        run_scheduler()
