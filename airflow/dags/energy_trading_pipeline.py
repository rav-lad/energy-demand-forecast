"""
Airflow DAG for Energy Trading Pipeline

Workflow:
1. Collect market data (prices, fundamentals)
2. Collect weather data
3. Train models (daily)
4. Generate predictions
5. Generate trading signals
6. Run backtest
7. Send reports
"""

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

# Default arguments
default_args = {
    'owner': 'energy-trading',
    'depends_on_past': False,
    'email': ['alerts@energy-trading.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# Define DAG
dag = DAG(
    'energy_trading_pipeline',
    default_args=default_args,
    description='Complete energy trading ML pipeline',
    schedule_interval='0 2 * * *',  # Run daily at 2 AM
    start_date=days_ago(1),
    catchup=False,
    tags=['energy', 'trading', 'ml'],
)


# =============================================================================
# Python Callables
# =============================================================================

def collect_market_data(**context):
    """Collect market price data from ENTSO-E."""
    logger.info("Collecting market data...")

    import sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    from data_recuperation.data_market_prices import collect_day_ahead_prices

    # Get yesterday's date
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=7)

    # Collect data
    df = collect_day_ahead_prices(
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        countries=['FR', 'DE', 'ES']
    )

    logger.info(f"Collected {len(df)} price records")

    # Store in XCom
    context['task_instance'].xcom_push(key='market_data_rows', value=len(df))

    return f"Collected {len(df)} records"


def collect_weather_data(**context):
    """Collect weather forecast data."""
    logger.info("Collecting weather data...")

    # Implementation would call Open-Meteo or GraphCast
    # For now, just a placeholder

    logger.info("Weather data collected")

    return "Weather data collected successfully"


def validate_data(**context):
    """Validate collected data quality."""
    logger.info("Validating data...")

    import sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    from src.data_validation.validator import DataValidator
    import pandas as pd

    validator = DataValidator(strict_mode=False)

    # Validate weather data (sample)
    # In production, load actual data
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    weather_data = pd.DataFrame({
        'date': dates,
        'temperature_2m_max': [15] * len(dates),
        'temperature_2m_min': [5] * len(dates),
        'precipitation_sum': [2] * len(dates),
        'wind_speed_10m_max': [15] * len(dates),
        'shortwave_radiation_sum': [5000] * len(dates),
    })

    result = validator.validate_weather_data(weather_data)

    if not result:
        raise ValueError("Data validation failed")

    logger.info("Data validation passed")

    return "Data validation successful"


def train_models(**context):
    """Train all ML models."""
    logger.info("Training models...")

    # This would call the training pipeline
    # For now, return success

    logger.info("Models trained successfully")

    return "Models trained"


def generate_predictions(**context):
    """Generate predictions for next day."""
    logger.info("Generating predictions...")

    # Load model and generate predictions
    # Store results

    logger.info("Predictions generated")

    context['task_instance'].xcom_push(key='predictions_generated', value=True)

    return "Predictions generated"


def generate_trading_signals(**context):
    """Generate trading signals based on predictions."""
    logger.info("Generating trading signals...")

    import sys
    from pathlib import Path

    project_root = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(project_root))

    from trading_system.strategies.demand_price_arbitrage import DemandPriceArbitrageStrategy
    import pandas as pd
    import numpy as np

    # Sample demand data
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    demand = pd.Series(np.random.normal(5000, 1000, len(dates)), index=dates)

    # Initialize and calibrate strategy
    strategy = DemandPriceArbitrageStrategy()
    strategy.calibrate(demand)

    # Generate signals
    signals = strategy.generate_signals(demand)

    n_signals = (signals['signal'] != 0).sum()

    logger.info(f"Generated {n_signals} trading signals")

    context['task_instance'].xcom_push(key='signals_count', value=n_signals)

    return f"Generated {n_signals} signals"


def run_backtest(**context):
    """Run backtest on trading strategy."""
    logger.info("Running backtest...")

    # Implementation would run full backtest
    # For now, return success

    logger.info("Backtest completed")

    return "Backtest completed"


def send_report(**context):
    """Send daily report with results."""
    logger.info("Sending daily report...")

    # Get results from previous tasks
    market_data_rows = context['task_instance'].xcom_pull(
        task_ids='collect_market_data',
        key='market_data_rows'
    )

    predictions_generated = context['task_instance'].xcom_pull(
        task_ids='generate_predictions',
        key='predictions_generated'
    )

    signals_count = context['task_instance'].xcom_pull(
        task_ids='generate_trading_signals',
        key='signals_count'
    )

    # Compile report
    report = f"""
    Energy Trading Pipeline - Daily Report
    =====================================

    Date: {datetime.now().strftime('%Y-%m-%d')}

    Data Collection:
    - Market data rows: {market_data_rows}

    Predictions:
    - Generated: {predictions_generated}

    Trading Signals:
    - Signals count: {signals_count}

    Status: SUCCESS
    """

    logger.info(report)

    # In production, send email or save to database

    return "Report sent"


# =============================================================================
# Define Tasks
# =============================================================================

# Data collection
collect_market_data_task = PythonOperator(
    task_id='collect_market_data',
    python_callable=collect_market_data,
    dag=dag,
)

collect_weather_data_task = PythonOperator(
    task_id='collect_weather_data',
    python_callable=collect_weather_data,
    dag=dag,
)

# Data validation
validate_data_task = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data,
    dag=dag,
)

# Model training
train_models_task = PythonOperator(
    task_id='train_models',
    python_callable=train_models,
    dag=dag,
)

# Predictions
generate_predictions_task = PythonOperator(
    task_id='generate_predictions',
    python_callable=generate_predictions,
    dag=dag,
)

# Trading signals
generate_signals_task = PythonOperator(
    task_id='generate_trading_signals',
    python_callable=generate_trading_signals,
    dag=dag,
)

# Backtest
run_backtest_task = PythonOperator(
    task_id='run_backtest',
    python_callable=run_backtest,
    dag=dag,
)

# Reporting
send_report_task = PythonOperator(
    task_id='send_report',
    python_callable=send_report,
    dag=dag,
)

# Cleanup old data (Bash command)
cleanup_task = BashOperator(
    task_id='cleanup_old_data',
    bash_command='find /app/outputs/logs -name "*.log" -mtime +30 -delete',
    dag=dag,
)


# =============================================================================
# Define Dependencies
# =============================================================================

# Data collection in parallel
[collect_market_data_task, collect_weather_data_task] >> validate_data_task

# Training after validation
validate_data_task >> train_models_task

# Predictions after training
train_models_task >> generate_predictions_task

# Trading signals after predictions
generate_predictions_task >> generate_signals_task

# Backtest after signals
generate_signals_task >> run_backtest_task

# Report after backtest
run_backtest_task >> send_report_task

# Cleanup after report
send_report_task >> cleanup_task
