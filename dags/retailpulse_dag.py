from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'zidio-retailpulse',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

with DAG(
    'retailpulse_ml_pipeline',
    default_args=default_args,
    description='A daily batch job to retrain models and generate forecasts',
    schedule_interval=timedelta(days=1),
    start_date=datetime(2026, 1, 1),
    catchup=False,
) as dag:

    # Task 1: Data Extraction & Validation (Great Expectations)
    run_etl = BashOperator(
        task_id='run_etl_validation',
        bash_command='python /app/src/data/make_dataset.py && python /app/src/data/validation.py',
    )

    # Task 2: Feature Engineering
    run_features = BashOperator(
        task_id='run_features',
        bash_command='python /app/src/features/build_features.py',
    )

    # Task 3: Retrain Segmentation (K-Means)
    train_segmentation = BashOperator(
        task_id='train_segmentation',
        bash_command='python /app/src/segmentation/kmeans_model.py',
    )

    # Task 4: Retrain Hybrid Forecasting (Prophet + LSTM)
    train_forecaster = BashOperator(
        task_id='train_forecaster',
        bash_command='python /app/src/forecasting/train_forecaster.py',
    )

    # Task 5: Retrain Churn (XGBoost + Optuna)
    train_churn = BashOperator(
        task_id='train_churn',
        bash_command='python /app/src/churn/churn_model.py',
    )

    # Task 6: Inventory Optimization
    run_inventory = BashOperator(
        task_id='run_inventory',
        bash_command='python /app/src/inventory/inventory_engine.py',
    )

    # Task 7: Data Drift Detection (Evidently AI)
    run_drift_detection = BashOperator(
        task_id='run_drift_detection',
        bash_command='python /app/src/mlops/data_drift.py',
    )

    run_etl >> run_features >> [train_segmentation, train_forecaster, train_churn] >> run_inventory >> run_drift_detection
