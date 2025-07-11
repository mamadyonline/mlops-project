# airflow/dags/ml_pipeline_dag.py

from datetime import datetime

from airflow.providers.standard.operators.python import PythonOperator

from airflow import DAG


def dummy_task():
    print("Hello from Airflow")


with DAG(
    dag_id="test_pipeline_dag",
    start_date=datetime(2025, 7, 10),
    schedule=None,
    catchup=False,
    tags=["mlops"],
) as dag:
    task = PythonOperator(
        task_id="run_dummy",
        python_callable=dummy_task,
    )

    task
