import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow.providers.standard.operators.python import ExternalPythonOperator

from airflow import DAG

# Calculate project root dynamically
DAG_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = DAG_DIR.parent.parent  # Adjust based on your structure
print(f"PROJECT_ROOT: {PROJECT_ROOT}")


def load_and_split_data(project_root_path):
    """Load data and split it into train/val/test sets"""
    try:
        # import sys
        import tempfile

        import joblib

        sys.path.append(project_root_path)
        from src.airflow_utils import add_project_root_to_path

        project_root = add_project_root_to_path()
        # Add to Python path
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        # local imports
        from src.preprocessing import get_splits, read_dataframe
        from src.s3_utils import upload_file_to_s3

        print("Local imports successful")

        filename = "data/heart.csv"
        df = read_dataframe(filename)
        X_train, X_val, X_test, y_train, y_val, y_test = get_splits(df)

        # Use a temporary file for better handling
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            data_path = tmp_file.name
            joblib.dump((X_train, X_val, X_test, y_train, y_val, y_test), data_path)

            # Upload to S3
            upload_file_to_s3(
                local_path=data_path,
                bucket="mlflow-project-artifacts-remote",
                key="pipeline_artifacts/data.pkl",
            )

        # Clean up temporary file
        os.unlink(data_path)

        print("Data loaded and uploaded successfully")
        return "success"

    except Exception as e:
        print(f"Error in load_and_split_data: {str(e)}")
        raise


def optimize_model_task(project_root_path):
    """Optimize model hyperparameters"""
    try:
        import tempfile

        import joblib

        sys.path.append(project_root_path)
        from src.airflow_utils import add_project_root_to_path

        project_root = add_project_root_to_path()
        # Add to Python path
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        from src.model import optimize_model
        from src.s3_utils import download_file_from_s3, upload_file_to_s3

        print("Local imports successful")
        # Use temporary files
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as data_tmp:
            data_path = data_tmp.name

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as params_tmp:
            params_path = params_tmp.name

        try:
            # Download data from S3
            download_file_from_s3(
                bucket="mlflow-project-artifacts-remote",
                key="pipeline_artifacts/data.pkl",
                local_path=data_path,
            )

            # Load data - Fix: correct unpacking
            X_train, X_val, X_test, y_train, y_val, y_test = joblib.load(data_path)

            # Optimize model
            best_params = optimize_model(X_train, X_val, y_train, y_val)

            # Save parameters
            joblib.dump(best_params, params_path)

            # Upload parameters to S3
            upload_file_to_s3(
                local_path=params_path,
                bucket="mlflow-project-artifacts-remote",
                key="pipeline_artifacts/best_params.pkl",
            )

            print("Model optimization completed successfully")
            return "success"

        finally:
            # Clean up temporary files
            if os.path.exists(data_path):
                os.unlink(data_path)
            if os.path.exists(params_path):
                os.unlink(params_path)

    except Exception as e:
        print(f"Error in optimize_model_task: {str(e)}")
        raise


def train_and_log_model(project_root_path):
    """Train final model and log to MLflow"""
    try:
        import tempfile

        import joblib

        sys.path.append(project_root_path)
        from src.airflow_utils import add_project_root_to_path

        project_root = add_project_root_to_path()
        # Add to Python path
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        import mlflow

        from src.model import train_model
        from src.s3_utils import download_file_from_s3, upload_file_to_s3

        print("Local imports successful")
        # Use temporary files
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as data_tmp:
            data_path = data_tmp.name

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as params_tmp:
            params_path = params_tmp.name

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as model_tmp:
            model_path = model_tmp.name

        try:
            # Download artifacts from S3
            download_file_from_s3(
                bucket="mlflow-project-artifacts-remote",
                key="pipeline_artifacts/data.pkl",
                local_path=data_path,
            )

            download_file_from_s3(
                bucket="mlflow-project-artifacts-remote",
                key="pipeline_artifacts/best_params.pkl",
                local_path=params_path,
            )

            # Load data and parameters - Fix: correct unpacking
            X_train, X_val, X_test, y_train, y_val, y_test = joblib.load(data_path)
            best_params = joblib.load(params_path)

            # Set up MLflow
            aws_profile = os.getenv("MY_AWS_PROFILE")
            if aws_profile:
                os.environ["AWS_PROFILE"] = aws_profile

            tracking_uri = os.getenv("MY_AWS_EC2_SERVER")
            if tracking_uri:
                mlflow.set_tracking_uri(f"http://{tracking_uri}:5000")
            else:
                print(
                    "Warning: MY_AWS_EC2_SERVER not set, using default MLflow tracking"
                )

            mlflow.set_experiment("heart-disease-experiment")

            # Train model
            trained_model = train_model(X_train, X_test, y_train, y_test, best_params)

            # Save model
            joblib.dump(trained_model, model_path)
            # Upload model to S3
            upload_file_to_s3(
                local_path=model_path,
                bucket="mlflow-project-artifacts-remote",
                key="pipeline_artifacts/trained_model.pkl",
            )

            print("Model training and logging completed successfully")
            return "success"

        finally:
            # Clean up temporary files
            if os.path.exists(data_path):
                os.unlink(data_path)
            if os.path.exists(params_path):
                os.unlink(params_path)

    except Exception as e:
        print(f"Error in train_and_log_model: {str(e)}")
        raise


# Define DAG
default_args = {
    "owner": "mamady",
    "start_date": datetime(2025, 6, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}


with DAG(
    "heart_disease_ml_training_pipeline",
    default_args=default_args,
    schedule=None,  # "@daily",  # run manually or set a schedule
    catchup=False,
    tags=["machine-learning", "s3", "mlops"],
    description="ML training pipeline for heart disease risk prediction",
) as dag:
    load_split = ExternalPythonOperator(
        task_id="load_and_split_data",
        python_callable=load_and_split_data,
        op_args=[str(PROJECT_ROOT)],
        python=sys.executable,
    )

    optimize = ExternalPythonOperator(
        task_id="optimize_model",
        python_callable=optimize_model_task,
        op_args=[str(PROJECT_ROOT)],
        python=sys.executable,
    )

    train = ExternalPythonOperator(
        task_id="train_model",
        python_callable=train_and_log_model,
        op_args=[str(PROJECT_ROOT)],
        python=sys.executable,
    )

    # # Define task dependencies
    load_split >> optimize >> train
