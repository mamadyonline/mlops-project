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


def load_latest_model_and_test_data(project_root_path):
    """Load the trained model directly from S3 and test data"""
    try:
        import tempfile

        import joblib

        sys.path.append(project_root_path)
        from src.airflow_utils import add_project_root_to_path

        project_root = add_project_root_to_path()
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from src.s3_utils import download_file_from_s3, upload_file_to_s3

        print("Local imports successful")

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as model_tmp:
            model_path = model_tmp.name

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as data_tmp:
            data_path = data_tmp.name

        try:
            # Download model from S3
            download_file_from_s3(
                bucket="mlflow-project-artifacts-remote",
                key="pipeline_artifacts/trained_model.pkl",
                local_path=model_path,
            )

        except Exception as e:
            print(f"Could not load model from S3: {e}")
            raise Exception(
                "Trained model not found in S3. Check if your training pipeline saves the model."
            )

        # Download test data
        download_file_from_s3(
            bucket="mlflow-project-artifacts-remote",
            key="pipeline_artifacts/data.pkl",
            local_path=data_path,
        )

        # Load test data
        _, _, X_test, _, _, y_test = joblib.load(data_path)

        # Save model and test data for next task
        upload_file_to_s3(
            local_path=model_path,
            bucket="mlflow-project-artifacts-remote",
            key="deployment_artifacts/model.pkl",
        )

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as test_tmp:
            test_path = test_tmp.name
            joblib.dump((X_test, y_test), test_path)

            upload_file_to_s3(
                local_path=test_path,
                bucket="mlflow-project-artifacts-remote",
                key="deployment_artifacts/test_data.pkl",
            )

        # Clean up temporary files
        os.unlink(data_path)
        os.unlink(model_path)
        os.unlink(test_path)

        print("Model and test data loaded successfully")
        return "success"

    except Exception as e:
        print(f"Error in load_latest_model_and_test_data: {str(e)}")
        raise


def generate_predictions(project_root_path):
    """Generate predictions on test data"""
    try:
        import tempfile

        import joblib

        sys.path.append(project_root_path)
        from src.airflow_utils import add_project_root_to_path

        project_root = add_project_root_to_path()

        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        import pandas as pd

        from src.s3_utils import download_file_from_s3, upload_file_to_s3

        print("✓ Local imports successful")

        # Download model and test data
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as model_tmp:
            model_path = model_tmp.name

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as test_tmp:
            test_path = test_tmp.name

        download_file_from_s3(
            bucket="mlflow-project-artifacts-remote",
            key="deployment_artifacts/model.pkl",
            local_path=model_path,
        )

        download_file_from_s3(
            bucket="mlflow-project-artifacts-remote",
            key="deployment_artifacts/test_data.pkl",
            local_path=test_path,
        )

        # Load model and test data
        model = joblib.load(model_path)
        X_test, y_test = joblib.load(test_path)

        # Generate predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[
            :, 1
        ]  # Get probability for positive class

        # Create predictions dataframe
        predictions_df = pd.DataFrame(
            {
                "true_label": y_test,
                "predicted_label": y_pred,
                "predicted_probability": y_pred_proba,
            }
        )

        # Save predictions
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as pred_tmp:
            pred_path = pred_tmp.name
            predictions_df.to_csv(pred_path, index=False)

            # Upload predictions to S3
            upload_file_to_s3(
                local_path=pred_path,
                bucket="mlflow-project-artifacts-remote",
                key="deployment_artifacts/predictions.csv",
            )

        # Also save as pickle for metrics calculation
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as pred_pkl_tmp:
            pred_pkl_path = pred_pkl_tmp.name
            joblib.dump((y_test, y_pred, y_pred_proba), pred_pkl_path)

            upload_file_to_s3(
                local_path=pred_pkl_path,
                bucket="mlflow-project-artifacts-remote",
                key="deployment_artifacts/predictions.pkl",
            )

        # Clean up temporary files
        os.unlink(model_path)
        os.unlink(test_path)
        os.unlink(pred_path)
        os.unlink(pred_pkl_path)

        print(f"Predictions generated for {len(y_test)} samples")
        return "success"

    except Exception as e:
        print(f"Error in generate_predictions: {str(e)}")
        raise


def calculate_performance_metrics(project_root_path):
    """Calculate and report model performance metrics"""
    try:
        import json
        import tempfile
        from datetime import datetime

        import joblib

        sys.path.append(project_root_path)
        from src.airflow_utils import add_project_root_to_path

        project_root = add_project_root_to_path()

        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        from sklearn.metrics import (
            accuracy_score,
            classification_report,
            confusion_matrix,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )

        from src.s3_utils import download_file_from_s3, upload_file_to_s3

        print("Local imports successful")

        # Download predictions
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as pred_tmp:
            pred_path = pred_tmp.name

        download_file_from_s3(
            bucket="mlflow-project-artifacts-remote",
            key="deployment_artifacts/predictions.pkl",
            local_path=pred_path,
        )

        # Load predictions
        y_test, y_pred, y_pred_proba = joblib.load(pred_path)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        try:
            auc_score = roc_auc_score(y_test, y_pred_proba)
        except Exception:
            auc_score = None

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)

        # Create metrics report
        metrics_report = {
            "timestamp": datetime.now().isoformat(),
            "test_samples": len(y_test),
            "metrics": {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1_score": float(f1),
                "auc_score": float(auc_score) if auc_score is not None else None,
            },
            "confusion_matrix": cm.tolist(),
            "classification_report": class_report,
        }

        # Print metrics summary
        print("\n" + "=" * 50)
        print("MODEL PERFORMANCE METRICS")
        print("=" * 50)
        print(f"Test Samples: {len(y_test)}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        if auc_score is not None:
            print(f"AUC Score: {auc_score:.4f}")
        print("\nConfusion Matrix:")
        print(cm)
        print("=" * 50)

        # Save metrics report
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as metrics_tmp:
            metrics_path = metrics_tmp.name
            json.dump(metrics_report, metrics_tmp, indent=2)

            # Upload metrics to S3
            upload_file_to_s3(
                local_path=metrics_path,
                bucket="mlflow-project-artifacts-remote",
                key=f"deployment_artifacts/metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            )

        # Also save latest metrics
        upload_file_to_s3(
            local_path=metrics_path,
            bucket="mlflow-project-artifacts-remote",
            key="deployment_artifacts/latest_metrics.json",
        )

        # Clean up temporary files
        os.unlink(pred_path)
        os.unlink(metrics_path)

        print("Performance metrics calculated and saved")
        return "success"

    except Exception as e:
        print(f"Error in calculate_performance_metrics: {str(e)}")
        raise


def monitor_model_task(project_root_path):
    """Monitor model with Evidently"""
    try:
        import tempfile
        from datetime import datetime

        import joblib
        import pandas as pd

        sys.path.append(project_root_path)
        from src.airflow_utils import add_project_root_to_path

        project_root = add_project_root_to_path()
        # Add to Python path
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))

        import mlflow
        from evidently import Report
        from evidently.metrics import (
            DatasetMissingValueCount,
            DriftedColumnsCount,
            DuplicatedRowCount,
        )

        from src.s3_utils import download_file_from_s3, upload_file_to_s3

        print("✓ Local imports successful")

        # Set up MLflow
        aws_profile = os.getenv("MY_AWS_PROFILE")
        if aws_profile:
            os.environ["AWS_PROFILE"] = aws_profile
        tracking_uri = os.getenv("MY_AWS_EC2_SERVER")
        if tracking_uri:
            mlflow.set_tracking_uri(f"http://{tracking_uri}:5000")
        mlflow.set_experiment("heart-disease-experiment")

        try:
            # Download data from S3
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as data_tmp:
                data_path = data_tmp.name
            download_file_from_s3(
                bucket="mlflow-project-artifacts-remote",
                key="pipeline_artifacts/data.pkl",
                local_path=data_path,
            )
            X_train, X_val, X_test, y_train, y_val, y_test = joblib.load(data_path)

            # Get latest model deployed
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as model_tmp:
                model_path = model_tmp.name
            download_file_from_s3(
                bucket="mlflow-project-artifacts-remote",
                key="deployment_artifacts/model.pkl",
                local_path=model_path,
            )

            train_df = pd.concat([X_train, y_train], axis=1)
            test_df = pd.concat([X_test, y_test], axis=1)

            report = Report(
                metrics=[
                    DuplicatedRowCount(),
                    DriftedColumnsCount(),
                    DatasetMissingValueCount(),
                ]
            )

            # Run the report
            report_snapshot = report.run(reference_data=train_df, current_data=test_df)

            # Save report to temporary file
            report_dict = report_snapshot.dict()
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".pkl", delete=False
            ) as report_tmp:
                report_path = report_tmp.name
            joblib.dump(report_dict, report_path)

            # Upload report to S3
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            s3_key = f"monitoring_reports/evidently_report_{timestamp}.pkl"
            upload_file_to_s3(
                local_path=report_path,
                bucket="mlflow-project-artifacts-remote",
                key=s3_key,
            )

            # Log basic metrics
            print("=== MONITORING REPORT ===")
            print(f"Report saved to S3: {s3_key}")

            # Extract some key metrics
            drift_metrics = report_dict.get("metrics", [])
            for metric in drift_metrics:
                if metric.get("metric") == "DriftedColumnsCount":
                    drifted_count = metric.get("result", {}).get(
                        "number_of_drifted_columns", 0
                    )
                    total_count = metric.get("result", {}).get("number_of_columns", 0)
                    print(f"Drifted columns: {drifted_count}/{total_count}")
                elif metric.get("metric") == "DatasetMissingValueCount":
                    missing_info = metric.get("result", {})
                    print(
                        f"Missing values - Reference: {missing_info.get('reference', {})}"
                    )
                    print(
                        f"Missing values - Current: {missing_info.get('current', {})}"
                    )

            print("Model monitoring completed successfully")
            return "success"

        finally:
            # Clean up temporary files
            if os.path.exists(data_path):
                os.unlink(data_path)
            if "report_path" in locals() and os.path.exists(report_path):
                os.unlink(report_path)

    except Exception as e:
        print(f"Error in monitor_model_task: {str(e)}")
        raise


# Define DAG
default_args = {
    "owner": "mamady",
    "start_date": datetime(2025, 6, 1),
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    "heart_disease_batch_deployment",
    default_args=default_args,
    schedule="@daily",  # Run manually or set schedule (e.g., "@daily")
    catchup=False,
    tags=["machine-learning", "deployment", "batch", "mlops"],
    description="Batch deployment pipeline for heart disease risk prediction model",
) as dag:
    load_model_data = ExternalPythonOperator(
        task_id="load_model_and_test_data",
        python_callable=load_latest_model_and_test_data,
        op_args=[str(PROJECT_ROOT)],
        python=sys.executable,
    )

    predict = ExternalPythonOperator(
        task_id="generate_predictions",
        python_callable=generate_predictions,
        op_args=[str(PROJECT_ROOT)],
        python=sys.executable,
    )

    metrics = ExternalPythonOperator(
        task_id="calculate_metrics",
        python_callable=calculate_performance_metrics,
        op_args=[str(PROJECT_ROOT)],
        python=sys.executable,
    )

    monitor = ExternalPythonOperator(
        task_id="monitor_model",
        python_callable=monitor_model_task,
        op_args=[str(PROJECT_ROOT)],
        python=sys.executable,
    )

    # Define task dependencies
    load_model_data >> predict >> metrics >> monitor
