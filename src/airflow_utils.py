import os
import sys
from pathlib import Path


def get_project_root():
    """Dynamically calculates the project root path"""
    # Check if running in Airflow task
    if "AIRFLOW_CTX_DAG_RUN_ID" in os.environ:
        dag_file = os.environ["AIRFLOW_CTX_DAG_RUN_ID"]
        return Path(dag_file).resolve().parent  # .parent.parent

    # Fallback for direct execution
    return Path(__file__).resolve().parent


def add_project_root_to_path():
    """Adds project root to sys.path and returns it"""
    root = get_project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root
