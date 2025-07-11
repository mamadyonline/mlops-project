from unittest.mock import patch

import pytest
import xgboost as xgb
from optuna.trial import FixedTrial
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.model import optimize_model  # adjust import if needed
from src.model import objective, train_model


@pytest.fixture
def sample_data():
    X, y = make_classification(
        n_samples=100, n_features=5, n_classes=2, random_state=42
    )
    return train_test_split(X, y, test_size=0.2, random_state=42)


def test_objective_returns_loss(sample_data):
    X_train, X_val, y_train, y_val = sample_data
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    fixed_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        # 'num_class': 2,
        "max_depth": 4,
        "learning_rate": 0.1,
        "min_child_weight": 1,
        "reg_alpha": 0.01,
        "reg_lambda": 1,
    }
    trial = FixedTrial(fixed_params)

    result = objective(trial, dtrain, dval, y_val)
    assert isinstance(result, float)
    assert result <= 0


def test_optimize_model_returns_params(sample_data):
    X_train, X_val, y_train, y_val = sample_data

    best_params = optimize_model(X_train, X_val, y_train, y_val)
    assert isinstance(best_params, dict)
    assert "max_depth" in best_params
    assert int(best_params["max_depth"]) >= 4


@patch("mlflow.start_run")
@patch("mlflow.log_params")
@patch("mlflow.log_metric")
@patch("mlflow.xgboost.log_model")
def test_train_model_logs_and_returns_model(
    mock_log_model, mock_log_metric, mock_log_params, mock_start_run, sample_data
):
    X_train, X_test, y_train, y_test = sample_data

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 4,
        "learning_rate": 0.1,
        "min_child_weight": 1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
    }

    model = train_model(X_train, X_test, y_train, y_test, params)
    assert hasattr(model, "predict")
    assert callable(model.predict)
    mock_log_params.assert_called_once()
    mock_log_metric.assert_called_once()
    # mock_log_model.assert_called_once()
