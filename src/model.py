from typing import Any, Dict, List, Union

import mlflow

numeric = Union[int, float]


def objective(
    trial: Any, dtrain: Any, dval: Any, y_val: List[int]
) -> Dict[str, Union[float, int]]:
    """Objective function trying to minimize area under ROC

    Args:
        trial (optuna.Trial): Optuna trial for fine-tuning.
        dtrain (xgb.DMatrix): Input for training in xgboost matrix format.
        dval (xgb.DMatrix): Input for evaluation in xgboost matrix format.
        y_val (List[int]): Output for training.

    Returns:
        float: Model training loss.
    """
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        # 'num_class': 2,
        "max_depth": trial.suggest_int("max_depth", 4, 15),
        "learning_rate": trial.suggest_float("learning_rate", 1e-7, 1.0, log=True),
        "min_child_weight": trial.suggest_float(
            "min_child_weight", 0.1, 100.0, log=True
        ),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-10, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-10, 10.0, log=True),
    }
    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=100,
        evals=[(dval, "validation")],
        early_stopping_rounds=20,
    )
    y_pred = booster.predict(dval)
    # Note: roc_auc_score() expects a 1D array of probabilities
    # for the positive class only when y_val is binary
    auc = roc_auc_score(y_val, y_pred)

    return -auc


def optimize_model(
    X_train: List[numeric],
    X_val: List[numeric],
    y_train: List[int],
    y_val: List[int],
    n_trials: int = 20,
):
    """Train an XGBoost model.

    Args:
        X_train (List[numeric]): Input for training the model.
        X_val (List[numeric]): Input for validation.
        y_train (List[int]): Output for training the model.
        y_val (List[int]): Output for validation.
        n_trials (int): The number of trials in optuna. Default to 20.

    Returns:
        Dict[str, Union[float, int]]: The best parameters after hyperparameter optimization.
    """
    import optuna
    import xgboost as xgb

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, dtrain, dval, y_val), n_trials=n_trials
    )
    best_params = study.best_params

    return best_params


def train_model(
    X_train: List[numeric],
    X_test: List[numeric],
    y_train: List[int],
    y_test: List[int],
    params: Dict[str, Union[float, int]],
):
    """Train an XGBoost model.

    Args:
        X_train (List[numeric]): Input for training the model.
        X_test (List[numeric]): Input for testing the model.
        y_train (List[int]): Output for training the model.
        y_test (List[int]): Output for testing the model.
        params (Dict[str, Union[float, int]]): The model hyperparameters.
        tracking_uri (str): Model tracking uri with MLFlow.
        experiment_name (str): Model tracking experiment name.
    """
    import xgboost as xgb
    from sklearn.metrics import accuracy_score

    with mlflow.start_run():
        # dtrain = xgb.DMatrix(X_train, label=y_train)
        # dtest = xgb.DMatrix(X_test, label=y_test)

        params["max_depth"] = int(params["max_depth"])
        mlflow.log_params(params)
        # train model
        xgb_model = xgb.XGBClassifier(**params)
        xgb_model.fit(X_train, y_train)
        # evaluate model
        y_pred = xgb_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.xgboost.log_model(xgb_model, artifact_path="models_mlflow")

        return xgb_model
