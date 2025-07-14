from pathlib import Path

import joblib
import mlflow

import src.model as model
import src.preprocessing as pp

if __name__ == "__main__":
    # MLFlow Settings
    TRACKING_URI = "http://localhost:6000"
    mlflow.set_tracking_uri(f"{TRACKING_URI}")
    mlflow.set_experiment("heart-disease-experiment")

    # Load
    filename = "data/heart.csv"
    df = pp.read_dataframe(filename)
    # Transform
    X_train, X_val, X_test, y_train, y_val, y_test = pp.get_splits(
        df, train_size=0.8, test_size=0.2, random_state=123
    )

    # Fine-tune
    best_params = model.optimize_model(X_train, X_val, y_train, y_val)
    # Train
    clf = model.train_model(X_train, X_test, y_train, y_test, best_params)

    SAVE = False
    if SAVE:
        models_dir = Path("models").mkdir(parents=True, exist_ok=True)
        model_path = models_dir / "model.pkl"
        with open(model_path, "wb") as fw:
            joblib.dump(clf, fw)
