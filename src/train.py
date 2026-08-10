"""Train churn models, track experiments in MLflow, and save the best pipeline."""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

try:  # pragma: no cover - import flexibility for direct script execution
    from src.preprocess import build_preprocessor, load_data
except ImportError:  # pragma: no cover
    from preprocess import build_preprocessor, load_data


def get_models() -> dict[str, object]:
    """Return the candidate models we compare in MLflow."""

    return {
        "logreg": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }


def _resolve_path(value: str | None, default: str) -> Path:
    return Path(value or default)


def train_models(
    data_path: str | Path | None = None,
    model_path: str | Path | None = None,
    experiment_name: str = "churn",
    tracking_uri: str | None = None,
) -> dict[str, object]:
    """Train all candidate models and return a summary for tests or CLI usage."""

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

    data_path = _resolve_path(str(data_path) if data_path is not None else os.getenv("DATA_PATH"), "data/telco_churn.csv")
    model_path = _resolve_path(str(model_path) if model_path is not None else os.getenv("MODEL_PATH"), "models/model.pkl")
    tracking_uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    X, y = load_data(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    preprocessor = build_preprocessor(X_train)
    mlflow.set_experiment(experiment_name)

    best_pipeline = None
    best_auc = -1.0
    best_name = None
    runs: list[dict[str, float | str]] = []

    for name, classifier in get_models().items():
        with mlflow.start_run(run_name=name):
            pipeline = Pipeline(
                steps=[
                    ("preprocess", preprocessor),
                    ("classifier", classifier),
                ]
            )
            pipeline.fit(X_train, y_train)

            probabilities = pipeline.predict_proba(X_test)[:, 1]
            predictions = pipeline.predict(X_test)

            metrics = {
                "roc_auc": roc_auc_score(y_test, probabilities),
                "pr_auc": average_precision_score(y_test, probabilities),
                "f1": f1_score(y_test, predictions),
            }

            mlflow.log_param("model", name)
            mlflow.log_param("train_rows", len(X_train))
            mlflow.log_param("test_rows", len(X_test))
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipeline, "model")

            runs.append({"name": name, **metrics})

            if metrics["roc_auc"] > best_auc:
                best_pipeline = pipeline
                best_auc = metrics["roc_auc"]
                best_name = name

    if best_pipeline is None or best_name is None:
        raise RuntimeError("No model was trained.")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, model_path)

    return {
        "best_name": best_name,
        "best_auc": best_auc,
        "model_path": str(model_path),
        "runs": runs,
    }


def main() -> dict[str, object]:
    return train_models()


if __name__ == "__main__":
    summary = main()
    print(
        f"BEST: {summary['best_name']} "
        f"(ROC-AUC={summary['best_auc']:.4f}) -> saved {summary['model_path']}"
    )
