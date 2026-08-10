from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from fastapi.testclient import TestClient

from api.main import create_app
from src.preprocess import build_preprocessor, load_data
from src.train import train_models


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "telco_churn.csv"


def test_load_data_handles_blank_total_charges(tmp_path):
    csv_path = tmp_path / "mini.csv"
    pd.DataFrame(
        [
            {
                "customerID": "0001",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 29.9,
                "TotalCharges": "",
                "Churn": "Yes",
            }
        ]
    ).to_csv(csv_path, index=False)

    X, y = load_data(csv_path)

    assert X.loc[0, "TotalCharges"] == 0.0
    assert y.iloc[0] == 1


def test_training_pipeline_and_model_quality(tmp_path):
    model_path = tmp_path / "model.pkl"
    tracking_uri = (tmp_path / "mlruns").as_uri()

    summary = train_models(
        data_path=DATA_PATH,
        model_path=model_path,
        experiment_name="pytest-churn",
        tracking_uri=tracking_uri,
    )

    assert model_path.exists()
    assert summary["best_auc"] >= 0.83

    model = joblib.load(model_path)
    X, _ = load_data(DATA_PATH)
    transformed = build_preprocessor(X).fit_transform(X.head(5))
    assert transformed.shape[0] == 5
    assert hasattr(model, "predict_proba")


def test_api_predicts_high_and_low_risk(tmp_path):
    model_path = tmp_path / "model.pkl"
    tracking_uri = (tmp_path / "mlruns").as_uri()
    train_models(
        data_path=DATA_PATH,
        model_path=model_path,
        experiment_name="pytest-api",
        tracking_uri=tracking_uri,
    )

    X, _ = load_data(DATA_PATH)
    model = joblib.load(model_path)
    probabilities = model.predict_proba(X)[:, 1]

    high_index = int(probabilities.argmax())
    low_index = int(probabilities.argmin())

    app = create_app(model_path=model_path)
    client = TestClient(app)

    high_payload = X.iloc[high_index].to_dict()
    low_payload = X.iloc[low_index].to_dict()

    high_response = client.post("/predict", json=high_payload)
    low_response = client.post("/predict", json=low_payload)

    assert high_response.status_code == 200
    assert low_response.status_code == 200
    assert high_response.json()["risk"] == "high"
    assert low_response.json()["risk"] == "low"
    assert 0.0 <= high_response.json()["churn_probability"] <= 1.0
    assert 0.0 <= low_response.json()["churn_probability"] <= 1.0


def test_health_endpoint_reports_ready_when_model_exists(tmp_path, monkeypatch):
    model_path = tmp_path / "model.pkl"
    tracking_uri = (tmp_path / "mlruns").as_uri()
    train_models(
        data_path=DATA_PATH,
        model_path=model_path,
        experiment_name="pytest-health",
        tracking_uri=tracking_uri,
    )
    app = create_app(model_path=model_path)
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["ready"] is True
