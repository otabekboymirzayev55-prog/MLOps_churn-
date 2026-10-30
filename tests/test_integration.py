from unittest.mock import Mock
from pathlib import Path

import joblib
import numpy as np
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from api.main import create_app
from api.schema import Customer
from dashboard_client import call_api, predict_url


class WorkingModel:
    def predict_proba(self, row):
        assert set(row.columns) == set(Customer.model_fields)
        return np.array([[0.3, 0.7]])


@pytest.fixture
def client(tmp_path):
    path = tmp_path / "model.pkl"
    joblib.dump(WorkingModel(), path)
    return TestClient(create_app(path))


def test_dashboard_request_matches_api(client):
    result = call_api(Customer.model_json_schema()["example"], "http://testserver/", client)
    assert result == {"churn_probability": 0.7, "will_churn": True, "risk": "high", "threshold": 0.5}
    assert client.get("/health").status_code == 200


@pytest.mark.parametrize("content", [None, b"corrupt model"])
def test_unavailable_model_is_not_ready(tmp_path, content):
    path = tmp_path / "model.pkl"
    if content is not None:
        path.write_bytes(content)
    client = TestClient(create_app(path))
    assert client.get("/health").status_code == 503
    assert client.post("/predict", json=Customer.model_json_schema()["example"]).status_code == 503


@pytest.mark.parametrize("field,value", [("gender", "unknown"), ("SeniorCitizen", 2), ("tenure", -1), ("MonthlyCharges", -1), ("PhoneService", "No"), ("InternetService", "No")])
def test_invalid_customer_rejected(client, field, value):
    payload = dict(Customer.model_json_schema()["example"])
    payload[field] = value
    assert client.post("/predict", json=payload).status_code == 422


@pytest.mark.parametrize("url", ["localhost:8000", "https://api/predict", "https://api?key=x", ""])
def test_invalid_api_url(url):
    with pytest.raises(ValueError):
        predict_url(url)


def test_invalid_response():
    response = Mock()
    response.json.return_value = {"churn_probability": "bad"}
    with pytest.raises(ValueError):
        call_api({}, "http://api:8000", Mock(post=Mock(return_value=response)))


def test_dashboard_form_renders():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "dashboard.py").run()
    assert not app.exception
    assert len(app.selectbox) == 16
    assert len(app.number_input) == 3
