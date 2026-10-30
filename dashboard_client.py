"""HTTP contract shared by the dashboard and integration tests."""
from urllib.parse import urlsplit

import requests


def predict_url(base_url: str) -> str:
    base_url = base_url.strip().rstrip("/")
    parts = urlsplit(base_url)
    if parts.scheme not in {"http", "https"} or not parts.netloc or parts.query or parts.fragment:
        raise ValueError("API_URL must be an HTTP(S) base URL without query or fragment")
    if parts.path.endswith("/predict"):
        raise ValueError("API_URL must be the base URL, without /predict")
    return base_url + "/predict"


def call_api(payload, base_url, client=requests):
    response = client.post(predict_url(base_url), json=payload, timeout=30)
    response.raise_for_status()
    result = response.json()
    probability = result.get("churn_probability")
    if (isinstance(probability, bool) or not isinstance(probability, (int, float))
            or not 0 <= probability <= 1 or result.get("risk") not in {"high", "low"}
            or not isinstance(result.get("will_churn"), bool)):
        raise ValueError("API returned an invalid prediction")
    return result
