"""FastAPI service that serves the trained churn model."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException

try:  # pragma: no cover - import flexibility for direct script execution
    from api.schema import Customer
except ImportError:  # pragma: no cover
    from schema import Customer


ROOT = Path(__file__).resolve().parents[1]


def get_model_path() -> Path:
    return Path(os.getenv("MODEL_PATH", str(ROOT / "models/model.pkl")))


def load_model(model_path: str | Path | None = None):
    path = Path(model_path) if model_path else get_model_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found at {path}. Train the model first with `python src/train.py`."
        )
    return joblib.load(path)


def create_app(model_path: str | Path | None = None) -> FastAPI:
    resolved_model_path = Path(model_path) if model_path else get_model_path()
    app = FastAPI(title="Churn Scoring API", version="1.0")

    @lru_cache(maxsize=1)
    def ready_model():
        model = load_model(resolved_model_path)
        probe = Customer.model_json_schema()["example"]
        probability = float(model.predict_proba(pd.DataFrame([probe]))[0, 1])
        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Invalid model probability")
        return model

    @app.get("/health")
    def health():
        try:
            ready_model()
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Model is unavailable or incompatible") from exc
        return {
            "status": "ok",
            "model": resolved_model_path.name,
            "ready": True,
        }

    @app.post("/predict")
    def predict(customer: Customer):
        try:
            model = ready_model()
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Model is unavailable or incompatible") from exc

        payload = customer.model_dump() if hasattr(customer, "model_dump") else customer.dict()
        row = pd.DataFrame([payload])
        try:
            probability = float(model.predict_proba(row)[0, 1])
            if not np.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError("Invalid model probability")
        except Exception as exc:
            raise HTTPException(status_code=503, detail="Prediction is unavailable") from exc

        return {
            "churn_probability": round(probability, 4),
            "will_churn": bool(probability >= 0.5),
            "risk": "high" if probability >= 0.5 else "low",
            "threshold": 0.5,
        }

    return app



app = create_app()
@app.get("/")
def root():
    return {"status": "ok"}
