"""Streamlit dashboard for scoring telecom churn customers."""

from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

API_URL = st.secrets.get("API_URL", "http://127.0.0.1:8000")
def predict(customer):
   response = requests.post(f"{API_URL.rstrip('/')}/predict", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()

FIELD_ORDER = [
    ("gender", "Gender", ["Female", "Male"]),
    ("SeniorCitizen", "Senior citizen", [0, 1]),
    ("Partner", "Partner", ["Yes", "No"]),
    ("Dependents", "Dependents", ["Yes", "No"]),
    ("tenure", "Tenure", None),
    ("PhoneService", "Phone service", ["Yes", "No"]),
    ("MultipleLines", "Multiple lines", ["No", "Yes", "No phone service"]),
    ("InternetService", "Internet service", ["DSL", "Fiber optic", "No"]),
    ("OnlineSecurity", "Online security", ["Yes", "No", "No internet service"]),
    ("OnlineBackup", "Online backup", ["Yes", "No", "No internet service"]),
    ("DeviceProtection", "Device protection", ["Yes", "No", "No internet service"]),
    ("TechSupport", "Tech support", ["Yes", "No", "No internet service"]),
    ("StreamingTV", "Streaming TV", ["Yes", "No", "No internet service"]),
    ("StreamingMovies", "Streaming movies", ["Yes", "No", "No internet service"]),
    ("Contract", "Contract", ["Month-to-month", "One year", "Two year"]),
    ("PaperlessBilling", "Paperless billing", ["Yes", "No"]),
    ("PaymentMethod", "Payment method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]),
    ("MonthlyCharges", "Monthly charges", None),
    ("TotalCharges", "Total charges", None),
]

DEFAULTS = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 5,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 89.9,
    "TotalCharges": 450.5,
}


def build_payload() -> dict[str, Any]:
    left, right = st.columns(2)
    payload: dict[str, Any] = {}

    for index, (key, label, options) in enumerate(FIELD_ORDER):
        container = left if index % 2 == 0 else right
        with container:
            if options is None:
                if key in {"tenure"}:
                    payload[key] = int(st.number_input(label, min_value=0, max_value=100, value=int(DEFAULTS[key])))
                else:
                    payload[key] = float(st.number_input(label, min_value=0.0, value=float(DEFAULTS[key])))
            else:
                payload[key] = st.selectbox(label, options, index=options.index(DEFAULTS[key]))

    return payload


def call_api(payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(f"{API_URL.rstrip('/')}/predict", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


st.set_page_config(page_title="Churn Predictor", page_icon="📉", layout="centered")
st.title("Telecom Churn Predictor")
st.caption("Enter a customer profile and get a churn score from the API.")

with st.form("churn-form"):
    payload = build_payload()
    submitted = st.form_submit_button("Score customer")

if submitted:
    try:
        result = call_api(payload)
    except Exception as exc:  # pragma: no cover - UI only
        st.error(f"Could not reach the API: {exc}")
    else:
        st.metric("Churn probability", f"{result['churn_probability']:.2%}")
        st.write(f"Risk: `{result['risk']}`")
        st.write(f"Will churn: `{result['will_churn']}`")

