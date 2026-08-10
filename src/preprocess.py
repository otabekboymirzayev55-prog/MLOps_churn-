"""Data loading and shared preprocessing for training and serving."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Churn"
DROP = ["customerID"]


def load_data(csv_path: str | Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the churn dataset and return features and target."""

    df = pd.read_csv(csv_path)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    df = df.drop(columns=[column for column in DROP if column in df.columns])

    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")

    y = (df[TARGET] == "Yes").astype(int)
    X = df.drop(columns=[TARGET])
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create the shared preprocessing pipeline used by train and serve."""

    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

