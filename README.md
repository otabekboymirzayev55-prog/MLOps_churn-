# Churn MLOps

Telecom churn prediction project built from the guide in `Churn_MLOps_Guide_EN.html`.

## What it does

- trains a churn classifier on the IBM telecom churn dataset
- tracks experiments in MLflow
- serves predictions through FastAPI
- exposes a Streamlit dashboard
- includes pytest checks, Docker, and GitHub Actions workflows

## Project layout

- `data/telco_churn.csv` - source dataset
- `src/preprocess.py` - shared feature pipeline
- `src/train.py` - training and MLflow tracking
- `api/schema.py` - typed request schema
- `api/main.py` - FastAPI service
- `dashboard.py` - Streamlit UI
- `tests/test_pipeline.py` - quality checks

## Run locally

Create a virtual environment and install dependencies:

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

Train the model:

```powershell
python src/train.py
```

Run the API:

```powershell
uvicorn api.main:app --reload
```

Run the dashboard:

```powershell
streamlit run dashboard.py
```

Run the tests:

```powershell
pytest
```

## Notes

- The repository includes the full telecom churn dataset and a trained model artifact.
- The same preprocessing code is used during training and serving to avoid train/serve skew.
- MLflow logs are written locally to `mlruns/` and are ignored by git.

