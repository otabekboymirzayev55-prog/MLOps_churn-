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
pip install -r requirements.lock.txt
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


## Start both services with Docker Compose

Run `docker compose up --build --wait api dashboard` from the repository root.
Training completes before the API starts; the dashboard waits for API readiness.
Open http://localhost:8501 (dashboard) and http://localhost:8000/docs (API).
Stop with `docker compose down`. To retrain, run `docker compose run --rm train`
and then `docker compose restart api dashboard`. The API caches its model until restart.

## Windows launcher

After installing dependencies and training the model, run `./start.ps1`.
The launcher starts both services and stops them when you press Ctrl+C.
The API health endpoint must return HTTP 200 before the dashboard starts.

## Configuration and deployment

`API_URL` is the API base URL, for example `https://your-api.example.com`,
without `/predict`. Streamlit secrets override the environment variable.
The default `http://127.0.0.1:8000` works when both processes share a machine.
Use `http://api:8000` inside Compose. HTTP requests originate from the Streamlit
server, so browser CORS changes are unnecessary.

`MODEL_PATH` selects a trusted joblib model. `/health` returns 503 when the
model cannot load or perform a prediction. Restart the API after replacing it.
Train and serve with the same `requirements.lock.txt` environment.
Retraining workflows save a model artifact; they do not automatically deploy it.
Download and validate the artifact before replacing a production model.

The Docker image starts only the API; Compose supplies training and dashboard
services. Expose port 8000 for the API and 8501 for Streamlit. A public deployment
still needs a hosting account, HTTPS routing and access controls appropriate
to its audience. No public hosting target is configured in this repository.

## Verification

Run `python -m pytest -q` for model quality, API validation, dashboard rendering,
missing/corrupt model handling and dashboard/API contract tests.
CI additionally checks Compose configuration and starts both containers.
Refresh dependency versions deliberately and rerun all tests before deployment.
