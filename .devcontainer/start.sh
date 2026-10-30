#!/usr/bin/env bash
set -euo pipefail
python src/train.py
uvicorn api.main:app --host 0.0.0.0 --port 8000 &
api_pid=$!
trap 'kill "$api_pid" 2>/dev/null || true' EXIT
streamlit run dashboard.py --server.address=0.0.0.0 --server.port=8501
