# Trend analysis

This repository contains the backend service for the rainfall trend-analysis website.

## Start the API

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for the API interface. See [backend/README.md](backend/README.md) for the upload formats, endpoints, frontend connection, and scientific limitations.

The frontend is not yet present in this repository. GitHub Pages serves static files only; host this Python API separately and configure the frontend's API URL and the backend's `CORS_ORIGINS` before production use.
