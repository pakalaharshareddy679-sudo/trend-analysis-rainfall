# Trend analysis API

FastAPI backend for the rainfall workflows in the NIH-DHAARA reference page. It accepts station rainfall CSVs or dated series sent by the React app, then returns annual/seasonal/monthly summaries, trend tests, daily extreme indices, SPI, and exploratory frequency estimates.

## Run locally on Windows

From the repository root in PowerShell:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs`. Health check: `http://127.0.0.1:8000/api/v1/health`.

If PowerShell blocks virtual-environment activation, use the environment's interpreter directly:

```powershell
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

## API flow

1. Submit JSON series to `POST /api/v1/jobs`, or a CSV to `POST /api/v1/jobs/csv`.
2. Poll the returned `status_url` until `status` is `completed`.
3. Read `result_url`; download the annual CSV from `/api/v1/jobs/{job_id}/annual.csv`.
4. Delete a result with `DELETE /api/v1/jobs/{job_id}`.

JSON request example:

```json
{
  "series": [
    {
      "name": "Station A",
      "dates": ["2001-01-01", "2001-01-02", "2001-01-03"],
      "values": [0, 12.4, 3.1],
      "unit": "mm",
      "frequency": "daily"
    }
  ],
  "options": {
    "rainy_day_mm": 1,
    "wet_day_mm": 2.5,
    "max_missing_per_year_pct": 100,
    "spi_scales": [1, 3, 6, 12],
    "drought_threshold": -1
  }
}
```

Supported CSV layouts include `Date,Station1,...`, `Station,Date,Rainfall`, `Year,Month,Day,Rainfall`, `Year,Jan,...,Dec`, `Year,Month,1,...,31`, and `Year,Rainfall`. Trace tokens `T`/`TR` are treated as zero; common negative missing-value codes are ignored. Select source units in the upload form (`mm`, `cm`, `in`, or `0.1mm`). Ambiguous or unrecognized layouts return an actionable 422 response.

## Connect the React app

Set the frontend API base URL to `http://127.0.0.1:8000` during local development. Configure Vite's `VITE_API_BASE_URL` or equivalent; do not hard-code this value in components. The backend allows `http://localhost:5173` by default. Set `CORS_ORIGINS` to a comma-separated list of exact origins when deploying the frontend elsewhere. For GitHub Pages the origin is the host only, for example `https://pakalaharshareddy679-sudo.github.io` (no repository path).

GitHub Pages serves static files and cannot run this Python API. Deploy this backend to a Python-capable host separately, then set the frontend API URL and backend `CORS_ORIGINS` for the deployed domains.

## Analysis notes and limits

- Mann-Kendall uses the tie-corrected variance and continuity correction; Sen slope is the median pairwise slope in the input time unit (annual trend is mm/year). Spearman, linear regression, Pettitt, and an explicitly labeled trend-free pre-whitening calculation are also returned.
- The IMD seasons are grouped as Winter (Jan-Feb), Pre-monsoon (Mar-May), Southwest monsoon (Jun-Sep), and Post-monsoon (Oct-Dec). Calendar-year annual totals remain distinct from any separately supplied hydrological-year series.
- Daily-only indices and frequency analysis are omitted for monthly/annual input. Extreme threshold percentiles are calculated over the selected base period. The Gumbel return levels are exploratory, not a substitute for a reviewed engineering frequency analysis.
- SPI uses month-of-year gamma fits and normal quantiles; scales with fewer than 10 calibration observations can be null. Confirm the exact SPI base period and treatment of zero rainfall against the supplied NIH-DHAARA report before publishing scientific results.
- Jobs and results live only in process memory. They are not written to disk; a process restart loses queued work and results, and a single-process host is recommended for this starter implementation. Large all-grid jobs need a durable queue and worker service.
- This API analyzes series supplied by the browser. It does not bundle IMD gridded data, parse NetCDF, or provide catchment shapefile weighting. The reference site handles NetCDF extraction in its separate browser explorer; submit its extracted series here.
- Defaults (`rainy_day_mm=1`, `wet_day_mm=2.5`, `max_missing_per_year_pct=100`) are UI defaults only. Verify thresholds and method conventions against the report and source MATLAB scripts before treating outputs as publication-ready.
