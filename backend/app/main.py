from __future__ import annotations

import csv
from io import StringIO
import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .analysis import _clean_value
from .jobs import create_job, delete_job, get_job
from .models import AnalysisOptions, AnalysisRequest, JobAccepted
from .series import materialize, parse_rainfall_csv


app = FastAPI(
    title="Trend analysis API",
    version="0.1.0",
    description="In-memory rainfall trend, extremes, frequency, and SPI analysis for the Trend analysis website.",
)

origins = [item.strip() for item in os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",") if item.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type"],
)


@app.get("/api/v1/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "trend-analysis-api", "storage": "memory-only"}


def _accepted(job_id: str) -> JobAccepted:
    return JobAccepted(job_id=job_id, status="queued", status_url=f"/api/v1/jobs/{job_id}",
                       result_url=f"/api/v1/jobs/{job_id}/result",
                       message="Analysis queued. Poll status_url until status is completed.")


@app.post("/api/v1/jobs", response_model=JobAccepted, status_code=202)
def submit_json_job(request: AnalysisRequest) -> JobAccepted:
    try:
        series = materialize(request.series)
        job = create_job(series, request.options)
        return _accepted(job.id)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/v1/jobs/csv", response_model=JobAccepted, status_code=202)
async def submit_csv_job(
    file: UploadFile = File(...),
    unit: str = Form(default="mm"),
    dayfirst: bool = Form(default=False),
    options_json: str = Form(default="{}"),
) -> JobAccepted:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=415, detail="Upload a .csv file")
    limit = max(1_000_000, int(os.getenv("MAX_UPLOAD_BYTES", "20971520")))
    content = bytearray()
    while chunk := await file.read(1024 * 1024):
        content.extend(chunk)
        if len(content) > limit:
            raise HTTPException(status_code=413, detail=f"CSV exceeds the {limit // 1_000_000} MB upload limit")
    try:
        parsed = parse_rainfall_csv(bytes(content), file.filename, unit, dayfirst)
        series = materialize(parsed)
        options = AnalysisOptions.model_validate_json(options_json)
        job = create_job(series, options)
        return _accepted(job.id)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        await file.close()


@app.get("/api/v1/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, object]:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or expired")
    return job.public()


@app.get("/api/v1/jobs/{job_id}/result")
def job_result(job_id: str) -> dict[str, object]:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or expired")
    if job.status in {"queued", "running"}:
        raise HTTPException(status_code=202, detail={"status": job.status, "message": "Analysis is still running"})
    if job.status == "failed":
        raise HTTPException(status_code=422, detail=job.error or "Analysis failed")
    return _clean_value(job.result or {})


@app.get("/api/v1/jobs/{job_id}/annual.csv")
def download_annual_csv(job_id: str) -> StreamingResponse:
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or expired")
    if job.status != "completed" or not job.result:
        raise HTTPException(status_code=409, detail="Result is not ready")
    buffer = StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["series", "year", "rainfall_mm", "MK_Z", "MK_p", "Sen_slope_mm_per_year", "Pettitt_year"])
    for result in job.result["series"]:
        trend = result.get("trend", {})
        mk = trend.get("mk", {})
        sen = trend.get("sen_slope_per_year")
        change = trend.get("pettitt", {}).get("change_year")
        for row in result.get("annual", []):
            writer.writerow([result["name"], row["year"], row["rainfall_mm"], mk.get("z"), mk.get("p"), sen, change])
    buffer.seek(0)
    return StreamingResponse(iter([buffer.getvalue()]), media_type="text/csv",
                             headers={"Content-Disposition": f'attachment; filename="rainfall-{job_id}.csv"'})


@app.delete("/api/v1/jobs/{job_id}", status_code=204)
def remove_job(job_id: str) -> None:
    if not delete_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found or expired")

