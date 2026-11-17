from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
import os
import threading
import time
import uuid
from typing import Any

from .analysis import analyze_payload
from .models import AnalysisOptions
from .series import RainfallSeries


@dataclass
class Job:
    id: str
    status: str
    created_at: str
    updated_at: str
    result: dict[str, Any] | None = None
    error: str | None = None

    def public(self) -> dict[str, Any]:
        return {"job_id": self.id, "status": self.status, "created_at": self.created_at,
                "updated_at": self.updated_at, "error": self.error}


_lock = threading.Lock()
_jobs: dict[str, tuple[Job, float]] = {}
_executor = ThreadPoolExecutor(max_workers=max(1, int(os.getenv("ANALYSIS_WORKERS", "2"))))
_ttl = max(60, int(os.getenv("JOB_TTL_SECONDS", "3600")))


def _prune() -> None:
    now = time.time()
    expired = [job_id for job_id, (_, created) in _jobs.items() if now - created > _ttl]
    for job_id in expired:
        _jobs.pop(job_id, None)
    while len(_jobs) >= 100:
        oldest = min(_jobs, key=lambda job_id: _jobs[job_id][1])
        _jobs.pop(oldest, None)


def _run(job_id: str, series: list[RainfallSeries], options: AnalysisOptions) -> None:
    with _lock:
        entry = _jobs.get(job_id)
        if not entry:
            return
        job = entry[0]
        job.status = "running"
        job.updated_at = datetime.now(timezone.utc).isoformat()
    try:
        result = analyze_payload(series, options)
        with _lock:
            job.result = result
            job.status = "completed"
            job.updated_at = datetime.now(timezone.utc).isoformat()
    except Exception as exc:  # Keep worker failure in the job resource.
        with _lock:
            job.status = "failed"
            job.error = str(exc)[:500]
            job.updated_at = datetime.now(timezone.utc).isoformat()


def create_job(series: list[RainfallSeries], options: AnalysisOptions) -> Job:
    with _lock:
        _prune()
        now = datetime.now(timezone.utc).isoformat()
        job = Job(uuid.uuid4().hex, "queued", now, now)
        _jobs[job.id] = (job, time.time())
    _executor.submit(_run, job.id, series, options)
    return job


def get_job(job_id: str) -> Job | None:
    with _lock:
        _prune()
        entry = _jobs.get(job_id)
        return entry[0] if entry else None


def delete_job(job_id: str) -> bool:
    with _lock:
        return _jobs.pop(job_id, None) is not None
