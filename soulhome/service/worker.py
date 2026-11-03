"""Job execution: status transitions, retry vs dead-letter decisions, result bookkeeping."""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor

from ..pipeline.run import PIPELINE_VERSION, ScanMeta, run_scan
from ..pipeline.stages import PipelineError
from .store import ArtifactStore, JobStore

log = logging.getLogger("soulhome.worker")


class RetryableJobError(Exception):
    """Raised to the queue runtime (Modal retries / SQS visibility timeout) to request a retry."""


def process_job(job_id: str, store: JobStore, artifacts: ArtifactStore, inference=None):
    job = store.get(job_id)
    if job is None:
        raise KeyError(job_id)
    if job.status == "succeeded":  # at-least-once delivery: duplicate messages are no-ops
        return job
    if job.status == "failed":
        return job
    artifacts.refresh()
    job = store.update(job_id, status="running", attempts=job.attempts + 1, started_at=time.time(), error=None)
    out_dir = artifacts.result_dir(job_id, PIPELINE_VERSION)
    try:
        model = run_scan(
            job_id,
            job.tier,
            artifacts.input_dir(job_id),
            out_dir,
            ScanMeta(**job.meta),
            inference=inference,
            on_stage=lambda s: store.update(job_id, current_stage=s),
        )
    except Exception as e:  # noqa: BLE001
        err = e if isinstance(e, PipelineError) else PipelineError("worker", "INTERNAL", f"{type(e).__name__}: {e}", True)
        retry = err.retryable and job.attempts < job.max_attempts
        log.warning("job %s attempt %d failed: %s (retry=%s)", job_id, job.attempts, err, retry)
        job = store.update(
            job_id,
            status="retrying" if retry else "failed",
            error=err.as_dict(),
            dead_lettered=not retry,
            finished_at=None if retry else time.time(),
        )
        if retry:
            raise RetryableJobError(str(err)) from e
        return job
    artifacts.commit()
    room = model.rooms[0]
    result = {"pipeline_version": PIPELINE_VERSION, "model": "model.json", "floorplan": "floorplan.svg"}
    if (out_dir / "mesh_clean.glb").exists():
        result["mesh"] = "mesh_clean.glb"
    summary = {
        "floor_area_m2": room.floor_area_m2,
        "ceiling_height_m": room.ceiling_height_m,
        "walls": len(room.walls),
        "doors": sum(o.type == "door" for o in room.openings),
        "windows": sum(o.type == "window" for o in room.openings),
        "objects": len(room.objects),
        "confidence": model.quality.confidence,
        "warnings": model.quality.warnings,
        "stages": [{"name": s.name, "status": s.status, "duration_ms": s.duration_ms} for s in model.provenance.stages],
    }
    return store.update(job_id, status="succeeded", result=result, summary=summary, current_stage=None, finished_at=time.time())


class LocalDispatcher:
    """In-process queue for local dev/tests: thread pool + exponential backoff retries."""

    def __init__(self, store: JobStore, artifacts: ArtifactStore, inference=None, workers: int = 2, base_delay: float = 0.5):
        self.store, self.artifacts, self.inference = store, artifacts, inference
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.base_delay = base_delay

    def __call__(self, job_id: str):
        return self.pool.submit(self._run, job_id)

    def _run(self, job_id: str):
        attempt = 0
        while True:
            try:
                return process_job(job_id, self.store, self.artifacts, self.inference)
            except RetryableJobError:
                time.sleep(self.base_delay * 2**attempt)
                attempt += 1
            except Exception:  # noqa: BLE001
                log.exception("job %s crashed outside the pipeline", job_id)
                return None
