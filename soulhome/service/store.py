"""Job state + artifact storage behind small interfaces, so the same worker/API code runs against an
in-process dict (tests, local dev), a Modal Dict + Volume (this deployment), or Postgres + S3 (see
``db/schema.sql`` for the AWS layout)."""

from __future__ import annotations

import hashlib
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Protocol

from pydantic import BaseModel, Field

Status = Literal["queued", "running", "retrying", "succeeded", "failed"]
ALLOWED = {
    "queued": {"running", "failed"},
    "running": {"succeeded", "failed", "retrying"},
    "retrying": {"running", "failed"},
    "succeeded": set(),
    "failed": {"queued"},  # manual re-drive from the dead-letter state
}
TERMINAL = {"succeeded", "failed"}


class Job(BaseModel):
    id: str = Field(default_factory=lambda: "scn_" + uuid.uuid4().hex[:16])
    tier: str
    status: Status = "queued"
    meta: dict = {}
    files: list[str] = []
    idempotency_key: str | None = None
    request_hash: str | None = None
    attempts: int = 0
    max_attempts: int = 3
    current_stage: str | None = None
    error: dict | None = None
    dead_lettered: bool = False
    result: dict | None = None
    summary: dict | None = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None


class KV(Protocol):
    def get(self, key: str, default: Any = None) -> Any: ...
    def put(self, key: str, value: Any) -> None: ...
    def put_if_absent(self, key: str, value: Any) -> bool: ...
    def keys(self) -> Iterable[str]: ...


class LocalKV:
    def __init__(self):
        self._d: dict[str, Any] = {}
        self._lock = threading.Lock()

    def get(self, key, default=None):
        with self._lock:
            return self._d.get(key, default)

    def put(self, key, value):
        with self._lock:
            self._d[key] = value

    def put_if_absent(self, key, value) -> bool:
        with self._lock:
            if key in self._d:
                return False
            self._d[key] = value
            return True

    def keys(self):
        with self._lock:
            return list(self._d.keys())


class ModalKV:
    """modal.Dict adapter. ``put(..., skip_if_exists=True)`` is atomic, which is what makes
    idempotency keys safe under concurrent retries from a flaky mobile connection."""

    def __init__(self, d):
        self.d = d

    def get(self, key, default=None):
        return self.d.get(key, default)

    def put(self, key, value):
        self.d[key] = value

    def put_if_absent(self, key, value) -> bool:
        return bool(self.d.put(key, value, skip_if_exists=True))

    def keys(self):
        return list(self.d.keys())


class IdempotencyConflict(Exception):
    pass


class InvalidTransition(Exception):
    pass


class JobStore:
    def __init__(self, kv: KV):
        self.kv = kv

    def create(self, job: Job) -> tuple[Job, bool]:
        if job.idempotency_key:
            if not self.kv.put_if_absent(f"idem:{job.idempotency_key}", job.id):
                existing = self.get(self.kv.get(f"idem:{job.idempotency_key}"))
                if existing is None:  # key reserved by a create that has not finished writing yet
                    raise IdempotencyConflict("request with this Idempotency-Key is in flight")
                if existing.request_hash != job.request_hash:
                    raise IdempotencyConflict("Idempotency-Key reused with a different payload")
                return existing, False
        self.kv.put(f"job:{job.id}", job.model_dump())
        return job, True

    def get(self, job_id: str | None) -> Job | None:
        if not job_id:
            return None
        raw = self.kv.get(f"job:{job_id}")
        return Job(**raw) if raw else None

    def update(self, job_id: str, **changes) -> Job:
        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        new_status = changes.get("status")
        if new_status and new_status != job.status and new_status not in ALLOWED[job.status]:
            raise InvalidTransition(f"{job.status} -> {new_status}")
        job = job.model_copy(update={**changes, "updated_at": time.time()})
        self.kv.put(f"job:{job_id}", job.model_dump())
        return job

    def list(self, limit: int = 50) -> list[Job]:
        jobs = [self.get(k[4:]) for k in self.kv.keys() if k.startswith("job:")]
        jobs = [j for j in jobs if j]
        return sorted(jobs, key=lambda j: -j.created_at)[:limit]

    def metrics(self) -> dict:
        """Per-status counts, per-stage failure counts and stage latency percentiles - enough to
        drive a "where do scans die" dashboard."""
        jobs = self.list(limit=10_000)
        by_status: dict[str, int] = {}
        by_tier: dict[str, int] = {}
        failures: dict[str, int] = {}
        stage_ms: dict[str, list[float]] = {}
        for j in jobs:
            by_status[j.status] = by_status.get(j.status, 0) + 1
            by_tier[j.tier] = by_tier.get(j.tier, 0) + 1
            if j.error:
                key = f"{j.error.get('stage')}/{j.error.get('code')}"
                failures[key] = failures.get(key, 0) + 1
            for s in (j.summary or {}).get("stages", []):
                stage_ms.setdefault(s["name"], []).append(s["duration_ms"])

        def pct(v, q):
            v = sorted(v)
            return round(v[min(len(v) - 1, int(q * len(v)))], 1)

        return {
            "jobs": len(jobs),
            "by_status": by_status,
            "by_tier": by_tier,
            "dead_lettered": sum(j.dead_lettered for j in jobs),
            "failures_by_stage": failures,
            "stage_latency_ms": {k: {"p50": pct(v, 0.5), "p95": pct(v, 0.95), "n": len(v)} for k, v in stage_ms.items()},
        }


def request_hash(tier: str, meta: dict, file_digests: list[tuple[str, str]]) -> str:
    import json

    h = hashlib.sha256()
    h.update(tier.encode())
    h.update(json.dumps(meta, sort_keys=True).encode())
    for name, digest in sorted(file_digests):
        h.update(name.encode())
        h.update(digest.encode())
    return h.hexdigest()


class ArtifactStore:
    """Versioned layout for large binary scan data:

        scans/{scan_id}/input/...                 immutable uploads
        scans/{scan_id}/results/v{pipeline}/...   one directory per pipeline version

    Re-processing with a new pipeline version never overwrites earlier results (on S3 the same
    prefixes get lifecycle rules: inputs -> IA after 30 days, superseded results -> expire)."""

    def __init__(self, root: str | Path, on_write: Callable[[], None] | None = None, on_read: Callable[[], None] | None = None):
        self.root = Path(root)
        self.on_write = on_write or (lambda: None)
        self.on_read = on_read or (lambda: None)

    def input_dir(self, scan_id: str) -> Path:
        return self.root / "scans" / scan_id / "input"

    def result_dir(self, scan_id: str, version: str) -> Path:
        return self.root / "scans" / scan_id / "results" / f"v{version}"

    def commit(self):
        self.on_write()

    def refresh(self):
        self.on_read()
