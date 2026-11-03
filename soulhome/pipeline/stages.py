"""Stage execution with timing, structured logs and typed failures."""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager

from ..contract import StageReport

log = logging.getLogger("soulhome.pipeline")


class PipelineError(Exception):
    """A failure attributed to one stage.

    ``retryable`` drives the job queue: transient infrastructure faults (GPU OOM, network, storage)
    are retried; deterministic failures (bad input, geometry that fails validation) go straight to
    the dead-letter state so we never burn GPU time on a scan that cannot succeed."""

    def __init__(self, stage: str, code: str, message: str, retryable: bool = False):
        super().__init__(f"[{stage}/{code}] {message}")
        self.stage, self.code, self.message, self.retryable = stage, code, message, retryable

    def as_dict(self) -> dict:
        return {"stage": self.stage, "code": self.code, "message": self.message, "retryable": self.retryable}


TRANSIENT = (TimeoutError, ConnectionError, MemoryError)


def classify(exc: BaseException) -> tuple[str, bool]:
    name = type(exc).__name__
    if isinstance(exc, TRANSIENT) or "OutOfMemory" in name or "CUDA" in str(exc)[:200]:
        return "TRANSIENT", True
    if isinstance(exc, OSError):
        return "IO_ERROR", True
    if isinstance(exc, (ValueError, KeyError, IndexError)):
        return "BAD_INPUT_OR_GEOMETRY", False
    return "INTERNAL", False


class StageRecorder:
    def __init__(self, scan_id: str, on_stage=None):
        self.scan_id = scan_id
        self.reports: list[StageReport] = []
        self.on_stage = on_stage  # callback(stage_name) for live status reporting

    def _log(self, **fields):
        log.info(json.dumps({"scan_id": self.scan_id, **fields}, default=str))

    @contextmanager
    def stage(self, name: str):
        detail: dict = {}
        if self.on_stage:
            self.on_stage(name)
        t0 = time.perf_counter()
        try:
            yield detail
        except PipelineError as e:
            self._finish(name, "failed", t0, {**detail, "error": e.as_dict()})
            raise
        except Exception as e:  # noqa: BLE001 - convert to a typed stage failure
            code, retryable = classify(e)
            err = PipelineError(name, code, f"{type(e).__name__}: {e}", retryable)
            self._finish(name, "failed", t0, {**detail, "error": err.as_dict()})
            raise err from e
        self._finish(name, "ok", t0, detail)

    def skip(self, name: str, reason: str):
        self.reports.append(StageReport(name=name, status="skipped", duration_ms=0.0, detail={"reason": reason}))
        self._log(stage=name, status="skipped", reason=reason)

    def _finish(self, name, status, t0, detail):
        ms = (time.perf_counter() - t0) * 1000
        self.reports.append(StageReport(name=name, status=status, duration_ms=round(ms, 2), detail=detail))
        self._log(stage=name, status=status, duration_ms=round(ms, 2), **{k: v for k, v in detail.items() if k != "error"},
                  error=detail.get("error"))
