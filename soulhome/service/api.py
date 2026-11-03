"""REST API for mobile clients (versioned under /v1).

Upload -> 202 + job id -> poll status (with Retry-After hints) -> fetch the spatial model.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from ..contract import SCHEMA_VERSION, SpatialModel, Tier
from ..pipeline.run import IMAGE_EXTS, MESH_EXTS, PIPELINE_VERSION, ScanMeta
from ..spatial.zones import zone_report
from .store import TERMINAL, ArtifactStore, IdempotencyConflict, Job, JobStore, request_hash

MAX_UPLOAD_BYTES = 300 * 1024 * 1024
ALLOWED_EXTS = set(MESH_EXTS) | set(IMAGE_EXTS) | {"json", "npz"}
STATIC = Path(__file__).parent / "static"
SAMPLE_PANORAMA_URL = "https://raw.githubusercontent.com/sunset1995/HorizonNet/master/assets/demo.png"


def _safe_name(name: str) -> str:
    name = Path(name or "upload").name
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext not in ALLOWED_EXTS:
        raise HTTPException(415, f"unsupported file type .{ext}")
    return name


def create_app(store: JobStore, artifacts: ArtifactStore, dispatch: Callable[[str], object]) -> FastAPI:
    app = FastAPI(
        title="Soul Home reconstruction API",
        version=PIPELINE_VERSION,
        description="Scan upload -> async reconstruction -> versioned spatial data contract "
        f"(schema {SCHEMA_VERSION}) for the Vastu / Feng Shui rule engine.",
    )

    def status_body(job: Job, request: Request) -> dict:
        base = str(request.base_url).rstrip("/")
        body = job.model_dump(exclude={"request_hash", "files"})
        body["links"] = {"self": f"{base}/v1/scans/{job.id}"}
        if job.status == "succeeded":
            body["links"].update(
                model=f"{base}/v1/scans/{job.id}/model",
                floorplan=f"{base}/v1/scans/{job.id}/floorplan.svg",
                zones=f"{base}/v1/scans/{job.id}/zones",
            )
            if job.result and "mesh" in job.result:
                body["links"]["mesh"] = f"{base}/v1/scans/{job.id}/mesh.glb"
        if job.status not in TERMINAL:
            body["poll_after_s"] = 2 if job.status == "queued" else 3
        return body

    def enqueue(job: Job, request: Request) -> JSONResponse:
        try:
            job, created = store.create(job)
        except IdempotencyConflict as e:
            raise HTTPException(409, str(e)) from e
        if created:
            artifacts.commit()
            try:
                dispatch(job.id)
            except Exception as e:  # noqa: BLE001 - queue unavailable: fail visibly, not silently
                store.update(job.id, status="failed", error={"stage": "dispatch", "code": "QUEUE_UNAVAILABLE", "message": str(e), "retryable": True})
                raise HTTPException(503, "job queue unavailable") from e
        body = status_body(store.get(job.id), request)
        return JSONResponse(body, status_code=202 if created else 200, headers={"Location": body["links"]["self"]})

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index():
        return (STATIC / "index.html").read_text()

    @app.get("/healthz")
    def healthz():
        return {"ok": True, "pipeline_version": PIPELINE_VERSION, "schema_version": SCHEMA_VERSION}

    @app.post("/v1/scans", status_code=202)
    async def create_scan(
        request: Request,
        tier: Tier = Form(...),
        metadata: str = Form("{}", description="ScanMeta JSON: camera_height_m, north_deg, room_name, up_axis, units"),
        files: list[UploadFile] = File(...),
        idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
    ):
        try:
            meta = ScanMeta(**json.loads(metadata or "{}")).model_dump(exclude_none=True)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(422, f"invalid metadata: {e}") from e
        # stage uploads in a temp dir, hash them, then move under the scan's input prefix
        staged, digests, total = [], [], 0
        tmp = Path(tempfile.mkdtemp(prefix="upload_"))
        for f in files:
            name = _safe_name(f.filename)
            h = hashlib.sha256()
            dst = tmp / name
            with dst.open("wb") as out:
                while chunk := await f.read(1 << 20):
                    total += len(chunk)
                    if total > MAX_UPLOAD_BYTES:
                        raise HTTPException(413, "upload too large")
                    h.update(chunk)
                    out.write(chunk)
            staged.append(dst)
            digests.append((name, h.hexdigest()))
        job = Job(tier=tier.value, meta=meta, files=[n for n, _ in digests], idempotency_key=idempotency_key,
                  request_hash=request_hash(tier.value, meta, digests))
        dst_dir = artifacts.input_dir(job.id)
        dst_dir.mkdir(parents=True, exist_ok=True)
        for p in staged:
            shutil.move(str(p), dst_dir / p.name)
        shutil.rmtree(tmp, ignore_errors=True)
        resp = enqueue(job, request)
        if resp.status_code == 200:  # duplicate of an earlier request: drop the orphaned copy
            shutil.rmtree(dst_dir.parent, ignore_errors=True)
        return resp

    @app.post("/v1/demo/panorama-photo", status_code=202)
    def create_photo_demo(request: Request):
        """Process a real indoor panorama (the MIT-licensed HorizonNet sample) through the GPU path:
        HorizonNet layout + Depth Anything V2 + YOLO11."""
        import urllib.request

        job = Job(tier="panorama")
        d = artifacts.input_dir(job.id)
        d.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(SAMPLE_PANORAMA_URL, timeout=20) as r:
                (d / "panorama.png").write_bytes(r.read())
        except Exception as e:  # noqa: BLE001
            raise HTTPException(502, f"could not fetch sample panorama: {e}") from e
        job.meta = ScanMeta(camera_height_m=1.6, north_deg=0.0, room_name="bedroom (sample photo)").model_dump(exclude_none=True)
        job.files = ["panorama.png"]
        return enqueue(job, request)

    @app.post("/v1/demo/{tier}", status_code=202)
    def create_demo(tier: Tier, request: Request, layout_net: bool = False):
        """Generate a synthetic L-shaped room scan for the chosen tier and process it."""
        from ..demo import write_mesh_demo, write_panorama_demo

        job = Job(tier=tier.value)
        d = artifacts.input_dir(job.id)
        meta = write_panorama_demo(d, with_layout=layout_net) if tier is Tier.panorama else write_mesh_demo(d, tier.value)
        job.meta = ScanMeta(**meta, layout_backend="horizonnet" if layout_net else "auto").model_dump(exclude_none=True)
        job.files = sorted(p.name for p in d.iterdir())
        return enqueue(job, request)

    @app.get("/v1/scans")
    def list_scans(request: Request, limit: int = 20):
        return [status_body(j, request) for j in store.list(limit)]

    @app.get("/v1/scans/{scan_id}")
    def get_scan(scan_id: str, request: Request):
        job = store.get(scan_id)
        if job is None:
            raise HTTPException(404, "scan not found")
        body = status_body(job, request)
        headers = {"Retry-After": str(body["poll_after_s"])} if "poll_after_s" in body else {}
        return JSONResponse(body, headers=headers)

    def _result_file(scan_id: str, name: str) -> Path:
        job = store.get(scan_id)
        if job is None:
            raise HTTPException(404, "scan not found")
        if job.status != "succeeded":
            raise HTTPException(409, f"scan is {job.status}")
        artifacts.refresh()
        p = artifacts.result_dir(scan_id, job.result["pipeline_version"]) / name
        if not p.exists():
            raise HTTPException(404, f"{name} not available")
        return p

    @app.get("/v1/scans/{scan_id}/model", response_model=SpatialModel)
    def get_model(scan_id: str):
        return Response(_result_file(scan_id, "model.json").read_text(), media_type="application/json")

    @app.get("/v1/scans/{scan_id}/floorplan.svg")
    def get_floorplan(scan_id: str):
        return Response(_result_file(scan_id, "floorplan.svg").read_text(), media_type="image/svg+xml")

    @app.get("/v1/scans/{scan_id}/mesh.glb")
    def get_mesh(scan_id: str):
        return FileResponse(_result_file(scan_id, "mesh_clean.glb"), media_type="model/gltf-binary")

    @app.get("/v1/scans/{scan_id}/zones")
    def get_zones(scan_id: str, system: str = "16"):
        if system not in ("16", "8", "grid"):
            raise HTTPException(422, "system must be 16, 8 or grid")
        model = SpatialModel.model_validate_json(_result_file(scan_id, "model.json").read_text())
        try:
            return {"system": system, "north_deg": model.north_deg, "rooms": zone_report(model, system)}
        except ValueError as e:
            raise HTTPException(422, str(e)) from e

    @app.post("/v1/scans/{scan_id}/redrive", status_code=202)
    def redrive(scan_id: str, request: Request):
        """Move a dead-lettered scan back onto the queue (after a fix has been deployed)."""
        job = store.get(scan_id)
        if job is None:
            raise HTTPException(404, "scan not found")
        if job.status != "failed":
            raise HTTPException(409, f"scan is {job.status}")
        store.update(scan_id, status="queued", attempts=0, dead_lettered=False, error=None)
        dispatch(scan_id)
        return status_body(store.get(scan_id), request)

    @app.get("/v1/metrics")
    def metrics():
        return store.metrics()

    return app
