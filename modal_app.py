"""Modal deployment.

    modal deploy modal_app.py        # web API + CPU pipeline workers + GPU inference
    modal serve  modal_app.py        # hot-reloading dev URL

Topology
    api (ASGI, CPU)            FastAPI; uploads land on the `soulhome-scans` Volume, job state in a
                               Modal Dict; enqueues with `process_scan.spawn()`.
    process_scan (CPU)         the geometry pipeline; Modal Retries for transient failures, our own
                               attempt counter decides retry vs dead-letter.
    Inference (GPU, T4)        HorizonNet (layout) + Depth Anything V2 metric-indoor (depth) +
                               YOLO11 (objects). Loaded once per container (`@modal.enter`), weights
                               cached on a Volume, scales to zero after 2 idle minutes. Only called
                               for real panorama uploads; mesh tiers never touch a GPU.
"""

from __future__ import annotations

import modal

APP_NAME = "soulhome"
app = modal.App(APP_NAME)

scans = modal.Volume.from_name("soulhome-scans", create_if_missing=True)
model_cache = modal.Volume.from_name("soulhome-model-cache", create_if_missing=True)
jobs = modal.Dict.from_name("soulhome-jobs", create_if_missing=True)

# pinned to the versions the test suite and the real-model run were validated with
CORE_PKGS = [
    "numpy==2.5.3",
    "scipy==1.18.1",
    "opencv-python-headless==5.0.0.93",
    "shapely==2.2.0",
    "trimesh==5.2.0",
    "fast_simplification==0.2.0",
    "pydantic==2.13.5",
    "fastapi[standard]==0.162.0",
    "python-multipart==0.0.32",
    "pillow==12.3.0",
    "usd-core==26.11",
]

cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(*CORE_PKGS)
    .add_local_python_source("soulhome")
)

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(*CORE_PKGS)
    .pip_install("torch==2.15.0", "torchvision==0.29.2")  # PyPI Linux wheels ship CUDA
    .pip_install("transformers==5.22.1", "ultralytics==8.4.200", "huggingface_hub==2.4.0")
    .env(
        {
            "HF_HOME": "/cache/hf",
            "YOLO_CONFIG_DIR": "/cache/ultralytics",
            "SOULHOME_YOLO_WEIGHTS": "/cache/yolo11s.pt",
            "SOULHOME_LAYOUT_VARIANT": "mp3d",
        }
    )
    .add_local_python_source("soulhome")
)


@app.cls(image=gpu_image, gpu="T4", volumes={"/cache": model_cache}, scaledown_window=120, timeout=600, max_containers=4)
class Inference:
    @modal.enter()
    def load(self):
        import os

        os.makedirs("/cache", exist_ok=True)
        os.chdir("/cache")  # ultralytics downloads release weights into the cwd on first use
        from soulhome.inference.models import TorchInference

        self.m = TorchInference()
        model_cache.commit()  # persist downloaded weights for the next cold start
        print(f"models loaded in {self.m.load_seconds:.1f}s: {self.m.models}")

    @modal.method()
    def info(self) -> dict:
        return self.m.models

    @modal.method()
    def depth_points(self, rgb):
        return self.m.depth_points(rgb).astype("float32")

    @modal.method()
    def detect(self, rgb):
        return self.m.detect(rgb)

    @modal.method()
    def layout(self, rgb):
        return self.m.layout(rgb)


class RemoteInference:
    """Adapter giving the CPU pipeline the InferenceBackend interface over the GPU class."""

    def __init__(self):
        self._cls = Inference()
        self._models = None

    @property
    def models(self):
        if self._models is None:
            self._models = self._cls.info.remote()
        return self._models

    def depth_points(self, rgb):
        return self._cls.depth_points.remote(rgb)

    def detect(self, rgb):
        return self._cls.detect.remote(rgb)

    def layout(self, rgb):
        return self._cls.layout.remote(rgb)


def _stores():
    from soulhome.service.store import ArtifactStore, JobStore, ModalKV

    return JobStore(ModalKV(jobs)), ArtifactStore("/data", on_write=scans.commit, on_read=scans.reload)


class _LazyInference:
    """Only touch the GPU class when a stage actually needs a model (mesh tiers and synthetic demos
    never do, so they never pay a GPU cold start)."""

    def __init__(self):
        self._real = None

    def _get(self):
        if self._real is None:
            self._real = RemoteInference()
        return self._real

    @property
    def models(self):
        return self._real.models if self._real else {}

    def depth_points(self, rgb):
        return self._get().depth_points(rgb)

    def detect(self, rgb):
        return self._get().detect(rgb)

    def layout(self, rgb):
        return self._get().layout(rgb)


@app.function(
    image=cpu_image,
    volumes={"/data": scans},
    cpu=2.0,
    memory=4096,
    timeout=900,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0, initial_delay=5.0),
)
def process_scan(job_id: str):
    import logging

    from soulhome.service.worker import process_job

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    store, artifacts = _stores()
    job = process_job(job_id, store, artifacts, inference=_LazyInference())
    return {"id": job.id, "status": job.status}


@app.function(image=cpu_image, volumes={"/data": scans}, min_containers=0, scaledown_window=300)
@modal.concurrent(max_inputs=32)
@modal.asgi_app(label="soulhome-api")
def api():
    from soulhome.service.api import create_app

    store, artifacts = _stores()
    return create_app(store, artifacts, dispatch=lambda job_id: process_scan.spawn(job_id))


@app.local_entrypoint()
def smoke(tier: str = "lidar"):
    """`modal run modal_app.py --tier panorama`: run one synthetic scan through the deployed
    worker path without going through HTTP."""
    import tempfile
    from pathlib import Path

    from soulhome.demo import write_mesh_demo, write_panorama_demo
    from soulhome.service.store import Job

    store, _ = _stores()
    job = Job(tier=tier)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        meta = write_panorama_demo(d) if tier == "panorama" else write_mesh_demo(d, tier)
        with scans.batch_upload(force=True) as batch:
            for p in d.iterdir():
                batch.put_file(str(p), f"/scans/{job.id}/input/{p.name}")
    job.meta = meta
    store.create(job)
    print(process_scan.remote(job.id))
    print(store.get(job.id).summary)
