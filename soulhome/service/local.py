"""Local dev server: ``python -m soulhome.service.local`` (or ``uvicorn soulhome.service.local:app``)."""

from __future__ import annotations

import logging
import os

from .api import create_app
from .store import ArtifactStore, JobStore, LocalKV
from .worker import LocalDispatcher


def build(root: str | None = None, inference=None):
    store = JobStore(LocalKV())
    artifacts = ArtifactStore(root or os.environ.get("SOULHOME_DATA", "./data"))
    if inference is None and os.environ.get("SOULHOME_LOCAL_MODELS") == "1":
        from ..inference.models import TorchInference

        inference = TorchInference()
    dispatch = LocalDispatcher(store, artifacts, inference)
    return create_app(store, artifacts, dispatch), store, dispatch


logging.basicConfig(level=logging.INFO, format="%(message)s")
app, _, _ = build()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
