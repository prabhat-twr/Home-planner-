"""Opt-in: runs the real HorizonNet + Depth Anything V2 + YOLO11 path on a real panorama.
Enable with SOULHOME_MODEL_TESTS=1 (downloads ~200 MB of weights)."""

import os
import urllib.request

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("SOULHOME_MODEL_TESTS") != "1", reason="set SOULHOME_MODEL_TESTS=1")


def test_real_panorama(tmp_path):
    from soulhome.inference.models import TorchInference
    from soulhome.pipeline.run import ScanMeta, run_scan
    from soulhome.service.api import SAMPLE_PANORAMA_URL

    d = tmp_path / "in"
    d.mkdir()
    with urllib.request.urlopen(SAMPLE_PANORAMA_URL, timeout=30) as r:
        (d / "panorama.png").write_bytes(r.read())
    m = run_scan("real", "panorama", d, tmp_path / "out", ScanMeta(camera_height_m=1.6, north_deg=0), inference=TorchInference())
    room = m.rooms[0]
    # sanity envelope for a hotel bedroom; exact values are pinned in the README run log
    assert 6 < room.floor_area_m2 < 25
    assert 2.2 < room.ceiling_height_m < 3.2
    labels = {o.catalog_id for o in room.objects}
    assert {"bed", "tv"} <= labels
