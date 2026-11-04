import cv2
import numpy as np
import pytest

from soulhome.geometry.equirect import pixel_to_lonlat
from soulhome.geometry.panorama_input import to_equirect
from soulhome.pipeline.run import ScanMeta, run_scan
from soulhome.pipeline.stages import PipelineError


def _coded_equirect(w=2048):
    """Channel 0 encodes longitude, channel 1 latitude (both 0..250)."""
    h = w // 2
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    lon, lat = pixel_to_lonlat(uu, vv, w, h)
    img = np.zeros((h, w, 3), np.float32)
    img[..., 0] = (lon + np.pi) / (2 * np.pi) * 250
    img[..., 1] = (lat + np.pi / 2) / np.pi * 250
    return img


def _cylindrical_from(eq, hfov_deg, vfov_deg, height):
    """Render what a phone Pano would produce from an equirect scene."""
    h_e, w_e = eq.shape[:2]
    f = (height / 2) / np.tan(np.radians(vfov_deg) / 2)
    width = int(round(f * np.radians(hfov_deg)))
    xx, yy = np.meshgrid(np.arange(width), np.arange(height))
    lon = (xx - (width - 1) / 2) / f
    lat = np.arctan(((height - 1) / 2 - yy) / f)
    mx = ((lon + np.pi) / (2 * np.pi) * w_e - 0.5).astype(np.float32)
    my = ((np.pi / 2 - lat) / np.pi * h_e - 0.5).astype(np.float32)
    return cv2.remap(eq, mx, my, cv2.INTER_LINEAR)


def test_equirect_passthrough():
    img = np.zeros((1000, 2000, 3), np.uint8)
    out, cov = to_equirect(img)
    assert out.shape == (1024, 2048, 3) and cov.projection == "equirectangular" and cov.full_circle


def test_iphone_pano_dimensions_are_read_as_partial_cylindrical():
    out, cov = to_equirect(np.zeros((3784, 10840, 3), np.uint8))
    assert cov.projection == "cylindrical"
    assert 195 < cov.hfov_deg < 225 and abs(cov.vfov_deg - 65) < 0.5
    assert out.shape == (1024, 2048, 3)


@pytest.mark.parametrize("hfov", [360, 210])
def test_cylindrical_resampling_recovers_the_sphere(hfov):
    eq = _coded_equirect()
    cyl = _cylindrical_from(eq, hfov, 65, 900)
    out, cov = to_equirect(cyl, 2048, "cylindrical", vfov_deg=65)
    assert abs(cov.hfov_deg - hfov) < 1.5
    valid = (out[..., 0] > 0) | (out[..., 1] > 0)
    h, w = out.shape[:2]
    uu, vv = np.meshgrid(np.arange(w), np.arange(h))
    lon, lat = pixel_to_lonlat(uu, vv, w, h)
    inside = (np.abs(np.degrees(lon)) < hfov / 2 - 2) & (np.abs(np.degrees(lat)) < 30)
    assert valid[inside].mean() > 0.99
    diff = np.abs(out[inside] - eq[inside][:, :])[:, :2]
    assert np.median(diff) < 1.0  # < 1/250 of the range: sub-pixel accurate


def test_rejects_portrait_and_too_narrow(tmp_path):
    d = tmp_path / "in"
    d.mkdir()
    cv2.imwrite(str(d / "photo.jpg"), np.zeros((300, 400, 3), np.uint8))
    with pytest.raises(PipelineError) as e:
        run_scan("t", "panorama", d, tmp_path / "o", ScanMeta())
    assert e.value.code == "NOT_A_PANORAMA" and not e.value.retryable
    cv2.imwrite(str(d / "photo.jpg"), np.zeros((400, 1200, 3), np.uint8))
    with pytest.raises(PipelineError) as e:
        run_scan("t", "panorama", d, tmp_path / "o", ScanMeta(pano_hfov_deg=90))
    assert e.value.code == "PANORAMA_TOO_NARROW"


def test_phone_pano_is_rejected_by_default(tmp_path):
    from soulhome.demo import write_panorama_demo

    d = tmp_path / "in"
    meta = write_panorama_demo(d, with_layout=True)
    (d / "panorama.png").unlink()
    cv2.imwrite(str(d / "panorama.jpg"), np.full((3784, 10840, 3), 128, np.uint8))
    with pytest.raises(PipelineError) as e:
        run_scan("t", "panorama", d, tmp_path / "o", ScanMeta(**meta))
    assert e.value.code == "PANORAMA_COVERAGE_INSUFFICIENT" and not e.value.retryable
    assert "Pano" in e.value.message and "360" in e.value.message


def test_partial_pano_runs_when_allowed_with_low_confidence(tmp_path):
    from soulhome.demo import write_panorama_demo

    d = tmp_path / "in"
    meta = write_panorama_demo(d, with_layout=True)
    full = run_scan("t", "panorama", d, tmp_path / "o1", ScanMeta(**meta))
    (d / "panorama.png").unlink()
    cv2.imwrite(str(d / "panorama.jpg"), np.full((3784, 10840, 3), 128, np.uint8))
    part = run_scan("t", "panorama", d, tmp_path / "o2", ScanMeta(**meta, allow_partial_panorama=True))
    assert any("partial panorama" in w for w in part.quality.warnings)
    assert part.quality.confidence <= 0.35 < full.quality.confidence
    load = next(s for s in part.provenance.stages if s.name == "load_inputs")
    assert load.detail["panorama"]["projection"] == "cylindrical"
