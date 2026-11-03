import numpy as np

from soulhome.geometry.polygon import polygon_iou
from soulhome.layout.horizonnet import decode_layout, from_horizonnet_raw
from soulhome.synthetic import horizonnet_outputs


def test_perfect_layout_decodes_exactly(room):
    c, f, k = horizonnet_outputs(room)
    dec = decode_layout(c, f, k, camera_height=1.5)
    assert dec.method == "corners+line_fit"
    assert len(dec.polygon) == 6
    assert abs(dec.ceiling_height - 2.7) < 0.01
    assert polygon_iou(dec.polygon, room.polygon) > 0.995


def test_noisy_layout_is_robust(room):
    rng = np.random.default_rng(0)
    c, f, k = horizonnet_outputs(room)
    c = c + rng.normal(0, 0.004, c.shape)
    f = f + rng.normal(0, 0.004, f.shape)
    dec = decode_layout(c, f, k * 0.9, camera_height=1.5)
    assert len(dec.polygon) == 6
    assert abs(dec.ceiling_height - 2.7) < 0.05
    assert polygon_iou(dec.polygon, room.polygon) > 0.95


def test_missing_corners_falls_back_to_contour(room):
    c, f, _ = horizonnet_outputs(room)
    dec = decode_layout(c, f, np.zeros_like(c), camera_height=1.5)
    assert dec.method == "boundary_contour"
    assert polygon_iou(dec.polygon, room.polygon) > 0.97


def test_wrong_camera_height_scales_room(room):
    c, f, k = horizonnet_outputs(room)
    dec = decode_layout(c, f, k, camera_height=3.0)  # 2x the true height -> 4x the area
    from shapely.geometry import Polygon

    assert abs(Polygon(dec.polygon).area / Polygon(room.polygon).area - 4.0) < 0.05


def test_raw_adapter_sign_convention():
    y_bon = np.stack([np.full(8, -0.3), np.full(8, 0.6)])  # HorizonNet: positive = below horizon
    ceil, floor, cor = from_horizonnet_raw(y_bon, np.zeros(8))
    assert (ceil > 0).all() and (floor < 0).all()
    np.testing.assert_allclose(cor, 0.5)
