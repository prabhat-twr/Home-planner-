import numpy as np

from soulhome.objects.catalog import CATALOG, resolve
from soulhome.objects.placement import nms_equirect, place_perspective
from soulhome.spatial.zones import bearing, direction_16, direction_8, grid_zone, wall_facing


def test_catalog_has_76_unique_entries():
    assert len(CATALOG) == 76
    assert len({e.id for e in CATALOG}) == 76
    assert resolve("couch").id == "sofa"  # COCO name
    assert resolve("Dining Table").id == "dining_table"
    assert resolve("pooja mandir").id == "pooja_mandir"
    assert resolve("spaceship") is None


def test_bearings_and_directions():
    assert bearing(0, 1, 0) == 0
    assert bearing(1, 0, 0) == 90
    assert bearing(0, 1, 90) == 90  # +Y points east
    assert direction_16(0) == "N" and direction_16(11.0) == "N" and direction_16(12) == "NNE"
    assert direction_16(225) == "SW" and direction_8(350) == "N" and direction_8(200) == "S"


def test_grid_zone_and_wall_facing():
    sq = np.array([(0, 0), (3, 0), (3, 3), (0, 3)], float)
    assert grid_zone((1.5, 1.5), sq, 0) == "BRAHMASTHAN"
    assert grid_zone((2.9, 2.9), sq, 0) == "NE"
    assert grid_zone((0.1, 0.1), sq, 0) == "SW"
    assert grid_zone((2.9, 2.9), sq, 180) == "SW"
    # bottom wall of a CCW square faces into the room = north when +Y is north
    assert wall_facing((0, 0), (3, 0), 0) == 0


def test_perspective_backprojection():
    sq = np.array([(-5, -5), (5, -5), (5, 5), (-5, 5)], float)
    k = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1.0]])
    # OpenCV camera at 1.5 m looking along +Y (z_cam -> +Y_world, y_cam -> -Z_world)
    pose = np.eye(4)
    pose[:3, :3] = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    pose[:3, 3] = [0, 0, 1.5]
    # floor point (0, 3, 0) projects to v = 240 + 500 * 1.5 / 3 = 490
    det = {"label": "potted plant", "confidence": 0.8, "bbox": [300, 380, 340, 490]}
    p = place_perspective(det, k, pose, sq)
    assert abs(p.position[0]) < 1e-6
    assert abs(p.position[1] - (3.0 + 0.2)) < 1e-6  # front contact + half catalog depth


def test_nms_across_seam():
    dets = [
        {"label": "sofa", "confidence": 0.9, "bbox": [1000, 200, 30, 300]},  # wraps the seam
        {"label": "sofa", "confidence": 0.7, "bbox": [1005, 205, 25, 295]},
        {"label": "tv", "confidence": 0.7, "bbox": [400, 100, 450, 150]},
    ]
    kept = nms_equirect(dets, 1024)
    assert len(kept) == 2
