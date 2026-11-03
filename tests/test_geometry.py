import numpy as np
import pytest

from soulhome.geometry.equirect import (
    equirect_to_perspective,
    lonlat_to_pixel,
    lonlat_to_vec,
    perspective_camera_rays,
    pixel_to_lonlat,
    vec_to_lonlat,
    yaw_pitch_rotation,
)
from soulhome.geometry.planes import align_floor, manhattan_yaw, ransac_plane, rotation_between
from soulhome.geometry.polygon import manhattan_regularize, polygon_iou, signed_area

from .conftest import poly_close


def test_pixel_lonlat_roundtrip():
    u = np.array([0, 100.3, 511.5, 1023])
    v = np.array([0, 50, 255.5, 511])
    lon, lat = pixel_to_lonlat(u, v, 1024, 512)
    u2, v2 = lonlat_to_pixel(lon, lat, 1024, 512)
    np.testing.assert_allclose(u2, u, atol=1e-9)
    np.testing.assert_allclose(v2, v, atol=1e-9)


def test_axes_convention():
    # centre column looks along +Y, right of centre along +X, top row is +Z
    np.testing.assert_allclose(lonlat_to_vec(0, 0), [0, 1, 0], atol=1e-12)
    np.testing.assert_allclose(lonlat_to_vec(np.pi / 2, 0), [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(lonlat_to_vec(0, np.pi / 2), [0, 0, 1], atol=1e-12)
    v = np.random.default_rng(0).normal(size=(100, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    np.testing.assert_allclose(lonlat_to_vec(*vec_to_lonlat(v)), v, atol=1e-12)


def test_yaw_rotation_is_clockwise_towards_x():
    r = yaw_pitch_rotation(90, 0)
    np.testing.assert_allclose(r @ [0, 1, 0], [1, 0, 0], atol=1e-12)
    np.testing.assert_allclose(yaw_pitch_rotation(0, 90) @ [0, 1, 0], [0, 0, 1], atol=1e-12)


def test_perspective_sampling_matches_projection():
    # an equirect image whose value encodes lon -> perspective centre pixel must read the yaw back
    w, h = 2048, 1024
    lon, _ = pixel_to_lonlat(np.arange(w), 0, w, h)
    img = np.tile(((lon + np.pi) / (2 * np.pi)).astype(np.float32), (h, 1))
    for yaw in (0, 45, 135):
        face = equirect_to_perspective(img, yaw, 0, 90, 65)
        expected = (np.radians(yaw) + np.pi) / (2 * np.pi)
        assert abs(face[32, 32] - expected) < 2e-3
    rays = perspective_camera_rays(5, 90)
    np.testing.assert_allclose(rays[2, 2], [0, 1, 0])
    # pixel-centre convention: corner pixel of a 5px 90deg view sits at tan = (4 - 2) / 2.5
    np.testing.assert_allclose(rays[0, 4], [0.8, 1, 0.8], atol=1e-12)


def test_ransac_plane_with_outliers():
    rng = np.random.default_rng(1)
    pts = np.c_[rng.uniform(-2, 2, (2000, 2)), rng.normal(0.5, 0.003, 2000)]
    pts = np.r_[pts, rng.uniform(-2, 2, (800, 3))]
    pl = ransac_plane(pts, 0.01, normal_hint=np.array([0, 0, 1.0]))
    assert abs(pl.normal @ [0, 0, 1]) > 0.999
    assert abs(-pl.d - 0.5) < 0.005


def test_align_floor_removes_tilt_and_offset():
    rng = np.random.default_rng(2)
    floor = np.c_[rng.uniform(-2, 2, (3000, 2)), np.zeros(3000)]
    wall = np.c_[rng.uniform(-2, 2, 1500), np.full(1500, 2.0), rng.uniform(0, 2.5, 1500)]
    pts = np.r_[floor, wall]
    tilt = rotation_between(np.array([0, 0, 1.0]), np.array([0.05, -0.03, 1.0]))
    moved = pts @ tilt.T + [0.3, 0.1, 1.4]
    al = align_floor(moved)
    out = al.apply(moved)
    assert np.abs(out[:3000, 2]).max() < 1e-4  # sub-0.1 mm


@pytest.mark.parametrize("yaw_deg", [0, 12.5, 37, -20])
def test_manhattan_yaw_and_regularize(room, yaw_deg):
    yaw = np.radians(yaw_deg)
    rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    rng = np.random.default_rng(3)
    # dense noisy samples along the room outline
    pts = []
    for a, b in zip(room.polygon, np.roll(room.polygon, -1, axis=0)):
        t = rng.uniform(0, 1, 400)[:, None]
        pts.append(a + t * (b - a))
    pts = np.concatenate(pts) @ rot.T + rng.normal(0, 0.01, (2400, 2))
    est = manhattan_yaw(pts)
    err = (np.degrees(est - yaw) + 45) % 90 - 45
    assert abs(err) < 0.5
    noisy = room.polygon @ rot.T + rng.normal(0, 0.03, room.polygon.shape)
    reg = manhattan_regularize(noisy, est)
    assert signed_area(reg) > 0
    poly_close(reg, room.polygon @ rot.T, 0.08)


def test_regularize_drops_tiny_jogs():
    poly = np.array([(0, 0), (4, 0), (4, 1.5), (4.05, 1.5), (4.05, 3), (0, 3)], float)
    reg = manhattan_regularize(poly, 0.0, min_wall=0.15)
    assert len(reg) == 4
    assert polygon_iou(reg, np.array([(0, 0), (4.03, 0), (4.03, 3), (0, 3)])) > 0.98
