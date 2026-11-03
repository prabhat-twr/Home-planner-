"""Equirectangular <-> spherical <-> perspective projection math.

Conventions (shared by every module):
    lon in [-pi, pi): 0 at the image centre column (+Y), increasing to the right (+X)
    lat in [-pi/2, pi/2]: +pi/2 at the top row (+Z)
    unit ray: x = cos(lat) sin(lon), y = cos(lat) cos(lon), z = sin(lat)
"""

from __future__ import annotations

import numpy as np


def pixel_to_lonlat(u, v, width: int, height: int):
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    lon = (u + 0.5) / width * 2 * np.pi - np.pi
    lat = np.pi / 2 - (v + 0.5) / height * np.pi
    return lon, lat


def lonlat_to_pixel(lon, lat, width: int, height: int):
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    lon = (lon + np.pi) % (2 * np.pi) - np.pi
    u = (lon + np.pi) / (2 * np.pi) * width - 0.5
    v = (np.pi / 2 - lat) / np.pi * height - 0.5
    return u, v


def lonlat_to_vec(lon, lat):
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    c = np.cos(lat)
    return np.stack([c * np.sin(lon), c * np.cos(lon), np.sin(lat)], axis=-1)


def vec_to_lonlat(vec):
    vec = np.asarray(vec, dtype=np.float64)
    n = np.linalg.norm(vec, axis=-1, keepdims=True)
    vec = vec / np.where(n == 0, 1, n)
    lon = np.arctan2(vec[..., 0], vec[..., 1])
    lat = np.arcsin(np.clip(vec[..., 2], -1, 1))
    return lon, lat


def equirect_rays(width: int, height: int) -> np.ndarray:
    """(H, W, 3) unit rays for every pixel centre."""
    uu, vv = np.meshgrid(np.arange(width), np.arange(height))
    lon, lat = pixel_to_lonlat(uu, vv, width, height)
    return lonlat_to_vec(lon, lat)


def yaw_pitch_rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Camera->world rotation for a virtual pinhole camera looking along +Y,
    pitched up by ``pitch_deg`` and then yawed clockwise (towards +X) by ``yaw_deg``."""
    p, y = np.radians(pitch_deg), np.radians(yaw_deg)
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    rz = np.array([[np.cos(y), np.sin(y), 0], [-np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return rz @ rx


def perspective_intrinsics(size: int, fov_deg: float) -> np.ndarray:
    f = 0.5 * size / np.tan(np.radians(fov_deg) / 2)
    c = (size - 1) / 2
    return np.array([[f, 0, c], [0, f, c], [0, 0, 1.0]])


def perspective_camera_rays(size: int, fov_deg: float) -> np.ndarray:
    """(S, S, 3) *unnormalised* camera-frame rays with forward component == 1.

    Camera frame: +X right, +Y forward, +Z up (image rows grow downward = -Z)."""
    k = perspective_intrinsics(size, fov_deg)
    jj, ii = np.meshgrid(np.arange(size), np.arange(size))
    x = (jj - k[0, 2]) / k[0, 0]
    z = -(ii - k[1, 2]) / k[1, 1]
    return np.stack([x, np.ones_like(x), z], axis=-1)


def equirect_to_perspective(
    image: np.ndarray, yaw_deg: float, pitch_deg: float, fov_deg: float = 90.0, size: int = 512
) -> np.ndarray:
    """Sample a pinhole view out of an equirectangular image (bilinear, wraps in lon)."""
    import cv2

    h, w = image.shape[:2]
    rays = perspective_camera_rays(size, fov_deg) @ yaw_pitch_rotation(yaw_deg, pitch_deg).T
    lon, lat = vec_to_lonlat(rays)
    u, v = lonlat_to_pixel(lon, lat, w, h)
    # pad one column on each side so bilinear sampling wraps around the seam
    padded = np.concatenate([image[:, -1:], image, image[:, :1]], axis=1)
    return cv2.remap(
        padded,
        (u + 1).astype(np.float32),
        v.astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def perspective_depth_to_points(depth: np.ndarray, yaw_deg: float, pitch_deg: float, fov_deg: float) -> np.ndarray:
    """Planar (z-buffer) depth of a virtual pinhole view -> world-frame points (N, 3)
    relative to the camera centre."""
    rays = perspective_camera_rays(depth.shape[0], fov_deg)
    pts_cam = rays * depth[..., None]
    pts = pts_cam.reshape(-1, 3) @ yaw_pitch_rotation(yaw_deg, pitch_deg).T
    return pts[np.isfinite(depth.reshape(-1)) & (depth.reshape(-1) > 0)]


def equirect_depth_to_points(depth: np.ndarray, stride: int = 1) -> np.ndarray:
    """Ray-distance equirect depth map (H, W) -> (N, 3) points around the camera centre."""
    d = depth[::stride, ::stride]
    h, w = depth.shape
    vv, uu = np.mgrid[0:h:stride, 0:w:stride]
    lon, lat = pixel_to_lonlat(uu, vv, w, h)
    pts = lonlat_to_vec(lon, lat) * d[..., None]
    pts = pts.reshape(-1, 3)
    ok = np.isfinite(d.reshape(-1)) & (d.reshape(-1) > 0)
    return pts[ok]
