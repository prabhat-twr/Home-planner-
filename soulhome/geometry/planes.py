"""Plane fitting, gravity alignment and Manhattan-yaw estimation on point clouds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Plane:
    normal: np.ndarray  # unit (3,)
    d: float  # n . p + d = 0
    inliers: np.ndarray  # bool mask over input points

    def distance(self, pts: np.ndarray) -> np.ndarray:
        return pts @ self.normal + self.d


def fit_plane_lstsq(pts: np.ndarray) -> tuple[np.ndarray, float]:
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = vt[-1]
    return n, -float(n @ c)


def ransac_plane(
    pts: np.ndarray,
    threshold: float = 0.02,
    iterations: int = 300,
    normal_hint: np.ndarray | None = None,
    max_angle_deg: float = 20.0,
    seed: int = 0,
) -> Plane | None:
    """RANSAC plane fit, optionally restricted to normals within ``max_angle_deg`` of a hint."""
    n_pts = len(pts)
    if n_pts < 3:
        return None
    rng = np.random.default_rng(seed)
    cos_lim = np.cos(np.radians(max_angle_deg))
    best_count, best = 0, None
    idx = rng.integers(0, n_pts, size=(iterations, 3))
    for a, b, c in idx:
        n = np.cross(pts[b] - pts[a], pts[c] - pts[a])
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        if normal_hint is not None:
            if abs(n @ normal_hint) < cos_lim:
                continue
        d = -n @ pts[a]
        count = int(np.count_nonzero(np.abs(pts @ n + d) < threshold))
        if count > best_count:
            best_count, best = count, (n, d)
    if best is None:
        return None
    n, d = best
    mask = np.abs(pts @ n + d) < threshold
    n, d = fit_plane_lstsq(pts[mask])  # refine on inliers
    if normal_hint is not None and n @ normal_hint < 0:
        n, d = -n, -d
    mask = np.abs(pts @ n + d) < threshold
    return Plane(n, float(d), mask)


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation mapping unit vector a onto unit vector b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < 1e-12:
        if c > 0:
            return np.eye(3)
        # 180 deg: rotate about any axis orthogonal to a
        axis = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        axis = np.cross(a, axis)
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


@dataclass
class GravityAlignment:
    rotation: np.ndarray  # applied first
    z_offset: float  # then subtract so the floor sits at z = 0
    floor: Plane

    def apply(self, pts: np.ndarray) -> np.ndarray:
        out = pts @ self.rotation.T
        out[:, 2] -= self.z_offset
        return out


def align_floor(pts: np.ndarray, up_hint=(0, 0, 1), threshold: float = 0.02, max_tilt_deg: float = 25.0) -> GravityAlignment:
    """Find the floor (lowest large plane roughly orthogonal to ``up_hint``), rotate it to +Z and
    translate it to z=0."""
    up = np.asarray(up_hint, dtype=float)
    up /= np.linalg.norm(up)
    h = pts @ up
    # candidate floor points: the lowest 30% along the up axis
    low = pts[h < np.quantile(h, 0.3)]
    plane = ransac_plane(low, threshold=threshold, normal_hint=up, max_angle_deg=max_tilt_deg)
    if plane is None:
        raise ValueError("no floor plane found")
    rot = rotation_between(plane.normal, np.array([0, 0, 1.0]))
    # point on plane: -d * n
    floor_pt = -plane.d * plane.normal
    z0 = float((rot @ floor_pt)[2])
    full = Plane(plane.normal, plane.d, np.abs(pts @ plane.normal + plane.d) < threshold)
    return GravityAlignment(rot, z0, full)


def manhattan_yaw(xy: np.ndarray, weights: np.ndarray | None = None, cell: float = 0.05) -> float:
    """Dominant wall orientation (radians in [0, pi/2)) of 2D wall points.

    Rotating the points by ``-yaw`` makes walls axis aligned. Uses the classic histogram-entropy
    criterion: axis-aligned walls produce the "peakiest" x/y histograms."""
    if len(xy) < 10:
        return 0.0
    xy = xy - xy.mean(axis=0)

    def score(theta):
        c, s = np.cos(-theta), np.sin(-theta)
        r = xy @ np.array([[c, s], [-s, c]])
        total = 0.0
        for k in range(2):
            bins = np.floor(r[:, k] / cell).astype(int)
            bins -= bins.min()
            hist = np.bincount(bins, weights=weights).astype(float)
            p = hist[hist > 0] / hist.sum()
            total += -(p * np.log(p)).sum()
        return total

    coarse = np.radians(np.arange(0, 90, 1.0))
    best = coarse[int(np.argmin([score(t) for t in coarse]))]
    fine = best + np.radians(np.arange(-1, 1.01, 0.1))
    best = fine[int(np.argmin([score(t) for t in fine]))]
    return float(best % (np.pi / 2))


def rot2(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])
