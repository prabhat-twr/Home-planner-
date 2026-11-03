"""Point cloud (gravity aligned, floor at z=0) -> room footprint, ceiling height, walls, openings."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .planes import manhattan_yaw, ransac_plane
from .planes import rot2
from .polygon import AxisLine, ensure_ccw, manhattan_regularize, simplify, vertices_from_axis_lines


@dataclass
class OpeningGeom:
    type: str
    wall_index: int
    offset_m: float
    width_m: float
    sill_height_m: float
    height_m: float


@dataclass
class RoomGeometry:
    polygon: np.ndarray  # (N, 2) CCW
    ceiling_height: float
    yaw: float
    openings: list[OpeningGeom] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


def estimate_ceiling_height(pts: np.ndarray) -> float:
    z = pts[:, 2]
    high = pts[z > np.quantile(z, 0.8)]
    plane = ransac_plane(high, threshold=0.02, normal_hint=np.array([0, 0, 1.0]), max_angle_deg=10)
    if plane is not None and plane.inliers.sum() > 0.05 * len(high):
        return float(np.median(high[plane.inliers, 2]))
    return float(np.quantile(z, 0.99))


def wall_band(pts: np.ndarray, ceiling: float, lo: float = 0.75, hi: float = 0.95) -> np.ndarray:
    z = pts[:, 2]
    return pts[(z > lo * ceiling) & (z < hi * ceiling)]


def footprint_from_viewpoint(
    pts: np.ndarray, ceiling: float, viewpoint=(0.0, 0.0), bins: int = 720, simplify_tol: float = 0.06
) -> tuple[np.ndarray, float]:
    """Single-viewpoint capture (panorama): angular sweep of upper-wall points around the camera.

    The upper wall band sits above doors and furniture, so per azimuth a low percentile of the
    horizontal distance is a robust estimate of the wall position."""
    band = wall_band(pts, ceiling)
    rel = band[:, :2] - np.asarray(viewpoint)
    az = np.arctan2(rel[:, 0], rel[:, 1])
    dist = np.hypot(rel[:, 0], rel[:, 1])
    idx = ((az + np.pi) / (2 * np.pi) * bins).astype(int) % bins
    order = np.argsort(idx)
    idx, dist = idx[order], dist[order]
    splits = np.searchsorted(idx, np.arange(bins + 1))
    r = np.full(bins, np.nan)
    for b in range(bins):
        seg = dist[splits[b] : splits[b + 1]]
        if len(seg) >= 3:
            r[b] = np.quantile(seg, 0.25)
    ok = np.isfinite(r)
    if ok.sum() < bins * 0.5:
        raise ValueError(f"wall coverage too low ({ok.mean():.0%} of azimuths)")
    # fill gaps by circular interpolation
    xs = np.arange(bins)
    r = np.interp(xs, xs[ok], r[ok], period=bins)
    centres = (xs + 0.5) / bins * 2 * np.pi - np.pi
    contour = np.stack([np.sin(centres) * r, np.cos(centres) * r], axis=1) + np.asarray(viewpoint)
    yaw = manhattan_yaw(band[:, :2])
    poly = simplify(ensure_ccw(contour), simplify_tol)
    poly = manhattan_regularize(poly, yaw)
    return refine_walls(poly, yaw, band[:, :2])


def footprint_from_occupancy(
    pts: np.ndarray, ceiling: float, cell: float = 0.05, simplify_tol: float = 0.08
) -> tuple[np.ndarray, float]:
    """Multi-view capture (LiDAR / ARCore mesh): rasterise the floor, close small holes,
    take the largest external contour."""
    import cv2

    floor = pts[np.abs(pts[:, 2]) < 0.04][:, :2]
    if len(floor) < 50:
        raise ValueError("too few floor points for occupancy footprint")
    lo = floor.min(axis=0) - 5 * cell
    ij = np.floor((floor - lo) / cell).astype(int)
    shape = ij.max(axis=0) + 6
    grid = np.zeros((shape[1], shape[0]), np.uint8)
    grid[ij[:, 1], ij[:, 0]] = 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, k)
    grid = cv2.morphologyEx(grid, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("no floor contour found")
    c = max(contours, key=cv2.contourArea).reshape(-1, 2).astype(float)
    # pixel centre -> metres; contour traces pixel centres of the boundary cells, which sit
    # half a cell inside the true edge
    poly = (c + 0.5) * cell + lo
    band = wall_band(pts, ceiling, 0.6, 0.95)
    yaw = manhattan_yaw(band[:, :2]) if len(band) > 100 else manhattan_yaw(poly)
    poly = simplify(ensure_ccw(poly), simplify_tol)
    poly = manhattan_regularize(poly, yaw)
    # the raster contour is only good to ~1 cell; snap walls onto the measured wall surfaces
    return refine_walls(poly, yaw, band[:, :2])


def refine_walls(
    poly: np.ndarray, yaw: float, wall_xy: np.ndarray, search: float = 0.15, iterations: int = 2
) -> tuple[np.ndarray, float]:
    """Re-estimate the Manhattan yaw and each wall's offset from the wall points that support it.

    The initial footprint (angular sweep / raster contour) is quantised; this step fits each wall
    line to its supporting points (total least squares for orientation, median for offset)."""
    for _ in range(iterations):
        aligned_pts = wall_xy @ rot2(-yaw).T
        aligned_poly = poly @ rot2(-yaw).T
        lines, angles, weights = [], [], []
        for a, b in zip(aligned_poly, np.roll(aligned_poly, -1, axis=0)):
            d = b - a
            horizontal = abs(d[0]) >= abs(d[1])
            ax, ox = (0, 1) if horizontal else (1, 0)  # along-axis, offset-axis
            offset = (a[ox] + b[ox]) / 2
            lo, hi = sorted((a[ax], b[ax]))
            margin = 0.1 * (hi - lo)
            sel = (
                (np.abs(aligned_pts[:, ox] - offset) < search)
                & (aligned_pts[:, ax] > lo + margin)
                & (aligned_pts[:, ax] < hi - margin)
            )
            sup = aligned_pts[sel]
            length = hi - lo
            if len(sup) >= 20:
                offset = float(np.median(sup[:, ox]))
                c = sup - sup.mean(axis=0)
                _, _, vt = np.linalg.svd(c, full_matrices=False)
                angles.append(np.arctan2(vt[0, 1], vt[0, 0]))
                weights.append(len(sup))
            lines.append(AxisLine(horizontal, offset, length))
        verts = vertices_from_axis_lines(lines)
        poly = ensure_ccw(verts @ rot2(yaw).T)
        if weights:
            w = np.asarray(weights, float)
            a4 = 4 * np.asarray(angles)
            delta = np.arctan2((w * np.sin(a4)).sum(), (w * np.cos(a4)).sum()) / 4
            if abs(delta) < 1e-5:
                break
            yaw = yaw + delta
            poly = poly  # re-snapped in the next iteration with the corrected yaw
    yaw = (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4  # canonical range [-45, 45) deg
    return poly, float(yaw)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out, start = [], None
    for i, v in enumerate(np.append(mask, False)):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    return out


def detect_openings(
    pts: np.ndarray,
    polygon: np.ndarray,
    ceiling: float,
    cell: float = 0.05,
    wall_tol: float = 0.07,
    behind_min: float = 0.2,
) -> list[OpeningGeom]:
    """Find doors/windows as see-through gaps in each wall.

    A wall cell is "open" when it has no on-wall points *and* there is evidence behind the wall
    plane (geometry seen through it). Requiring see-through evidence stops furniture occlusion in
    front of a wall from being mistaken for a doorway."""
    out: list[OpeningGeom] = []
    nz = int(np.ceil(ceiling / cell))
    for wi, (a, b) in enumerate(zip(polygon, np.roll(polygon, -1, axis=0))):
        t = b - a
        length = float(np.linalg.norm(t))
        if length < 0.5:
            continue
        t = t / length
        n_in = np.array([-t[1], t[0]])  # CCW polygon -> interior on the left
        rel = pts[:, :2] - a
        s = rel @ t
        dn = rel @ n_in
        in_span = (s > 0) & (s < length) & (pts[:, 2] > 0.02) & (pts[:, 2] < ceiling - 0.02)
        on_wall = in_span & (np.abs(dn) < wall_tol)
        behind = in_span & (dn < -behind_min) & (dn > -6.0)
        ns = int(np.ceil(length / cell))

        def grid(mask):
            g = np.zeros((ns, nz), bool)
            si = np.clip((s[mask] / cell).astype(int), 0, ns - 1)
            zi = np.clip((pts[mask, 2] / cell).astype(int), 0, nz - 1)
            g[si, zi] = True
            return g

        wall_g, behind_g = grid(on_wall), grid(behind)
        open_g = ~wall_g & behind_g
        zc = (np.arange(nz) + 0.5) * cell

        def frac(g, z0, z1):
            sel = (zc > z0) & (zc < z1)
            return g[:, sel].mean(axis=1) if sel.any() else np.zeros(ns)

        door_cols = frac(open_g, 0.3, 1.8) > 0.6
        # window: see-through at mid height but *not* see-through below the sill. (The sill wall
        # itself may be hidden behind furniture, so we don't require it to be observed.)
        window_cols = (frac(open_g, 1.1, 1.6) > 0.6) & (frac(open_g, 0.1, 0.6) < 0.2) & ~door_cols
        for kind, cols, min_w, max_w in (("door", door_cols, 0.6, 2.5), ("window", window_cols, 0.4, 4.0)):
            # close 1-cell gaps caused by sampling noise
            cols = cols.copy()
            for i in range(1, ns - 1):
                if not cols[i] and cols[i - 1] and cols[i + 1]:
                    cols[i] = True
            for s0, s1 in _runs(cols):
                width = (s1 - s0) * cell
                if not (min_w <= width <= max_w):
                    continue
                col_open = open_g[s0:s1].mean(axis=0) > 0.5
                zs = np.nonzero(col_open)[0]
                if len(zs) == 0:
                    continue
                # contiguous open run that contains the reference height
                ref = int((0.9 if kind == "door" else 1.35) / cell)
                runs = [r for r in _runs(col_open) if r[0] <= ref < r[1]] or _runs(col_open)
                z0, z1 = max(runs, key=lambda r: r[1] - r[0])
                sill = 0.0 if kind == "door" else z0 * cell
                top = z1 * cell
                out.append(OpeningGeom(kind, wi, s0 * cell, width, round(sill, 3), round(top - sill, 3)))
    return out


def extract_room(
    pts: np.ndarray, mode: str, viewpoint=(0.0, 0.0), find_openings: bool = True
) -> RoomGeometry:
    ceiling = estimate_ceiling_height(pts)
    if mode == "viewpoint":
        poly, yaw = footprint_from_viewpoint(pts, ceiling, viewpoint)
    elif mode == "occupancy":
        poly, yaw = footprint_from_occupancy(pts, ceiling)
    else:
        raise ValueError(mode)
    openings = detect_openings(pts, poly, ceiling) if find_openings else []
    return RoomGeometry(poly, ceiling, yaw, openings, {"n_points": int(len(pts))})
