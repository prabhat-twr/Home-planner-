"""2D polygon utilities: simplification and Manhattan regularisation into clean wall vectors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .planes import rot2


def signed_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def ensure_ccw(poly: np.ndarray) -> np.ndarray:
    return poly if signed_area(poly) > 0 else poly[::-1].copy()


def simplify(poly: np.ndarray, tolerance: float) -> np.ndarray:
    """Douglas-Peucker on a closed polygon."""
    import cv2

    approx = cv2.approxPolyDP(poly.astype(np.float32).reshape(-1, 1, 2), tolerance, True)
    return approx.reshape(-1, 2).astype(np.float64)


@dataclass
class AxisLine:
    horizontal: bool  # True -> y = offset (wall runs along x); False -> x = offset
    offset: float
    weight: float  # supporting length


def _merge_runs(lines: list[AxisLine]) -> list[AxisLine]:
    """Merge consecutive (circular) lines that share an orientation (length-weighted offset)."""
    if not lines:
        return lines
    merged: list[AxisLine] = []
    for ln in lines:
        if merged and merged[-1].horizontal == ln.horizontal:
            m = merged[-1]
            w = m.weight + ln.weight
            m.offset = (m.offset * m.weight + ln.offset * ln.weight) / max(w, 1e-12)
            m.weight = w
        else:
            merged.append(AxisLine(ln.horizontal, ln.offset, ln.weight))
    if len(merged) > 1 and merged[0].horizontal == merged[-1].horizontal:
        a, b = merged[0], merged.pop()
        w = a.weight + b.weight
        a.offset = (a.offset * a.weight + b.offset * b.weight) / max(w, 1e-12)
        a.weight = w
    return merged


def vertices_from_axis_lines(lines: list[AxisLine], min_wall: float = 0.15) -> np.ndarray:
    """Closed sequence of alternating axis-aligned lines -> polygon vertices.

    Walls shorter than ``min_wall`` are dropped iteratively (their neighbours then merge)."""
    lines = _merge_runs(list(lines))
    for _ in range(len(lines) + 1):
        if len(lines) < 4:
            raise ValueError(f"degenerate Manhattan polygon ({len(lines)} walls)")
        verts = []
        for i, ln in enumerate(lines):
            nxt = lines[(i + 1) % len(lines)]
            h, v = (ln, nxt) if ln.horizontal else (nxt, ln)
            verts.append((v.offset, h.offset))
        verts = np.array(verts)
        lengths = np.linalg.norm(verts - np.roll(verts, 1, axis=0), axis=1)  # wall i spans v[i-1] -> v[i]
        shortest = int(np.argmin(lengths))
        if lengths[shortest] >= min_wall:
            return verts
        lines.pop(shortest)
        lines = _merge_runs(lines)
    raise ValueError("Manhattan regularisation did not converge")


def manhattan_regularize(poly: np.ndarray, yaw: float, min_wall: float = 0.15) -> np.ndarray:
    """Snap every edge of a (noisy, simplified) polygon to one of two orthogonal directions
    defined by ``yaw`` and return the clean CCW polygon in the original frame."""
    poly = ensure_ccw(np.asarray(poly, dtype=float))
    aligned = poly @ rot2(-yaw).T
    lines = []
    for a, b in zip(aligned, np.roll(aligned, -1, axis=0)):
        d = b - a
        length = float(np.hypot(*d))
        if length < 1e-9:
            continue
        horizontal = abs(d[0]) >= abs(d[1])
        mid = (a + b) / 2
        lines.append(AxisLine(horizontal, float(mid[1] if horizontal else mid[0]), length))
    verts = vertices_from_axis_lines(lines, min_wall=min_wall)
    return ensure_ccw(verts @ rot2(yaw).T)


def point_in_polygon(pt, poly: np.ndarray) -> bool:
    from shapely.geometry import Point, Polygon

    return Polygon(poly).covers(Point(pt))


def polygon_iou(a: np.ndarray, b: np.ndarray) -> float:
    from shapely.geometry import Polygon

    pa, pb = Polygon(a), Polygon(b)
    return pa.intersection(pb).area / pa.union(pb).area
