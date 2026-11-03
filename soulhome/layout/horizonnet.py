"""Decode HorizonNet / LGT-Net style 1D layout outputs into a metric Manhattan room polygon.

Both network families predict, for every panorama column, the latitude of the wall-ceiling and
wall-floor boundaries plus a corner (wall-wall junction) probability. With a known camera height
that is enough to recover the metric room footprint:

    floor distance   d(col)  = h_cam / tan(-floor_lat)
    ceiling height   H       = h_cam + median(d * tan(ceil_lat))
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry.equirect import pixel_to_lonlat
from ..geometry.polygon import AxisLine, ensure_ccw, manhattan_regularize, simplify, vertices_from_axis_lines
from ..geometry.planes import manhattan_yaw, rot2


@dataclass
class DecodedLayout:
    polygon: np.ndarray
    ceiling_height: float
    yaw: float
    corner_columns: list[int]
    method: str


def from_horizonnet_raw(y_bon: np.ndarray, y_cor: np.ndarray, logits: bool = True):
    """Adapt the reference HorizonNet output tensors.

    ``y_bon`` (2, W): ceiling/floor boundary in radians, *positive downwards* (HorizonNet maps it to
    rows with ``(y / pi + 0.5) * H``). ``y_cor`` (1, W) or (W,): corner logits."""
    y_bon = np.asarray(y_bon, float).reshape(2, -1)
    cor = np.asarray(y_cor, float).reshape(-1)
    if logits:
        cor = 1 / (1 + np.exp(-cor))
    return -y_bon[0], -y_bon[1], cor


def _circular_peaks(signal: np.ndarray, threshold: float, min_dist: int) -> list[int]:
    from scipy.signal import find_peaks

    w = len(signal)
    tiled = np.concatenate([signal, signal, signal])
    peaks, _ = find_peaks(tiled, height=threshold, distance=min_dist)
    return sorted({int(p - w) for p in peaks if w <= p < 2 * w})


def decode_layout(
    ceil_lat: np.ndarray,
    floor_lat: np.ndarray,
    corner_prob: np.ndarray,
    camera_height: float,
    corner_threshold: float = 0.5,
    min_wall: float = 0.15,
) -> DecodedLayout:
    w = len(floor_lat)
    lon, _ = pixel_to_lonlat(np.arange(w), 0, w, 1)
    floor_lat = np.minimum(floor_lat, -1e-3)
    dist = camera_height / np.tan(-floor_lat)
    dist = np.clip(dist, 0.1, 30.0)
    ceiling = camera_height + float(np.median(dist * np.tan(np.clip(ceil_lat, 1e-3, None))))
    pts = np.stack([dist * np.sin(lon), dist * np.cos(lon)], axis=1)

    corners = _circular_peaks(corner_prob, corner_threshold, max(3, w // 128))
    if len(corners) >= 4:
        # fit one wall line per corner-to-corner span (robust: uses every column, not just corners)
        segs = []
        for i, c0 in enumerate(corners):
            c1 = corners[(i + 1) % len(corners)]
            span = (c1 - c0) % w
            margin = max(1, int(0.1 * span))
            cols = (c0 + np.arange(margin, span - margin)) % w
            if len(cols) >= 3:
                segs.append(pts[cols])
        angles, weights = [], []
        for s in segs:
            c = s - s.mean(axis=0)
            _, _, vt = np.linalg.svd(c, full_matrices=False)
            angles.append(np.arctan2(vt[0, 1], vt[0, 0]))
            weights.append(np.linalg.norm(s[-1] - s[0]))
        a4 = 4 * np.asarray(angles)
        wts = np.asarray(weights)
        yaw = float(np.arctan2((wts * np.sin(a4)).sum(), (wts * np.cos(a4)).sum()) / 4)
        lines = []
        for s, length in zip(segs, weights):
            r = s @ rot2(-yaw).T
            d = r[-1] - r[0]
            horizontal = abs(d[0]) >= abs(d[1])
            lines.append(AxisLine(horizontal, float(np.median(r[:, 1] if horizontal else r[:, 0])), float(length)))
        poly = ensure_ccw(vertices_from_axis_lines(lines, min_wall) @ rot2(yaw).T)
        method = "corners+line_fit"
    else:
        yaw = manhattan_yaw(pts)
        poly = manhattan_regularize(simplify(ensure_ccw(pts), 0.06), yaw, min_wall)
        method = "boundary_contour"
    return DecodedLayout(poly, ceiling, yaw, corners, method)
