"""Fuse monocular depth with a layout-network room polygon.

Monocular depth is good at *relative* structure (what is in front of what) but its absolute scale
drifts across the panorama. The layout network gives metric wall positions (from the floor/ceiling
boundaries and camera height). So we rescale every depth ray by a smooth per-azimuth factor that
makes the depth of the upper wall band agree with the layout walls. The result: depth points that
live in the same metric frame as the layout, usable for opening detection and object placement.
"""

from __future__ import annotations

import numpy as np


def ray_polygon_distance(az: np.ndarray, polygon: np.ndarray, origin=(0.0, 0.0)) -> np.ndarray:
    """Horizontal distance from ``origin`` along azimuth ``az`` (atan2(x, y)) to the polygon."""
    d = np.stack([np.sin(az), np.cos(az)], axis=1)
    o = np.asarray(origin, float)
    best = np.full(len(az), np.inf)
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        e = b - a
        den = d[:, 0] * (-e[1]) - d[:, 1] * (-e[0])
        rx, ry = a[0] - o[0], a[1] - o[1]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (rx * (-e[1]) - ry * (-e[0])) / den
            s = (d[:, 0] * ry - d[:, 1] * rx) / den
        ok = (t > 0) & (s >= -1e-9) & (s <= 1 + 1e-9)
        best = np.where(ok & (t < best), t, best)
    return best


def scale_depth_to_layout(
    points_cam: np.ndarray,
    polygon: np.ndarray,
    camera_height: float,
    ceiling_height: float,
    bins: int = 256,
    min_lat_deg: float = 3.0,
    smooth: int = 9,
) -> tuple[np.ndarray, dict]:
    """``points_cam``: depth points relative to the (upright) camera centre.
    Returns points in the room frame (floor at z = 0, camera at (0, 0, camera_height)).

    Per azimuth, the reference band is the upper wall: from just above the horizon (above most
    furniture) up to 80% of the angle at which the layout says the wall meets the ceiling."""
    from scipy.ndimage import median_filter

    horiz = np.hypot(points_cam[:, 0], points_cam[:, 1])
    az = np.arctan2(points_cam[:, 0], points_cam[:, 1])
    lat = np.degrees(np.arctan2(points_cam[:, 2], horiz))
    idx = ((az + np.pi) / (2 * np.pi) * bins).astype(int) % bins
    centres = (np.arange(bins) + 0.5) / bins * 2 * np.pi - np.pi
    r_layout = ray_polygon_distance(centres, polygon)
    top_lat = np.degrees(np.arctan2(ceiling_height - camera_height, r_layout))
    band = (lat > min_lat_deg) & (lat < 0.8 * top_lat[idx])
    ratio = np.full(bins, np.nan)
    order = np.argsort(idx)
    splits = np.searchsorted(idx[order], np.arange(bins + 1))
    for b in range(bins):
        sel = order[splits[b] : splits[b + 1]]
        sel = sel[band[sel]]
        if len(sel) >= 5 and np.isfinite(r_layout[b]):
            # the wall surface is the *nearest* thing in the upper band; anything farther was seen
            # through a door/window, so a low quantile ignores openings
            r_depth = np.quantile(horiz[sel], 0.2)
            if r_depth > 0:
                ratio[b] = r_layout[b] / r_depth
    ok = np.isfinite(ratio)
    if ok.sum() < bins * 0.3:
        raise ValueError(f"depth/layout overlap too small ({ok.mean():.0%} of azimuths)")
    # reject bins that disagree wildly with the panorama-wide scale (mirrors, glass, mislabelled band)
    med = np.median(ratio[ok])
    ok &= np.abs(np.log(np.where(ok, ratio, 1.0) / med)) < np.log(1.5)
    xs = np.arange(bins)
    ratio = np.interp(xs, xs[ok], ratio[ok], period=bins)
    ratio = median_filter(ratio, size=smooth, mode="wrap")
    out = points_cam * ratio[idx][:, None]
    out[:, 2] += camera_height
    stats = {
        "scale_median": round(float(np.median(ratio)), 3),
        "scale_spread": round(float(np.quantile(ratio, 0.9) / np.quantile(ratio, 0.1)), 3),
        "coverage": round(float(ok.mean()), 3),
    }
    return out, stats
