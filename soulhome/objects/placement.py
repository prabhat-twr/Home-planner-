"""Back-project 2D detections into 3D room coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry.equirect import lonlat_to_vec, pixel_to_lonlat
from .catalog import CatalogEntry, resolve


@dataclass
class Placement:
    entry: CatalogEntry
    label: str
    confidence: float
    position: np.ndarray  # (3,)
    size: tuple[float, float, float] | None
    method: str


def _clamp_inside(xy: np.ndarray, polygon: np.ndarray, inset: float) -> tuple[np.ndarray, bool]:
    from shapely.geometry import Point, Polygon
    from shapely.ops import nearest_points

    poly = Polygon(polygon)
    p = Point(xy)
    if poly.contains(p):
        return xy, False
    inner = poly.buffer(-inset) if poly.buffer(-inset).area > 0 else poly
    q = nearest_points(inner, p)[0]
    return np.array([q.x, q.y]), True


def place_equirect(
    det: dict,
    width: int,
    height: int,
    camera: np.ndarray,
    polygon: np.ndarray,
    points: np.ndarray | None = None,
    point_pixels: np.ndarray | None = None,
) -> Placement | None:
    """Place one equirect detection ``{label, confidence, bbox:[u0,v0,u1,v1]}``.

    * ``depth_median`` when per-pixel 3D points are available: median of the non-floor points
      inside the box (visible front surface), pushed back by half the catalog depth.
    * ``floor_ray`` otherwise: the bbox bottom-centre ray is intersected with the floor plane,
      which gives the object's front contact point; again pushed back by half the catalog depth.
    """
    entry = resolve(det["label"])
    if entry is None:
        return None
    u0, v0, u1, v1 = det["bbox"]
    if u1 < u0:  # box wraps across the panorama seam
        u1 += width
    uc = (u0 + u1) / 2
    lon_c, lat_bottom = pixel_to_lonlat(uc, v1, width, height)
    _, lat_top = pixel_to_lonlat(uc, v0, width, height)
    horiz = np.array([np.sin(lon_c), np.cos(lon_c)])
    cam_h = float(camera[2])
    w_cat, d_cat, h_cat = entry.size
    method = "floor_ray"
    pos = None
    if points is not None and point_pixels is not None:
        uu, vv = point_pixels[:, 0], point_pixels[:, 1]
        uu = np.where(uu < u0, uu + width, uu)
        inside = (uu >= u0) & (uu <= u1) & (vv >= v0) & (vv <= v1) & (points[:, 2] > 0.05)
        sel = points[inside]
        if len(sel) >= 10:
            xy = np.median(sel[:, :2], axis=0)
            top = float(np.quantile(sel[:, 2], 0.95))
            dist = float(np.linalg.norm(xy - camera[:2]))
            pos = np.array([*(xy + horiz * d_cat / 2), 0.0])
            method = "depth_median"
    if pos is None:
        if lat_bottom >= -0.02:
            return None  # bottom edge at/above horizon: no floor contact visible
        dist = cam_h / np.tan(-float(lat_bottom))
        xy = camera[:2] + horiz * (dist + d_cat / 2)
        top = cam_h + (dist + d_cat / 2) * np.tan(float(lat_top))
        pos = np.array([*xy, 0.0])
    if entry.mount in ("wall", "ceiling"):
        pos[2] = max(0.0, top - h_cat / 2)
    ang_w = (u1 - u0) / width * 2 * np.pi
    size = (float(max(ang_w * dist, 0.05)), d_cat, float(max(top, 0.05)))
    xy, _ = _clamp_inside(pos[:2], polygon, min(d_cat / 2, 0.3))
    pos[:2] = xy
    return Placement(entry, det["label"], float(det.get("confidence", 1.0)), pos, size, method)


def place_perspective(det: dict, intrinsics: np.ndarray, cam_to_world: np.ndarray, polygon: np.ndarray) -> Placement | None:
    """AR-frame detection with a known camera pose (world frame already normalised, Z up):
    bottom-centre pixel ray intersected with the floor plane z=0.

    Camera convention: OpenCV (x right, y down, z forward)."""
    entry = resolve(det["label"])
    if entry is None:
        return None
    u0, v0, u1, v1 = det["bbox"]
    k_inv = np.linalg.inv(np.asarray(intrinsics, float))
    ray_cam = k_inv @ np.array([(u0 + u1) / 2, v1, 1.0])
    t_wc = np.asarray(cam_to_world, float)
    origin = t_wc[:3, 3]
    ray = t_wc[:3, :3] @ ray_cam
    if ray[2] >= -1e-6:
        return None
    t = -origin[2] / ray[2]
    hit = origin + t * ray
    horiz = ray[:2] / max(np.linalg.norm(ray[:2]), 1e-9)
    xy = hit[:2] + horiz * entry.size[1] / 2
    xy, _ = _clamp_inside(xy, polygon, min(entry.size[1] / 2, 0.3))
    return Placement(entry, det["label"], float(det.get("confidence", 1.0)), np.array([*xy, 0.0]), entry.size, "floor_ray")


def place_provided(det: dict, polygon: np.ndarray) -> Placement | None:
    """Detections that already carry a 3D position (e.g. iOS RoomPlan objects)."""
    entry = resolve(det["label"])
    if entry is None:
        return None
    pos = np.asarray(det["position"], float)
    size = tuple(det.get("size") or entry.size)
    return Placement(entry, det["label"], float(det.get("confidence", 1.0)), pos, size, "provided")


def nms_equirect(dets: list[dict], width: int, iou_thr: float = 0.4, contain_thr: float = 0.7) -> list[dict]:
    """Cross-tile NMS for equirect boxes (seam aware). Besides IoU, a box mostly *contained* in a
    stronger one is suppressed: a tile edge truncates an object into a partial box that never
    reaches a high IoU with the full one."""
    dets = sorted(dets, key=lambda d: -d.get("confidence", 0))
    kept: list[dict] = []

    def overlap(a, b):
        best_iou = best_contain = 0.0
        for shift in (-width, 0, width):
            ax0, ay0, ax1, ay1 = a
            bx0, by0, bx1, by1 = b[0] + shift, b[1], b[2] + shift, b[3]
            iw = max(0, min(ax1, bx1) - max(ax0, bx0))
            ih = max(0, min(ay1, by1) - max(ay0, by0))
            inter = iw * ih
            area_a, area_b = (ax1 - ax0) * (ay1 - ay0), (bx1 - bx0) * (by1 - by0)
            union = area_a + area_b - inter
            best_iou = max(best_iou, inter / union if union > 0 else 0)
            best_contain = max(best_contain, inter / min(area_a, area_b) if min(area_a, area_b) > 0 else 0)
        return best_iou, best_contain

    def unwrap(b):
        b = list(b)
        if b[2] < b[0]:
            b[2] += width
        return b

    for d in dets:
        d = {**d, "bbox": unwrap(d["bbox"])}
        def dup(k):
            if k["label"] != d["label"]:
                return False
            i, c = overlap(k["bbox"], d["bbox"])
            return i > iou_thr or c > contain_thr

        if not any(dup(k) for k in kept):
            kept.append(d)
    return [{**k, "bbox": [k["bbox"][0], k["bbox"][1], k["bbox"][2] % width, k["bbox"][3]]} for k in kept]


def dedupe_placements(placements: list[Placement], radius: float = 0.6) -> list[Placement]:
    """Same catalog object placed twice within ``radius`` metres -> keep the most confident."""
    kept: list[Placement] = []
    for p in sorted(placements, key=lambda p: -p.confidence):
        if all(k.entry.id != p.entry.id or np.linalg.norm(k.position[:2] - p.position[:2]) > radius for k in kept):
            kept.append(p)
    return kept


def equirect_pixels_of_points(points_rel: np.ndarray, width: int, height: int) -> np.ndarray:
    from ..geometry.equirect import lonlat_to_pixel, vec_to_lonlat

    lon, lat = vec_to_lonlat(points_rel)
    u, v = lonlat_to_pixel(lon, lat, width, height)
    return np.stack([u, v], axis=1)


__all__ = [
    "Placement",
    "place_equirect",
    "place_perspective",
    "place_provided",
    "nms_equirect",
    "dedupe_placements",
    "equirect_pixels_of_points",
    "lonlat_to_vec",
]
