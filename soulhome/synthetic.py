"""Analytic synthetic rooms: ground truth for regression tests and the live demo endpoints.

The same room description can be rendered as
  * an equirectangular ray-distance depth map (panorama tier),
  * HorizonNet-style boundary / corner outputs (layout-net decoding),
  * a triangle mesh with real door/window holes (LiDAR / ARCore tiers),
  * 2D detections on the panorama (object back-projection).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry.equirect import equirect_rays, lonlat_to_pixel, pixel_to_lonlat, vec_to_lonlat
from .geometry.polygon import ensure_ccw

BEHIND_GAP = 1.5  # metres of free space visible through openings


@dataclass
class SynthOpening:
    type: str
    wall: int
    offset: float
    width: float
    sill: float
    height: float


@dataclass
class SynthObject:
    label: str
    center: tuple[float, float]  # floor xy
    size: tuple[float, float, float]  # (sx, sy, height)

    def bounds(self):
        cx, cy = self.center
        sx, sy, h = self.size
        return np.array([cx - sx / 2, cy - sy / 2, 0.0]), np.array([cx + sx / 2, cy + sy / 2, h])


@dataclass
class SynthRoom:
    polygon: np.ndarray
    ceiling: float = 2.7
    camera: tuple[float, float, float] = (0.0, 0.0, 1.5)
    openings: list[SynthOpening] = field(default_factory=list)
    objects: list[SynthObject] = field(default_factory=list)

    @property
    def walls(self):
        p = self.polygon
        return list(zip(p, np.roll(p, -1, axis=0)))


def l_shaped_room() -> SynthRoom:
    poly = ensure_ccw(np.array([(-2.0, -1.5), (3.0, -1.5), (3.0, 2.0), (0.5, 2.0), (0.5, 3.5), (-2.0, 3.5)]))
    room = SynthRoom(poly)
    # identify walls by geometry so the definition is independent of vertex order
    def wall_idx(a, b):
        for i, (p, q) in enumerate(room.walls):
            if np.allclose(p, a) and np.allclose(q, b):
                return i
        raise KeyError((a, b))

    room.openings = [
        SynthOpening("door", wall_idx((-2.0, -1.5), (3.0, -1.5)), 1.2, 0.9, 0.0, 2.05),
        SynthOpening("window", wall_idx((3.0, -1.5), (3.0, 2.0)), 1.0, 1.2, 0.9, 1.2),
    ]
    room.objects = [
        SynthObject("sofa", (-1.5, 1.0), (0.9, 2.0, 0.85)),
        SynthObject("dining table", (1.5, 0.3), (1.2, 0.8, 0.75)),
        SynthObject("potted plant", (-1.6, 3.1), (0.4, 0.4, 1.1)),
    ]
    return room


# ---------------------------------------------------------------------------------- panorama


def _ray_aabb(origin, dirs, lo, hi):
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / dirs
        t1 = (lo - origin) * inv
        t2 = (hi - origin) * inv
    tmin = np.nanmax(np.minimum(t1, t2), axis=-1)
    tmax = np.nanmin(np.maximum(t1, t2), axis=-1)
    hit = (tmax >= np.maximum(tmin, 0)) & (tmin > 0)
    return np.where(hit, tmin, np.inf)


def render_depth_panorama(room: SynthRoom, width: int = 1024, height: int = 512) -> np.ndarray:
    """Ray-distance depth (H, W) from the camera, with see-through openings and furniture."""
    cam = np.asarray(room.camera, float)
    d = equirect_rays(width, height)
    dx, dy, dz = d[..., 0], d[..., 1], d[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        t_floor = np.where(dz < 0, -cam[2] / dz, np.inf)
        t_ceil = np.where(dz > 0, (room.ceiling - cam[2]) / dz, np.inf)
    t = np.minimum(t_floor, t_ceil)
    for wi, (a, b) in enumerate(room.walls):
        e = b - a
        length = np.linalg.norm(e)
        # solve cam + t*d = a + s*e  (2D)
        den = dx * (-e[1]) - dy * (-e[0])
        rx, ry = a[0] - cam[0], a[1] - cam[1]
        with np.errstate(divide="ignore", invalid="ignore"):
            tw = (rx * (-e[1]) - ry * (-e[0])) / den
            s = (dx * ry - dy * rx) / den
        z = cam[2] + tw * dz
        hit = (tw > 1e-6) & (s >= 0) & (s <= 1) & (z >= 0) & (z <= room.ceiling)
        through = np.zeros_like(hit)
        for op in room.openings:
            if op.wall != wi:
                continue
            sl = s * length
            through |= (sl >= op.offset) & (sl <= op.offset + op.width) & (z >= op.sill) & (z <= op.sill + op.height)
        tw_eff = np.where(through, tw + BEHIND_GAP / np.maximum(np.hypot(dx, dy), 1e-6), tw)
        t = np.where(hit & (tw_eff < t), tw_eff, t)
    for obj in room.objects:
        lo, hi = obj.bounds()
        t = np.minimum(t, _ray_aabb(cam, d, lo, hi))
    return t.astype(np.float32)


def horizonnet_outputs(room: SynthRoom, width: int = 1024):
    """Exact per-column ceiling/floor boundary latitudes and a corner heat-map, i.e. what a
    perfect HorizonNet would predict for this room (openings/furniture ignored)."""
    cam = np.asarray(room.camera, float)
    lon, _ = pixel_to_lonlat(np.arange(width), 0, width, 1)
    dirs = np.stack([np.sin(lon), np.cos(lon)], axis=1)
    dist = np.full(width, np.inf)
    for a, b in room.walls:
        e = b - a
        den = dirs[:, 0] * (-e[1]) - dirs[:, 1] * (-e[0])
        rx, ry = a[0] - cam[0], a[1] - cam[1]
        with np.errstate(divide="ignore", invalid="ignore"):
            tw = (rx * (-e[1]) - ry * (-e[0])) / den
            s = (dirs[:, 0] * ry - dirs[:, 1] * rx) / den
        ok = (tw > 0) & (s >= 0) & (s <= 1)
        dist = np.where(ok & (tw < dist), tw, dist)
    floor_lat = -np.arctan2(cam[2], dist)
    ceil_lat = np.arctan2(room.ceiling - cam[2], dist)
    corner = np.zeros(width)
    for v in room.polygon:
        rel = v - cam[:2]
        u, _ = lonlat_to_pixel(np.arctan2(rel[0], rel[1]), 0, width, 1)
        cols = np.arange(width)
        dd = np.minimum(np.abs(cols - u), width - np.abs(cols - u))
        corner = np.maximum(corner, np.exp(-0.5 * (dd / 3.0) ** 2))
    return ceil_lat, floor_lat, corner


def panorama_detections(room: SynthRoom, width: int = 1024, height: int = 512) -> list[dict]:
    """Tight equirect bounding boxes (pixels) of the synthetic objects."""
    cam = np.asarray(room.camera, float)
    dets = []
    for obj in room.objects:
        lo, hi = obj.bounds()
        corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        lon, lat = vec_to_lonlat(corners - cam)
        u, v = lonlat_to_pixel(lon, lat, width, height)
        dets.append(
            {"label": obj.label, "confidence": 0.9, "bbox": [float(u.min()), float(v.min()), float(u.max()), float(v.max())]}
        )
    return dets


def shaded_panorama(room: SynthRoom, width: int = 1024, height: int = 512) -> np.ndarray:
    """A cheap greyscale render (depth-shaded) so demo scans have a viewable image."""
    depth = render_depth_panorama(room, width, height)
    img = np.clip(255 * (1.2 - np.log1p(depth) / np.log1p(8)), 0, 255).astype(np.uint8)
    return np.repeat(img[..., None], 3, axis=2)


# -------------------------------------------------------------------------------------- mesh


def _quad(p0, p1, p2, p3):
    return [p0, p1, p2, p3], [[0, 1, 2], [0, 2, 3]]


def room_mesh(room: SynthRoom, with_objects: bool = True):
    """Triangle mesh of the room interior with real holes for openings, plus a back-plane behind
    each opening (what a scanner sees through a door / window)."""
    import trimesh

    verts, faces = [], []

    def add(vs, fs):
        base = len(verts)
        verts.extend(vs)
        faces.extend([[base + i for i in f] for f in fs])

    h = room.ceiling
    c = np.asarray(room.camera[:2])
    poly = room.polygon
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):  # fan from an interior (star) point
        add([(*c, 0), (*a, 0), (*b, 0)], [[0, 1, 2]])
        add([(*c, h), (*b, h), (*a, h)], [[0, 1, 2]])
    for wi, (a, b) in enumerate(room.walls):
        e = b - a
        length = np.linalg.norm(e)
        t = e / length
        n_out = np.array([t[1], -t[0]])
        ops = sorted([o for o in room.openings if o.wall == wi], key=lambda o: o.offset)
        rects, cursor = [], 0.0
        for o in ops:
            rects.append((cursor, o.offset, 0, h))
            rects.append((o.offset, o.offset + o.width, 0, o.sill))
            rects.append((o.offset, o.offset + o.width, o.sill + o.height, h))
            cursor = o.offset + o.width
            # back plane seen through the opening
            m = 0.3
            s0, s1 = o.offset - m, o.offset + o.width + m
            pts = [a + t * s + n_out * BEHIND_GAP for s in (s0, s1)]
            add(*_quad((*pts[0], 0), (*pts[1], 0), (*pts[1], h), (*pts[0], h)))
        rects.append((cursor, length, 0, h))
        for s0, s1, z0, z1 in rects:
            if s1 - s0 < 1e-6 or z1 - z0 < 1e-6:
                continue
            p0, p1 = a + t * s0, a + t * s1
            add(*_quad((*p0, z0), (*p1, z0), (*p1, z1), (*p0, z1)))
    mesh = trimesh.Trimesh(np.array(verts, float), np.array(faces), process=True)
    if with_objects:
        parts = [mesh]
        for obj in room.objects:
            lo, hi = obj.bounds()
            box = trimesh.creation.box(extents=hi - lo)
            box.apply_translation((lo + hi) / 2)
            parts.append(box)
        mesh = trimesh.util.concatenate(parts)
    return mesh
