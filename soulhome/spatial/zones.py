"""Direction / zone queries on room geometry (the vocabulary of the Vastu & Feng Shui rule engine).

* ``direction_16`` - 16 Vastu directions (22.5 deg sectors) measured from the room centroid.
* ``grid_zone``    - 3x3 Vastu Purusha grid over the north-aligned bounding box; the centre cell is
  the Brahmasthan.
"""

from __future__ import annotations

import math

import numpy as np

DIRS16 = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
DIRS8 = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
GRID = [["NW", "N", "NE"], ["W", "BRAHMASTHAN", "E"], ["SW", "S", "SE"]]


def bearing(dx: float, dy: float, north_deg: float) -> float:
    """Compass bearing (deg clockwise from north) of a horizontal vector in the model frame."""
    return (north_deg + math.degrees(math.atan2(dx, dy))) % 360


def direction_16(b: float) -> str:
    return DIRS16[int(((b + 11.25) % 360) // 22.5)]


def direction_8(b: float) -> str:
    return DIRS8[int(((b + 22.5) % 360) // 45)]


def to_east_north(xy: np.ndarray, north_deg: float) -> np.ndarray:
    n = math.radians(north_deg)
    xy = np.atleast_2d(np.asarray(xy, float))
    e = xy[:, 0] * math.cos(n) + xy[:, 1] * math.sin(n)
    nn = -xy[:, 0] * math.sin(n) + xy[:, 1] * math.cos(n)
    return np.stack([e, nn], axis=1)


def grid_zone(xy, polygon: np.ndarray, north_deg: float) -> str:
    en_poly = to_east_north(polygon, north_deg)
    (e0, n0), (e1, n1) = en_poly.min(axis=0), en_poly.max(axis=0)
    e, n = to_east_north(xy, north_deg)[0]
    col = int(np.clip((e - e0) / max(e1 - e0, 1e-9) * 3, 0, 2))
    row = int(np.clip((n1 - n) / max(n1 - n0, 1e-9) * 3, 0, 2))
    return GRID[row][col]


def wall_facing(start, end, north_deg: float) -> float:
    """Bearing of the inward normal of a wall of a CCW polygon (interior on the left)."""
    t = np.asarray(end, float) - np.asarray(start, float)
    n_in = np.array([-t[1], t[0]])
    return bearing(n_in[0], n_in[1], north_deg)


def zone_report(model, system: str = "16") -> dict:
    """Objects and openings per zone for every room: ``{room_id: {zone: [items]}}``."""
    if model.north_deg is None:
        raise ValueError("north_deg unknown: zone queries need a compass heading")
    out: dict = {}
    for room in model.rooms:
        poly = np.asarray(room.floor_polygon)
        c = np.asarray(room.centroid)
        zones: dict[str, list] = {}

        def put(xy, item):
            if system == "grid":
                z = grid_zone(xy, poly, model.north_deg)
            else:
                b = bearing(xy[0] - c[0], xy[1] - c[1], model.north_deg)
                z = direction_16(b) if system == "16" else direction_8(b)
            zones.setdefault(z, []).append(item)

        for o in room.objects:
            put(o.position[:2], {"kind": "object", "id": o.id, "label": o.label, "element": o.element})
        for op in room.openings:
            put(op.center[:2], {"kind": op.type, "id": op.id, "wall_id": op.wall_id})
        out[room.id] = zones
    return out
