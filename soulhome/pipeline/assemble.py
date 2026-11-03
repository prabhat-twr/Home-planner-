"""Turn geometry + placements into the versioned contract, and gate it with validation."""

from __future__ import annotations

import numpy as np

from ..contract import Opening, PlacedObject, Room, Wall
from ..geometry.floorplan import RoomGeometry
from ..objects.placement import Placement
from ..spatial.zones import bearing, direction_16, grid_zone, wall_facing
from .stages import PipelineError


def _r(x, n=3):
    return round(float(x), n)


def build_room(geom: RoomGeometry, placements: list[Placement], north_deg: float | None, name: str = "room") -> Room:
    from shapely.geometry import Polygon

    poly = np.asarray(geom.polygon)
    sp = Polygon(poly)
    centroid = (sp.centroid.x, sp.centroid.y)
    walls = []
    for i, (a, b) in enumerate(zip(poly, np.roll(poly, -1, axis=0))):
        walls.append(
            Wall(
                id=f"w{i}",
                start=(_r(a[0]), _r(a[1])),
                end=(_r(b[0]), _r(b[1])),
                length_m=_r(np.linalg.norm(b - a)),
                height_m=_r(geom.ceiling_height),
                facing_deg=None if north_deg is None else _r(wall_facing(a, b, north_deg), 1),
            )
        )
    openings = []
    for k, o in enumerate(geom.openings):
        a, b = poly[o.wall_index], poly[(o.wall_index + 1) % len(poly)]
        t = (b - a) / np.linalg.norm(b - a)
        c = a + t * (o.offset_m + o.width_m / 2)
        openings.append(
            Opening(
                id=f"o{k}",
                type=o.type,
                wall_id=f"w{o.wall_index}",
                offset_m=_r(o.offset_m),
                width_m=_r(o.width_m),
                sill_height_m=_r(o.sill_height_m),
                height_m=_r(o.height_m),
                center=(_r(c[0]), _r(c[1]), _r(o.sill_height_m + o.height_m / 2)),
            )
        )
    objects = []
    for k, p in enumerate(placements):
        d16 = gz = None
        if north_deg is not None:
            d16 = direction_16(bearing(p.position[0] - centroid[0], p.position[1] - centroid[1], north_deg))
            gz = grid_zone(p.position[:2], poly, north_deg)
        objects.append(
            PlacedObject(
                id=f"obj{k}",
                catalog_id=p.entry.id,
                label=p.entry.label,
                category=p.entry.category,
                element=p.entry.element,
                confidence=_r(p.confidence),
                position=tuple(_r(v) for v in p.position),
                size=None if p.size is None else tuple(_r(v) for v in p.size),
                placement_method=p.method,
                direction_16=d16,
                grid_zone=gz,
            )
        )
    return Room(
        id="r0",
        name=name,
        floor_polygon=[(_r(x), _r(y)) for x, y in poly],
        floor_area_m2=_r(sp.area, 2),
        ceiling_height_m=_r(geom.ceiling_height),
        centroid=(_r(centroid[0]), _r(centroid[1])),
        walls=walls,
        openings=openings,
        objects=objects,
    )


def validate_room(room: Room) -> list[str]:
    """Hard gates raise (the scan is dead-lettered instead of silently shipping bad geometry to the
    rule engine); soft issues come back as warnings that lower the confidence score."""
    from shapely.geometry import Point, Polygon

    poly = Polygon(room.floor_polygon)

    def fail(code, msg):
        raise PipelineError("validate", code, msg, retryable=False)

    if not poly.is_valid:
        fail("POLYGON_INVALID", "floor polygon self-intersects")
    if not 1.5 <= room.floor_area_m2 <= 300:
        fail("AREA_IMPLAUSIBLE", f"floor area {room.floor_area_m2} m2 outside [1.5, 300]")
    if not 1.9 <= room.ceiling_height_m <= 6.0:
        fail("CEILING_IMPLAUSIBLE", f"ceiling height {room.ceiling_height_m} m outside [1.9, 6.0]")
    if len(room.walls) < 4:
        fail("TOO_FEW_WALLS", f"{len(room.walls)} walls")
    if min(w.length_m for w in room.walls) < 0.1:
        fail("WALL_TOO_SHORT", "wall shorter than 0.1 m")

    warnings = []
    walls = {w.id: w for w in room.walls}
    for o in room.openings:
        w = walls[o.wall_id]
        if o.offset_m < -0.01 or o.offset_m + o.width_m > w.length_m + 0.01:
            warnings.append(f"{o.id}: opening extends past its wall")
        if o.sill_height_m + o.height_m > room.ceiling_height_m + 0.02:
            warnings.append(f"{o.id}: opening taller than the room")
        if o.type == "door" and not 1.7 <= o.height_m <= 2.6:
            warnings.append(f"{o.id}: unusual door height {o.height_m} m")
    if not any(o.type == "door" for o in room.openings):
        warnings.append("no door detected (main-entrance rules will be skipped)")
    for obj in room.objects:
        if poly.distance(Point(obj.position[:2])) > 0.05:
            warnings.append(f"{obj.id}: object outside room footprint")
    return warnings


TIER_PRIOR = {"lidar": 0.97, "arcore": 0.9, "panorama": 0.8}


def confidence(tier: str, warnings: list[str], coverage: float | None = None) -> float:
    c = TIER_PRIOR[tier] - 0.05 * len(warnings)
    if coverage is not None:
        c *= 0.5 + 0.5 * coverage
    return round(float(np.clip(c, 0.05, 1.0)), 3)
