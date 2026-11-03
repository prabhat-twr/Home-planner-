"""Spatial data contract consumed by the downstream Vastu / Feng Shui rule engine.

This is the *only* thing the rule engine sees, so it is versioned with semver:

* PATCH - documentation / tolerances, no shape change
* MINOR - additive optional fields (old consumers keep working)
* MAJOR - anything that removes/renames a field or changes a unit/frame

Coordinate frame (all tiers are normalised into it):
    right-handed, metres, +Z up, floor plane at z = 0.
    +Y is the capture "forward" axis (panorama centre column / first AR frame).
    ``north_deg`` is the compass bearing (clockwise from true north) of +Y,
    so the bearing of a horizontal vector (dx, dy) is
    ``(north_deg + degrees(atan2(dx, dy))) % 360``.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator

SCHEMA_VERSION = "1.0.0"

Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]


class Tier(str, Enum):
    lidar = "lidar"  # iOS LiDAR mesh (USDZ / GLB / PLY / OBJ)
    arcore = "arcore"  # Android ARCore depth -> fused point cloud / mesh
    panorama = "panorama"  # photo / panorama fallback (monocular depth + layout nets)


class Opening(BaseModel):
    id: str
    type: Literal["door", "window", "opening"]
    wall_id: str
    offset_m: float = Field(description="distance from wall start to opening start, along the wall")
    width_m: float
    sill_height_m: float
    height_m: float
    center: Vec3


class Wall(BaseModel):
    id: str
    start: Vec2
    end: Vec2
    length_m: float
    height_m: float
    facing_deg: float | None = Field(
        None, description="compass bearing of the wall's inward-facing normal; null if north unknown"
    )


class PlacedObject(BaseModel):
    id: str
    catalog_id: str
    label: str
    category: str
    element: str | None = None
    confidence: float
    position: Vec3 = Field(description="floor-contact point (x, y, 0) or 3D centroid for wall-mounted items")
    size: Vec3 | None = Field(None, description="approximate (width, depth, height) in metres")
    placement_method: Literal["floor_ray", "depth_median", "mesh_raycast", "provided"]
    direction_16: str | None = None
    grid_zone: str | None = None


class Room(BaseModel):
    id: str
    name: str = "room"
    floor_polygon: list[Vec2] = Field(description="counter-clockwise, metres, closed implicitly")
    floor_area_m2: float
    ceiling_height_m: float
    centroid: Vec2
    walls: list[Wall]
    openings: list[Opening] = []
    objects: list[PlacedObject] = []

    @field_validator("floor_polygon")
    @classmethod
    def _min_vertices(cls, v: list[Vec2]) -> list[Vec2]:
        if len(v) < 3:
            raise ValueError("floor polygon needs >= 3 vertices")
        return v


class StageReport(BaseModel):
    name: str
    status: Literal["ok", "skipped", "failed"]
    duration_ms: float
    detail: dict = {}


class Quality(BaseModel):
    confidence: float = Field(ge=0, le=1)
    warnings: list[str] = []


class Provenance(BaseModel):
    tier: Tier
    pipeline_version: str
    stages: list[StageReport]
    models: dict[str, str] = {}


class SpatialModel(BaseModel):
    schema_version: str = SCHEMA_VERSION
    scan_id: str
    units: Literal["m"] = "m"
    frame: str = "right-handed, +Z up, floor at z=0, +Y = capture forward"
    north_deg: float | None = None
    rooms: list[Room]
    provenance: Provenance
    quality: Quality
