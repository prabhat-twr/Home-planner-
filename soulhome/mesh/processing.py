"""Mesh / point-cloud ingestion and cleanup for the LiDAR and ARCore tiers.

Everything is normalised into the contract frame (metres, +Z up, floor at z=0); the composed 4x4
transform is returned so camera poses and provided 3D detections can follow the same path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

Y_UP_TO_Z_UP = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1.0]])
UNIT_SCALES = {"m": 1.0, "cm": 0.01, "mm": 0.001, "in": 0.0254, "ft": 0.3048}


@dataclass
class Loaded:
    mesh: object | None  # trimesh.Trimesh, None for raw point clouds
    points: np.ndarray | None  # (N,3) when the input is a point cloud
    up_axis: str
    meters_per_unit: float | None
    fmt: str
    notes: list[str] = field(default_factory=list)


def load_scan(path: str | Path, up_axis: str | None = None) -> Loaded:
    import trimesh

    path = Path(path)
    ext = path.suffix.lower().lstrip(".")
    if ext == "usdz" or ext in ("usd", "usda", "usdc"):
        return _load_usd(path)
    if ext == "npy":  # raw (N,3) point cloud
        return Loaded(None, np.load(path).astype(float), up_axis or "z", None, ext)
    obj = trimesh.load(path, force=None, process=False)
    if isinstance(obj, trimesh.Scene):
        obj = obj.to_geometry() if hasattr(obj, "to_geometry") else obj.dump(concatenate=True)
    default_up = "y" if ext in ("glb", "gltf") else "z"  # glTF is Y-up by spec
    up = up_axis or default_up
    if isinstance(obj, trimesh.PointCloud) or (isinstance(obj, trimesh.Trimesh) and len(obj.faces) == 0):
        return Loaded(None, np.asarray(obj.vertices, float), up, 1.0 if ext in ("glb", "gltf") else None, ext)
    return Loaded(obj, None, up, 1.0 if ext in ("glb", "gltf") else None, ext)


def _load_usd(path: Path) -> Loaded:
    """Minimal USD/USDZ mesh reader (iOS LiDAR / RoomPlan exports): flattens every UsdGeom.Mesh into
    world space, fan-triangulates polygons, honours ``upAxis`` and ``metersPerUnit``."""
    import trimesh
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(path))
    up = UsdGeom.GetStageUpAxis(stage).lower()
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    verts, faces = [], []
    cache = UsdGeom.XformCache()
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        m = UsdGeom.Mesh(prim)
        pts = np.asarray(m.GetPointsAttr().Get() or [], float)
        counts = np.asarray(m.GetFaceVertexCountsAttr().Get() or [], int)
        idx = np.asarray(m.GetFaceVertexIndicesAttr().Get() or [], int)
        if len(pts) == 0:
            continue
        xf = np.asarray(cache.GetLocalToWorldTransform(prim), float)  # row-vector convention
        pts = np.c_[pts, np.ones(len(pts))] @ xf
        base = sum(len(v) for v in verts)
        verts.append(pts[:, :3])
        cursor = 0
        for c in counts:
            poly = idx[cursor : cursor + c]
            for k in range(1, c - 1):
                faces.append([base + poly[0], base + poly[k], base + poly[k + 1]])
            cursor += c
    if not verts:
        raise ValueError("USD file contains no meshes")
    mesh = trimesh.Trimesh(np.concatenate(verts), np.asarray(faces), process=False)
    return Loaded(mesh, None, up, float(mpu), path.suffix.lstrip("."))


def guess_units(extent_max: float) -> tuple[float, str]:
    """Pick the unit scale that makes a room-sized scan between 1.5 m and 60 m across."""
    for name in ("m", "cm", "mm", "in"):
        s = UNIT_SCALES[name]
        if 1.5 <= extent_max * s <= 60:
            return s, name
    return 1.0, "m"


def base_transform(loaded: Loaded, units: str | None, extent_max: float) -> tuple[np.ndarray, list[str]]:
    notes = []
    t = np.eye(4)
    if loaded.up_axis == "y":
        t = Y_UP_TO_Z_UP @ t
        notes.append("converted Y-up -> Z-up")
    if units:
        s = UNIT_SCALES[units]
    elif loaded.meters_per_unit:
        s = loaded.meters_per_unit
    else:
        s, name = guess_units(extent_max)
        if name != "m":
            notes.append(f"units auto-detected as {name}")
    t = np.diag([s, s, s, 1.0]) @ t
    return t, notes


def cleanup(mesh, min_component_area: float = 0.02):
    """Merge duplicate vertices, drop degenerate/duplicate faces and tiny floating fragments."""
    m = mesh.copy()
    m.merge_vertices()
    m.update_faces(m.nondegenerate_faces())
    m.update_faces(m.unique_faces())
    m.remove_unreferenced_vertices()
    before = len(m.faces)
    # connected components over face adjacency (scipy; avoids trimesh's networkx path)
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    adj = m.face_adjacency
    n = len(m.faces)
    graph = coo_matrix((np.ones(len(adj)), (adj[:, 0], adj[:, 1])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    comp_area = np.bincount(labels, weights=m.area_faces)
    keep = comp_area[labels] >= min_component_area
    if keep.any() and not keep.all():
        m.update_faces(keep)
        m.remove_unreferenced_vertices()
    return m, {"faces_in": int(len(mesh.faces)), "faces_out": int(len(m.faces)), "dropped_fragment_faces": before - int(len(m.faces))}


def decimate(mesh, max_faces: int = 300_000):
    if len(mesh.faces) <= max_faces:
        return mesh, False
    return mesh.simplify_quadric_decimation(face_count=max_faces), True


def sample_points(mesh, density: float = 4000.0, max_points: int = 800_000, seed: int = 0) -> np.ndarray:
    import trimesh

    n = int(min(max_points, max(20_000, mesh.area * density)))
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return np.asarray(pts, float)


def voxel_downsample(points: np.ndarray, voxel: float = 0.02) -> np.ndarray:
    keys = np.floor(points / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(idx)]


def export_glb(mesh, path: str | Path) -> None:
    """Export in glTF convention (Y-up)."""
    m = mesh.copy()
    m.apply_transform(np.linalg.inv(Y_UP_TO_Z_UP))
    m.export(str(path), file_type="glb")
