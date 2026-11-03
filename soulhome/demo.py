"""Write synthetic scan uploads (same file layout a mobile client would send) for demos and tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .mesh.processing import Y_UP_TO_Z_UP
from .synthetic import horizonnet_outputs, l_shaped_room, panorama_detections, render_depth_panorama, room_mesh, shaded_panorama

DEMO_META = {
    "panorama": {"camera_height_m": 1.5, "north_deg": 30.0, "room_name": "living room (synthetic panorama)"},
    # the mesh demos are captured in an AR world frame rotated 17 deg from the room, so the
    # same physical orientation needs a heading of 30 + 17
    "lidar": {"north_deg": 47.0, "room_name": "living room (synthetic LiDAR)"},
    "arcore": {"north_deg": 47.0, "room_name": "living room (synthetic ARCore)"},
}


def write_panorama_demo(out: Path, with_layout: bool = False) -> dict:
    import cv2

    out.mkdir(parents=True, exist_ok=True)
    room = l_shaped_room()
    np.save(out / "depth.npy", render_depth_panorama(room))
    cv2.imwrite(str(out / "panorama.png"), shaded_panorama(room))
    (out / "detections.json").write_text(json.dumps({"image_size": [1024, 512], "detections": panorama_detections(room)}))
    if with_layout:
        c, f, k = horizonnet_outputs(room)
        np.savez(out / "layout.npz", ceil_lat=c, floor_lat=f, corner=k)
    return DEMO_META["panorama"]


def _scene_transform(yaw_deg: float, offset=(0.4, -0.7, 0.0)) -> np.ndarray:
    """Arbitrary AR-session world frame: rotated about gravity and offset."""
    y = np.radians(yaw_deg)
    t = np.eye(4)
    t[:2, :2] = [[np.cos(y), -np.sin(y)], [np.sin(y), np.cos(y)]]
    t[:3, 3] = offset
    return t


def write_mesh_demo(out: Path, tier: str = "lidar", yaw_deg: float = 17.0, noise: float = 0.004, seed: int = 0) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    room = l_shaped_room()
    mesh = room_mesh(room)
    import trimesh

    mesh = mesh.subdivide_to_size(0.25)
    rng = np.random.default_rng(seed)
    mesh.vertices += rng.normal(0, noise, mesh.vertices.shape)  # sensor noise
    # floating fragments (reflections / moving people) that cleanup must drop
    frags = [trimesh.creation.icosphere(subdivisions=0, radius=0.03) for _ in range(5)]
    for fr in frags:
        fr.apply_translation(rng.uniform([-1.5, -1, 0.5], [2.5, 1.5, 2.2]))
    mesh = trimesh.util.concatenate([mesh, *frags])
    world = _scene_transform(yaw_deg)
    to_src = np.linalg.inv(Y_UP_TO_Z_UP) @ world  # Z-up room -> Y-up AR world (glTF / ARCore convention)
    mesh.apply_transform(to_src)
    dets = []
    for obj in room.objects:
        p = to_src @ np.array([*obj.center, 0.0, 1.0])
        dets.append({"label": obj.label, "confidence": 0.95, "position": p[:3].tolist(), "size": list(obj.size)})
    if tier == "arcore":
        pts, _ = trimesh.sample.sample_surface(mesh, 250_000, seed=seed)
        trimesh.PointCloud(pts).export(str(out / "scan.ply"))
    else:
        mesh.export(str(out / "scan.glb"))
    (out / "detections.json").write_text(json.dumps({"detections": dets}))
    return DEMO_META[tier]


__all__ = ["write_panorama_demo", "write_mesh_demo", "DEMO_META"]
