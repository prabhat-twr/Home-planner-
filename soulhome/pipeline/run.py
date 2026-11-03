"""End-to-end scan processing for all three capture tiers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel

from ..contract import Provenance, Quality, SpatialModel, Tier
from ..geometry.equirect import equirect_depth_to_points
from ..geometry.floorplan import RoomGeometry, detect_openings, extract_room
from ..geometry.fusion import scale_depth_to_layout
from ..geometry.planes import align_floor
from ..layout.horizonnet import decode_layout
from ..objects.placement import dedupe_placements, equirect_pixels_of_points, place_equirect, place_perspective, place_provided
from ..spatial.render import render_svg
from .assemble import build_room, confidence, validate_room
from .stages import PipelineError, StageRecorder

PIPELINE_VERSION = "0.1.0"
MESH_EXTS = ("usdz", "usd", "usdc", "usda", "glb", "gltf", "ply", "obj", "npy")
IMAGE_EXTS = ("jpg", "jpeg", "png", "webp")


class ScanMeta(BaseModel):
    camera_height_m: float | None = None  # guided capture asks the user to hold the phone at chest height
    north_deg: float | None = None  # compass bearing of the capture's forward axis
    room_name: str = "room"
    up_axis: Literal["y", "z"] | None = None
    units: Literal["m", "cm", "mm", "in", "ft"] | None = None
    layout_backend: Literal["auto", "depth", "horizonnet"] = "auto"


def _find(input_dir: Path, stem: str | None, exts) -> Path | None:
    for p in sorted(input_dir.iterdir()):
        if p.suffix.lower().lstrip(".") in exts and (stem is None or p.stem == stem):
            return p
    return None


def run_scan(
    scan_id: str,
    tier: Tier | str,
    input_dir: str | Path,
    output_dir: str | Path,
    meta: ScanMeta | None = None,
    inference=None,
    on_stage=None,
) -> SpatialModel:
    tier = Tier(tier)
    meta = meta or ScanMeta()
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rec = StageRecorder(scan_id, on_stage)
    models: dict[str, str] = dict(getattr(inference, "models", {}) or {})

    if tier is Tier.panorama:
        geom, placements, extra_warnings = _panorama(input_dir, meta, inference, rec, models)
    else:
        geom, placements, extra_warnings = _mesh(tier, input_dir, output_dir, meta, rec)

    with rec.stage("assemble") as d:
        room = build_room(geom, placements, meta.north_deg, meta.room_name)
        d.update(walls=len(room.walls), openings=len(room.openings), objects=len(room.objects))
    with rec.stage("validate") as d:
        warnings = extra_warnings + validate_room(room)
        d["warnings"] = len(warnings)
    if meta.north_deg is None:
        warnings.append("north_deg missing: direction/zone fields left empty")
    model = SpatialModel(
        scan_id=scan_id,
        north_deg=meta.north_deg,
        rooms=[room],
        provenance=Provenance(tier=tier, pipeline_version=PIPELINE_VERSION, stages=rec.reports, models=models),
        quality=Quality(confidence=confidence(tier.value, warnings), warnings=warnings),
    )
    with rec.stage("write_artifacts"):
        (output_dir / "floorplan.svg").write_text(render_svg(model))
    model.provenance.stages = rec.reports
    (output_dir / "model.json").write_text(model.model_dump_json(indent=2))
    return model


# ------------------------------------------------------------------------------------ panorama


def _panorama(input_dir: Path, meta: ScanMeta, inference, rec: StageRecorder, models: dict):
    warnings: list[str] = []
    with rec.stage("load_inputs") as d:
        img_path = _find(input_dir, None, IMAGE_EXTS)
        depth_path = _find(input_dir, "depth", ("npy",))
        layout_path = _find(input_dir, "layout", ("npz",))
        det_path = _find(input_dir, "detections", ("json",))
        rgb = None
        if img_path:
            import cv2

            bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise PipelineError("load_inputs", "IMAGE_UNREADABLE", img_path.name)
            h, w = bgr.shape[:2]
            if abs(w / h - 2.0) > 0.05:
                raise PipelineError("load_inputs", "NOT_EQUIRECTANGULAR", f"aspect {w}x{h}, expected 2:1")
            if w > 2048:
                bgr = cv2.resize(bgr, (2048, 1024), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb is None and depth_path is None and layout_path is None:
            raise PipelineError("load_inputs", "NO_INPUT", "need a panorama image, depth.npy or layout.npz")
        d.update(image=bool(img_path), depth=bool(depth_path), layout=bool(layout_path), detections=bool(det_path))

    img_wh = (rgb.shape[1], rgb.shape[0]) if rgb is not None else None
    cam_h = meta.camera_height_m or 1.6  # HorizonNet's training convention when unknown

    # 1) layout network: metric room polygon from per-column floor/ceiling boundaries
    layout_out = None
    if meta.layout_backend != "depth":
        if layout_path is not None:
            z = np.load(layout_path)
            layout_out = (z["ceil_lat"], z["floor_lat"], z["corner"])
        elif rgb is not None and inference is not None:
            with rec.stage("layout_inference") as d:
                layout_out = inference.layout(rgb)
                d["available"] = layout_out is not None
    if meta.layout_backend == "horizonnet" and layout_out is None:
        raise PipelineError("layout", "LAYOUT_UNAVAILABLE", "layout_backend=horizonnet but no layout output")

    # 2) depth: provided metric depth (tests / devices with depth) or the monocular model
    pts_cam = None
    if depth_path is not None:
        with rec.stage("depth") as d:
            depth = np.load(depth_path)
            pts_cam = equirect_depth_to_points(depth, max(1, depth.shape[1] // 1024))
            img_wh = img_wh or (depth.shape[1], depth.shape[0])
            d.update(source="provided", points=len(pts_cam))
    elif rgb is not None and inference is not None:
        with rec.stage("depth") as d:
            pts_cam = inference.depth_points(rgb)
            d.update(source="model", points=len(pts_cam))
    else:
        rec.skip("depth", "no depth input and no inference backend")
    pix = equirect_pixels_of_points(pts_cam, *img_wh) if (pts_cam is not None and img_wh) else None

    pts = None
    with rec.stage("layout") as d:
        if layout_out is not None:
            dec = decode_layout(*layout_out, camera_height=cam_h)
            geom = RoomGeometry(dec.polygon, dec.ceiling_height, dec.yaw)
            cam = np.array([0.0, 0.0, cam_h])
            d.update(method=f"horizonnet:{dec.method}", corners=len(dec.corner_columns))
            if pts_cam is not None:
                pts, fstats = scale_depth_to_layout(pts_cam, geom.polygon, cam_h, geom.ceiling_height)
                d["depth_fusion"] = fstats
                if fstats["scale_spread"] > 1.6:
                    warnings.append(f"monocular depth inconsistent with layout (spread {fstats['scale_spread']})")
        elif pts_cam is not None:
            # depth-only: find the floor (pano is captured upright, so allow only a small tilt)
            align = align_floor(pts_cam, max_tilt_deg=5.0)
            pts = align.apply(pts_cam)
            cam = align.apply(np.zeros((1, 3)))[0]
            est_h, scale = float(cam[2]), 1.0
            if meta.camera_height_m:
                scale = meta.camera_height_m / est_h
                pts, cam = pts * scale, cam * scale
                if abs(scale - 1) > 0.25:
                    warnings.append(f"depth scale corrected by {scale:.2f}x using camera height")
            geom = extract_room(pts, "viewpoint", viewpoint=cam[:2], find_openings=False)
            d.update(method="depth:angular_sweep", estimated_camera_height_m=round(est_h, 3), scale=round(scale, 4))
        else:
            raise PipelineError("layout", "NO_LAYOUT_SIGNAL", "neither layout-net output nor depth available")
        d.update(vertices=len(geom.polygon), ceiling_m=round(geom.ceiling_height, 3), yaw_deg=round(float(np.degrees(geom.yaw)), 2))

    if pts is not None:
        with rec.stage("openings") as d:
            geom.openings = detect_openings(pts, geom.polygon, geom.ceiling_height)
            d.update(doors=sum(o.type == "door" for o in geom.openings), windows=sum(o.type == "window" for o in geom.openings))
    else:
        rec.skip("openings", "no depth")

    dets, det_wh = [], img_wh
    if det_path is not None:
        raw = json.loads(det_path.read_text())
        dets = raw["detections"] if isinstance(raw, dict) else raw
        if isinstance(raw, dict) and raw.get("image_size"):
            det_wh = tuple(raw["image_size"])
    elif rgb is not None and inference is not None:
        with rec.stage("detect") as d:
            dets = inference.detect(rgb)
            d["detections"] = len(dets)
    placements = []
    if dets and det_wh:
        with rec.stage("place_objects") as d:
            if pix is not None and det_wh != img_wh:
                pix = pix * (np.asarray(det_wh) / np.asarray(img_wh))
            for det in dets:
                p = place_equirect(det, det_wh[0], det_wh[1], cam, geom.polygon, pts, pix)
                if p is not None:
                    placements.append(p)
            n = len(placements)
            placements = dedupe_placements(placements)
            d.update(placed=len(placements), duplicates_merged=n - len(placements), unmapped=len(dets) - n)
    else:
        rec.skip("place_objects", "no detections")
    return geom, placements, warnings


# ---------------------------------------------------------------------------------- mesh tiers


def _mesh(tier: Tier, input_dir: Path, output_dir: Path, meta: ScanMeta, rec: StageRecorder):
    from ..mesh import processing as mp

    warnings: list[str] = []
    with rec.stage("load_inputs") as d:
        path = _find(input_dir, "scan", MESH_EXTS) or _find(input_dir, None, MESH_EXTS)
        if path is None:
            raise PipelineError("load_inputs", "NO_INPUT", f"expected a mesh/point cloud ({', '.join(MESH_EXTS)})")
        up = meta.up_axis or ("y" if tier is Tier.arcore else None)  # ARCore world space is Y-up
        loaded = mp.load_scan(path, up)
        d.update(format=loaded.fmt, up_axis=loaded.up_axis, kind="mesh" if loaded.mesh is not None else "points")

    with rec.stage("normalize_units") as d:
        raw = loaded.mesh.vertices if loaded.mesh is not None else loaded.points
        extent = float(np.ptp(np.asarray(raw), axis=0).max())
        T, notes = mp.base_transform(loaded, meta.units, extent)
        warnings += notes
        mesh = pts = None
        if loaded.mesh is not None:
            mesh = loaded.mesh.copy()
            mesh.apply_transform(T)
        else:
            pts = np.c_[loaded.points, np.ones(len(loaded.points))] @ T.T
            pts = pts[:, :3]
        d.update(raw_extent=round(extent, 3), scale=round(float(T[0, :3].max() or T[0, :3].min() or 1), 5), notes=notes)

    if mesh is not None:
        with rec.stage("mesh_cleanup") as d:
            mesh, stats = mp.cleanup(mesh)
            d.update(stats)
        with rec.stage("decimate") as d:
            mesh, did = mp.decimate(mesh)
            d.update(decimated=did, faces=int(len(mesh.faces)))
        with rec.stage("sample_points") as d:
            pts = mp.sample_points(mesh)
            d["points"] = len(pts)
    else:
        with rec.stage("downsample") as d:
            pts = mp.voxel_downsample(pts, 0.02)
            d["points"] = len(pts)

    with rec.stage("gravity_align") as d:
        align = align_floor(pts)
        G = np.eye(4)
        G[:3, :3] = align.rotation
        G[2, 3] = -align.z_offset
        pts = align.apply(pts)
        if mesh is not None:
            mesh.apply_transform(G)
        T = G @ T
        tilt = float(np.degrees(np.arccos(np.clip(align.rotation[2, 2], -1, 1))))
        d.update(tilt_correction_deg=round(tilt, 2), floor_inlier_ratio=round(float(align.floor.inliers.mean()), 3))

    with rec.stage("floorplan") as d:
        geom = extract_room(pts, "occupancy")
        d.update(vertices=len(geom.polygon), ceiling_m=round(geom.ceiling_height, 3),
                 yaw_deg=round(float(np.degrees(geom.yaw)), 2),
                 doors=sum(o.type == "door" for o in geom.openings),
                 windows=sum(o.type == "window" for o in geom.openings))

    placements = []
    det_path = _find(input_dir, "detections", ("json",))
    if det_path is not None:
        with rec.stage("place_objects") as d:
            raw = json.loads(det_path.read_text())
            dets = raw["detections"] if isinstance(raw, dict) else raw
            for det in dets:
                if "position" in det:
                    p = np.asarray(det["position"], float)
                    if det.get("frame", "source") == "source":
                        p = (T @ np.r_[p, 1.0])[:3]
                    p = place_provided({**det, "position": p}, geom.polygon)
                elif "cam_to_world" in det:
                    pose = np.asarray(det["cam_to_world"], float)
                    if det.get("camera_convention", "opencv") == "arkit":
                        pose = pose @ np.diag([1, -1, -1, 1.0])
                    p = place_perspective(det, np.asarray(det["intrinsics"]), T @ pose, geom.polygon)
                else:
                    p = None
                if p is not None:
                    placements.append(p)
            d.update(placed=len(placements), unmapped=len(dets) - len(placements))
    else:
        rec.skip("place_objects", "no detections")

    if mesh is not None:
        with rec.stage("export_mesh") as d:
            mp.export_glb(mesh, output_dir / "mesh_clean.glb")
            d["faces"] = int(len(mesh.faces))
    return geom, placements, warnings
