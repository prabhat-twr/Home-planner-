"""Regression suite: every tier must reproduce the ground-truth room within tight tolerances.
Silent geometric drift fails CI here, not in the rule engine."""

import json

import numpy as np
import pytest
import trimesh

from soulhome.contract import SpatialModel
from soulhome.demo import DEMO_META, write_mesh_demo, write_panorama_demo
from soulhome.geometry.polygon import polygon_iou
from soulhome.pipeline.run import ScanMeta, run_scan
from soulhome.pipeline.stages import PipelineError

TRUE_AREA = 21.25


def _check_room(model, tol_area=0.02, n_walls=6):
    r = model.rooms[0]
    assert len(r.walls) == n_walls
    assert abs(r.floor_area_m2 - TRUE_AREA) / TRUE_AREA < tol_area
    assert abs(r.ceiling_height_m - 2.7) < 0.03
    doors = [o for o in r.openings if o.type == "door"]
    windows = [o for o in r.openings if o.type == "window"]
    assert len(doors) == 1 and len(windows) == 1
    assert abs(doors[0].width_m - 0.9) < 0.11 and abs(doors[0].height_m - 2.05) < 0.08
    assert abs(windows[0].width_m - 1.2) < 0.15 and abs(windows[0].sill_height_m - 0.9) < 0.08
    # contract round-trips
    SpatialModel.model_validate_json(model.model_dump_json())
    return r


def test_panorama_depth_tier(tmp_path, room):
    meta = write_panorama_demo(tmp_path / "in")
    m = run_scan("t", "panorama", tmp_path / "in", tmp_path / "out", ScanMeta(**meta))
    r = _check_room(m)
    assert polygon_iou(np.asarray(r.floor_polygon), room.polygon) > 0.99
    # objects land on their footprints and get Vastu directions
    truth = {o.label: np.asarray(o.center) for o in room.objects}
    for o in r.objects:
        assert np.linalg.norm(np.asarray(o.position[:2]) - truth[o.label]) < 0.3, o
        assert o.direction_16 and o.grid_zone
    assert {o.label for o in r.objects} == set(truth)
    assert (tmp_path / "out" / "model.json").exists() and (tmp_path / "out" / "floorplan.svg").exists()


def test_panorama_layout_net_tier(tmp_path, room):
    meta = write_panorama_demo(tmp_path / "in", with_layout=True)
    m = run_scan("t", "panorama", tmp_path / "in", tmp_path / "out", ScanMeta(**meta, layout_backend="horizonnet"))
    layout = next(s for s in m.provenance.stages if s.name == "layout")
    assert layout.detail["method"].startswith("horizonnet")
    _check_room(m)


def test_panorama_floor_ray_placement_without_depth(tmp_path, room):
    """Layout-net only (no depth): objects are placed by floor-ray back-projection."""
    d = tmp_path / "in"
    write_panorama_demo(d, with_layout=True)
    (d / "depth.npy").unlink()
    (d / "panorama.png").unlink()
    m = run_scan("t", "panorama", d, tmp_path / "out", ScanMeta(camera_height_m=1.5, north_deg=0))
    truth = {o.label: np.asarray(o.center) for o in room.objects}
    r = m.rooms[0]
    assert all(o.placement_method == "floor_ray" for o in r.objects)
    for o in r.objects:
        assert np.linalg.norm(np.asarray(o.position[:2]) - truth[o.label]) < 0.35, o


@pytest.mark.parametrize("tier", ["lidar", "arcore"])
def test_mesh_tiers(tmp_path, tier):
    meta = write_mesh_demo(tmp_path / "in", tier)
    m = run_scan("t", tier, tmp_path / "in", tmp_path / "out", ScanMeta(**meta))
    r = _check_room(m)
    # the AR world frame is rotated 17 deg; walls must still be exactly orthogonal
    for w in r.walls:
        d = np.subtract(w.end, w.start)
        ang = np.degrees(np.arctan2(d[1], d[0])) % 90
        assert min(abs(ang - 17), abs(ang - 17 - 90), abs(ang + 90 - 17)) < 0.5, ang
    assert len(r.objects) == 3
    # the same physical room gives the same Vastu answers as the panorama tier
    by_label = {o.label: o.grid_zone for o in r.objects}
    assert by_label["potted plant"] == "N" and by_label["sofa"] == "W"
    if tier == "lidar":
        assert (tmp_path / "out" / "mesh_clean.glb").exists()
        cleanup = next(s for s in m.provenance.stages if s.name == "mesh_cleanup")
        assert cleanup.detail["dropped_fragment_faces"] > 0  # floating noise removed


def test_unit_autodetect_millimetres(tmp_path):
    write_mesh_demo(tmp_path / "in", "lidar")
    mesh = trimesh.load(tmp_path / "in" / "scan.glb", force="mesh")
    (tmp_path / "in" / "scan.glb").unlink()
    mesh.apply_scale(1000.0)
    mesh.export(tmp_path / "in" / "scan.ply")  # PLY: no unit metadata, Y-up given by the client
    (tmp_path / "in" / "detections.json").write_text(json.dumps({"detections": []}))
    m = run_scan("t", "lidar", tmp_path / "in", tmp_path / "out", ScanMeta(up_axis="y"))
    assert any("mm" in w for w in m.quality.warnings)
    assert abs(m.rooms[0].floor_area_m2 - TRUE_AREA) / TRUE_AREA < 0.02


def test_usdz_input(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    from soulhome.synthetic import l_shaped_room, room_mesh

    mesh = room_mesh(l_shaped_room()).subdivide_to_size(0.25)
    path = tmp_path / "in" / "scan.usdc"
    path.parent.mkdir()
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 0.01)  # centimetres, as some exporters write
    g = UsdGeom.Mesh.Define(stage, "/room")
    g.CreatePointsAttr([tuple(v * 100) for v in mesh.vertices])
    g.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
    g.CreateFaceVertexIndicesAttr(mesh.faces.reshape(-1).tolist())
    stage.Save()
    m = run_scan("t", "lidar", path.parent, tmp_path / "out", ScanMeta())
    assert abs(m.rooms[0].floor_area_m2 - TRUE_AREA) / TRUE_AREA < 0.02


def test_missing_input_is_non_retryable(tmp_path):
    (tmp_path / "in").mkdir()
    with pytest.raises(PipelineError) as e:
        run_scan("t", "lidar", tmp_path / "in", tmp_path / "out")
    assert e.value.stage == "load_inputs" and not e.value.retryable


def test_degenerate_geometry_is_rejected(tmp_path):
    """A 'room' squashed to 1.2 m tall must be dead-lettered, never shipped."""
    from soulhome.synthetic import SynthRoom, render_depth_panorama
    from soulhome.geometry.polygon import ensure_ccw

    tiny = SynthRoom(ensure_ccw(np.array([(-1.5, -1.0), (1.5, -1.0), (1.5, 1.0), (-1.5, 1.0)])), ceiling=1.2, camera=(0, 0, 0.6))
    d = tmp_path / "in"
    d.mkdir()
    np.save(d / "depth.npy", render_depth_panorama(tiny))
    with pytest.raises(PipelineError) as e:
        run_scan("t", "panorama", d, tmp_path / "out", ScanMeta())
    assert e.value.code == "CEILING_IMPLAUSIBLE" and not e.value.retryable


def test_demo_metadata_is_valid():
    for meta in DEMO_META.values():
        ScanMeta(**meta)
