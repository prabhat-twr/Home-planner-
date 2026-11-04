import io
import time

import pytest
from fastapi.testclient import TestClient

from soulhome.service.local import build
from soulhome.service.worker import process_job


@pytest.fixture()
def client(tmp_path):
    app, store, dispatch = build(str(tmp_path))
    c = TestClient(app)
    c.store, c.dispatch = store, dispatch
    return c


def wait(client, scan_id, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        body = client.get(f"/v1/scans/{scan_id}").json()
        if body["status"] in ("succeeded", "failed"):
            return body
        time.sleep(0.1)
    raise TimeoutError(body)


def test_health_and_index(client):
    assert client.get("/healthz").json()["ok"]
    assert "Soul Home" in client.get("/").text


def test_demo_end_to_end(client):
    r = client.post("/v1/demo/lidar")
    assert r.status_code == 202
    sid = r.json()["id"]
    body = wait(client, sid)
    assert body["status"] == "succeeded", body
    assert body["summary"]["walls"] == 6
    model = client.get(f"/v1/scans/{sid}/model").json()
    assert model["schema_version"] == "1.0.0"
    assert client.get(f"/v1/scans/{sid}/floorplan.svg").text.startswith("<svg")
    assert client.get(f"/v1/scans/{sid}/mesh.glb").content[:4] == b"glTF"
    zones = client.get(f"/v1/scans/{sid}/zones?system=grid").json()
    assert "N" in zones["rooms"]["r0"]
    metrics = client.get("/v1/metrics").json()
    assert metrics["by_status"]["succeeded"] == 1


def test_upload_idempotency_and_status(client, tmp_path):
    from soulhome.demo import write_mesh_demo

    d = tmp_path / "upload"
    write_mesh_demo(d, "lidar")
    files = [("files", ("scan.glb", (d / "scan.glb").read_bytes())), ("files", ("detections.json", (d / "detections.json").read_bytes()))]
    data = {"tier": "lidar", "metadata": '{"north_deg": 47}'}
    r1 = client.post("/v1/scans", data=data, files=files, headers={"Idempotency-Key": "abc"})
    r2 = client.post("/v1/scans", data=data, files=files, headers={"Idempotency-Key": "abc"})
    assert r1.status_code == 202 and r2.status_code == 200
    assert r1.json()["id"] == r2.json()["id"]
    # same key, different payload -> conflict
    r3 = client.post("/v1/scans", data={"tier": "lidar", "metadata": '{"north_deg": 10}'}, files=files, headers={"Idempotency-Key": "abc"})
    assert r3.status_code == 409
    assert wait(client, r1.json()["id"])["status"] == "succeeded"


def test_validation_errors(client):
    bad = client.post("/v1/scans", data={"tier": "lidar"}, files=[("files", ("x.exe", b"MZ"))])
    assert bad.status_code == 415
    bad = client.post("/v1/scans", data={"tier": "lidar", "metadata": "{not json"}, files=[("files", ("scan.ply", b"x"))])
    assert bad.status_code == 422
    assert client.get("/v1/scans/nope").status_code == 404


def test_bad_input_is_dead_lettered_without_retry(client):
    r = client.post("/v1/scans", data={"tier": "lidar"}, files=[("files", ("scan.ply", io.BytesIO(b"garbage")))])
    body = wait(client, r.json()["id"])
    assert body["status"] == "failed" and body["dead_lettered"]
    assert body["attempts"] == 1
    assert body["error"]["stage"] == "load_inputs"
    assert client.get(f"/v1/scans/{body['id']}/model").status_code == 409
    assert client.get("/v1/metrics").json()["failures_by_stage"]


def test_transient_failures_retry_then_succeed(client, monkeypatch):
    import soulhome.service.worker as worker

    real = worker.run_scan
    calls = {"n": 0}

    def flaky(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TimeoutError("GPU worker timed out")
        return real(*a, **kw)

    monkeypatch.setattr(worker, "run_scan", flaky)
    sid = client.post("/v1/demo/arcore").json()["id"]
    body = wait(client, sid)
    assert body["status"] == "succeeded" and body["attempts"] == 2


def test_duplicate_delivery_is_noop(client):
    sid = client.post("/v1/demo/arcore").json()["id"]
    first = wait(client, sid)
    again = process_job(sid, client.store, client.dispatch.artifacts)
    assert again.attempts == first["attempts"]


def test_cors_preflight(client):
    r = client.options("/v1/scans", headers={"Origin": "https://example.netlify.app", "Access-Control-Request-Method": "POST",
                                             "Access-Control-Request-Headers": "idempotency-key"})
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] in ("*", "https://example.netlify.app")
