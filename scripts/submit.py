"""Upload a scan to a deployed API and wait for the result.

    python scripts/submit.py https://<workspace>--soulhome-api.modal.run panorama room.jpg --north 30 --height 1.5
    python scripts/submit.py https://<workspace>--soulhome-api.modal.run lidar scan.usdz
"""

import argparse
import json
import sys
import time
import uuid
from pathlib import Path

import httpx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_url")
    ap.add_argument("tier", choices=["panorama", "lidar", "arcore"])
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--north", type=float, help="compass bearing of the capture's forward axis")
    ap.add_argument("--height", type=float, help="camera height in metres (panorama tier)")
    ap.add_argument("--name", default="room")
    a = ap.parse_args()
    meta = {"room_name": a.name}
    if a.north is not None:
        meta["north_deg"] = a.north
    if a.height is not None:
        meta["camera_height_m"] = a.height
    with httpx.Client(base_url=a.base_url.rstrip("/"), timeout=120) as c:
        files = [("files", (p.name, p.read_bytes())) for p in a.files]
        r = c.post(
            "/v1/scans",
            data={"tier": a.tier, "metadata": json.dumps(meta)},
            files=files,
            headers={"Idempotency-Key": str(uuid.uuid4())},
        )
        r.raise_for_status()
        job = r.json()
        print("scan", job["id"])
        while job["status"] not in ("succeeded", "failed"):
            time.sleep(job.get("poll_after_s", 2))
            job = c.get(f"/v1/scans/{job['id']}").json()
            print(" ", job["status"], job.get("current_stage") or "")
        if job["status"] == "failed":
            print(json.dumps(job["error"], indent=2))
            sys.exit(1)
        print(json.dumps(job["summary"], indent=2))
        print("floor plan:", job["links"]["floorplan"])


if __name__ == "__main__":
    main()
