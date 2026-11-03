"""GPU inference: room layout (HorizonNet), monocular depth (Depth Anything V2 metric-indoor) and
object detection (YOLO11).

Heavy imports are lazy so the CPU-only geometry pipeline and its tests never pull in torch. On Modal
this class lives inside a GPU container (see ``modal_app.py``) and is loaded once per container.
"""

from __future__ import annotations

import os
import time
from typing import Protocol

import numpy as np

from ..geometry.equirect import (
    equirect_to_perspective,
    lonlat_to_pixel,
    perspective_camera_rays,
    vec_to_lonlat,
    yaw_pitch_rotation,
)

DEPTH_MODEL = os.environ.get("SOULHOME_DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf")
YOLO_WEIGHTS = os.environ.get("SOULHOME_YOLO_WEIGHTS", "yolo11s.pt")
LAYOUT_VARIANT = os.environ.get("SOULHOME_LAYOUT_VARIANT", "mp3d")  # "none" disables the layout net

# depth views (yaw, pitch), 90 deg FoV. Straight up/down cube faces are out of distribution for
# monocular depth models (they read a floor 1.5 m away as ~4 m), so we use two overlapping pitched
# rings instead; the fusion step reconciles their scales against the layout.
DEPTH_VIEWS = [(y, p) for p in (-35, 25) for y in range(0, 360, 45)]
# detection tiles: horizontal ring with overlap, slightly pitched down where furniture lives
DET_TILES = [(y, -15) for y in range(0, 360, 60)]


class InferenceBackend(Protocol):
    models: dict[str, str]

    def depth_points(self, rgb: np.ndarray) -> np.ndarray: ...

    def detect(self, rgb: np.ndarray) -> list[dict]: ...

    def layout(self, rgb: np.ndarray): ...


class TorchInference:
    def __init__(self, device: str | None = None, face_size: int = 518, load_detector: bool = True):
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.face_size = face_size
        self.models: dict[str, str] = {}
        t0 = time.time()
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL)
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL, dtype=self.dtype).to(self.device).eval()
        self.models["depth"] = DEPTH_MODEL
        self.detector = None
        if load_detector:
            from ultralytics import YOLO

            self.detector = YOLO(YOLO_WEIGHTS)
            self.models["detector"] = os.path.basename(YOLO_WEIGHTS)
        self.layout_net = None
        if LAYOUT_VARIANT != "none":
            from ..layout.horizonnet_model import CHECKPOINTS, HF_REPO, load_horizonnet

            self.layout_net = load_horizonnet(LAYOUT_VARIANT, self.device)
            self.models["layout"] = f"{HF_REPO}/{CHECKPOINTS[LAYOUT_VARIANT]}"
        self.load_seconds = time.time() - t0

    # ------------------------------------------------------------------ depth
    def depth_points(self, rgb: np.ndarray, stride: int = 2) -> np.ndarray:
        """Equirect RGB -> point cloud (N,3) around the camera from overlapping perspective views
        batched through Depth Anything V2 (metric indoor head)."""
        torch = self.torch
        s = self.face_size
        faces = [equirect_to_perspective(rgb, y, p, 90.0, s) for y, p in DEPTH_VIEWS]
        inputs = self.processor(images=faces, return_tensors="pt")
        with torch.inference_mode():
            out = self.depth_model(pixel_values=inputs["pixel_values"].to(self.device, self.dtype)).predicted_depth
            out = torch.nn.functional.interpolate(out[:, None].float(), size=(s, s), mode="bilinear", align_corners=False)[:, 0]
        depths = out.cpu().numpy()
        rays = perspective_camera_rays(s, 90.0)[::stride, ::stride]
        pts = []
        for (y, p), d in zip(DEPTH_VIEWS, depths):
            sub = d[::stride, ::stride]
            pc = (rays * sub[..., None]).reshape(-1, 3) @ yaw_pitch_rotation(y, p).T
            pts.append(pc[np.isfinite(sub.reshape(-1)) & (sub.reshape(-1) > 0)])
        return np.concatenate(pts)

    # -------------------------------------------------------------- detection
    def detect(self, rgb: np.ndarray, conf: float = 0.35, tile: int = 640) -> list[dict]:
        """YOLO11 on overlapping perspective tiles; boxes are mapped back to equirect pixels by
        projecting each tile box's outline through the tile camera."""
        if self.detector is None:
            return []
        h, w = rgb.shape[:2]
        tiles = [equirect_to_perspective(rgb, y, p, 90.0, tile) for y, p in DET_TILES]
        results = self.detector.predict(tiles, conf=conf, verbose=False)
        rays = perspective_camera_rays(tile, 90.0)
        dets = []
        for (yaw, pitch), res in zip(DET_TILES, results):
            names = res.names
            for box, c, k in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.cls.cpu().numpy()):
                x0, y0, x1, y1 = np.clip(box, 0, tile - 1)
                xs = np.linspace(x0, x1, 9)
                ys = np.linspace(y0, y1, 9)
                outline = np.concatenate(
                    [np.c_[xs, np.full(9, y0)], np.c_[xs, np.full(9, y1)], np.c_[np.full(9, x0), ys], np.c_[np.full(9, x1), ys]]
                ).astype(int)
                v = rays[outline[:, 1], outline[:, 0]] @ yaw_pitch_rotation(yaw, pitch).T
                lon, lat = vec_to_lonlat(v)
                lon = np.unwrap(lon - np.radians(yaw)) + np.radians(yaw)  # contiguous across the seam
                u = (lon + np.pi) / (2 * np.pi) * w - 0.5
                _, vv = lonlat_to_pixel(lon, lat, w, h)
                # u1 < u0 signals a box that wraps across the seam (handled by placement / NMS)
                u0, u1 = float(u.min()) % w, float(u.max()) % w
                dets.append({"label": names[int(k)], "confidence": float(c), "bbox": [u0, float(vv.min()), u1, float(vv.max())]})
        from ..objects.placement import nms_equirect

        return nms_equirect(dets, w)

    # ----------------------------------------------------------------- layout
    def layout(self, rgb: np.ndarray):
        """HorizonNet -> (ceil_lat, floor_lat, corner_prob) per column, or None if disabled."""
        if self.layout_net is None:
            return None
        from ..layout.horizonnet import from_horizonnet_raw
        from ..layout.horizonnet_model import predict

        y_bon, y_cor = predict(self.layout_net, rgb)
        return from_horizonnet_raw(y_bon, y_cor)
