"""Normalise panorama uploads onto a 2:1 equirectangular canvas.

Two kinds of panorama reach the pipeline:

* **360° spheres** (Photo Sphere, Insta360/Theta, guided 360 capture): already equirectangular,
  2:1. Used as-is.
* **Phone "Pano" mode** (iPhone Pano, most Android panorama modes): a *cylindrical* projection of a
  horizontal sweep. Rows are linear in tan(latitude), columns are linear in longitude, the sweep
  is often less than 360°, and the vertical field of view is that of the camera held in portrait
  (~60–70°). E.g. 10840×3784 from an iPhone ≈ 210° × 65°.

For the second kind we resample onto the equirect canvas so every downstream stage (HorizonNet,
depth views, detection tiles, back-projection) keeps using one camera model. Pixels the phone never
saw stay black and are reported as coverage, which drives warnings and the confidence score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .equirect import pixel_to_lonlat

DEFAULT_PHONE_VFOV_DEG = 65.0  # main camera, portrait orientation (typical for iPhone / Pixel Pano)


@dataclass
class PanoramaCoverage:
    projection: str  # "equirectangular" | "cylindrical"
    hfov_deg: float
    vfov_deg: float  # total vertical field of view covered (symmetric about the horizon)
    source_size: tuple[int, int]

    @property
    def full_circle(self) -> bool:
        return self.hfov_deg >= 350

    def as_dict(self) -> dict:
        return {
            "projection": self.projection,
            "hfov_deg": round(self.hfov_deg, 1),
            "vfov_deg": round(self.vfov_deg, 1),
            "source_size": list(self.source_size),
        }


def to_equirect(
    img: np.ndarray,
    out_width: int = 2048,
    projection: str = "auto",
    vfov_deg: float | None = None,
    hfov_deg: float | None = None,
) -> tuple[np.ndarray, PanoramaCoverage]:
    """Return (equirect image (out_width/2, out_width, 3), coverage).

    ``projection``: "auto" (2:1 -> equirectangular, wider -> cylindrical), "equirect" or
    "cylindrical". For cylindrical input the focal length comes from ``hfov_deg`` if the user gave
    the sweep angle, otherwise from ``vfov_deg`` (camera vertical field of view)."""
    import cv2

    h, w = img.shape[:2]
    aspect = w / h
    out_h = out_width // 2
    if projection == "auto":
        if abs(aspect - 2.0) <= 0.05:
            projection = "equirect"
        elif aspect > 2.05:
            projection = "cylindrical"
        else:
            raise ValueError(
                f"image is {w}x{h} (aspect {aspect:.2f}); a panorama must be at least twice as wide as it is tall"
            )

    if projection == "equirect":
        if abs(aspect - 2.0) <= 0.05:
            out = cv2.resize(img, (out_width, out_h), interpolation=cv2.INTER_AREA)
            return out, PanoramaCoverage("equirectangular", 360.0, 180.0, (w, h))
        # equirect with cropped poles: full 360° wide, centred on the horizon
        vfov = 360.0 * h / w
        scale = out_width / w
        body = cv2.resize(img, (out_width, max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
        out = np.zeros((out_h, out_width, 3), img.dtype)
        top = (out_h - body.shape[0]) // 2
        out[top : top + body.shape[0]] = body[: out_h - top]
        return out, PanoramaCoverage("equirectangular", 360.0, vfov, (w, h))

    # cylindrical: x = f * lon + w/2 ; y = h/2 - f * tan(lat)
    if hfov_deg:
        f = w / np.radians(min(hfov_deg, 360.0))
    else:
        vfov = vfov_deg or DEFAULT_PHONE_VFOV_DEG
        f = (h / 2) / np.tan(np.radians(vfov) / 2)
    hfov = w / f
    if hfov >= np.radians(340):  # a near-complete sweep: the user went all the way round
        hfov = 2 * np.pi
        f = w / hfov
    uu, vv = np.meshgrid(np.arange(out_width), np.arange(out_h))
    lon, lat = pixel_to_lonlat(uu, vv, out_width, out_h)
    map_x = f * lon + (w - 1) / 2
    map_y = (h - 1) / 2 - f * np.tan(np.clip(lat, -1.55, 1.55))
    inside = (np.abs(lon) <= hfov / 2) & (map_y >= 0) & (map_y <= h - 1)
    map_x = np.where(inside, map_x, -10).astype(np.float32)
    map_y = np.where(inside, map_y, -10).astype(np.float32)
    # downsample first so remap isn't aliasing a 10k-pixel-wide source
    pre = max(1.0, (w / (hfov / (2 * np.pi))) / out_width / 1.5)
    src = img
    if pre > 1.0:
        src = cv2.resize(img, (round(w / pre), round(h / pre)), interpolation=cv2.INTER_AREA)
        map_x = np.where(inside, map_x / pre, -10).astype(np.float32)
        map_y = np.where(inside, map_y / pre, -10).astype(np.float32)
    out = cv2.remap(src, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    vfov_cov = 2 * np.degrees(np.arctan((h / 2) / f))
    return out, PanoramaCoverage("cylindrical", float(np.degrees(hfov)), float(vfov_cov), (w, h))


def lens_vfov_from_exif(path) -> float | None:
    """Vertical FoV of a phone Pano from the lens' 35 mm-equivalent focal length.

    Pano mode is shot with the phone upright, so the frame's long side (36 mm equivalent) is vertical:
    vfov = 2 * atan(18 / f35). iPhone main lens (26 mm) -> ~69°, ultra-wide 0.5x (13 mm) -> ~108°.
    A small margin is taken off because Pano mode crops the frame while stitching."""
    try:
        from PIL import Image

        with Image.open(path) as im:
            exif = im.getexif().get_ifd(0x8769)
        f35 = exif.get(41989)  # FocalLengthIn35mmFilm
        if not f35:
            return None
        return float(np.degrees(2 * np.arctan(18.0 / float(f35)))) * 0.95
    except Exception:  # noqa: BLE001 - missing / malformed EXIF just means "unknown"
        return None
