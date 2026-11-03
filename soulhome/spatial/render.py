"""SVG floor plan of a SpatialModel (walls, openings, objects, Vastu grid, north arrow)."""

from __future__ import annotations

import math
from html import escape

import numpy as np

from .zones import GRID, to_east_north

ELEMENT_COLOURS = {"fire": "#e4572e", "water": "#2e86de", "earth": "#a0743b", "air": "#17a589", "space": "#8e44ad"}


def render_svg(model, px_per_m: float = 80, margin_m: float = 1.0) -> str:
    rooms = model.rooms
    allpts = np.concatenate([np.asarray(r.floor_polygon) for r in rooms])
    lo = allpts.min(axis=0) - margin_m
    hi = allpts.max(axis=0) + margin_m
    w, h = (hi - lo) * px_per_m

    def P(x, y):  # model (x right, y up) -> svg (y down)
        return (x - lo[0]) * px_per_m, (hi[1] - y) * px_per_m

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}" width="{w:.0f}" height="{h:.0f}" '
        'font-family="system-ui,sans-serif" font-size="12">',
        "<style>text{paint-order:stroke;stroke:#fbfaf7;stroke-width:3px;stroke-linejoin:round}</style>",
        f'<rect width="{w:.0f}" height="{h:.0f}" fill="#fbfaf7"/>',
    ]
    for room in rooms:
        poly = np.asarray(room.floor_polygon)
        pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in (P(*p) for p in poly))
        out.append(f'<polygon points="{pts}" fill="#f1ece2" stroke="none"/>')
        if model.north_deg is not None:
            # 3x3 Vastu grid over the north-aligned bounding box
            en = to_east_north(poly, model.north_deg)
            (e0, n0), (e1, n1) = en.min(axis=0), en.max(axis=0)
            n = math.radians(model.north_deg)
            inv = np.array([[math.cos(n), -math.sin(n)], [math.sin(n), math.cos(n)]])  # EN -> xy
            for k in (1, 2):
                for a, b in (
                    ((e0 + (e1 - e0) * k / 3, n0), (e0 + (e1 - e0) * k / 3, n1)),
                    ((e0, n0 + (n1 - n0) * k / 3), (e1, n0 + (n1 - n0) * k / 3)),
                ):
                    (x0, y0), (x1, y1) = P(*(inv @ a)), P(*(inv @ b))
                    out.append(
                        f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="#c9b99a" stroke-dasharray="4 4"/>'
                    )
            for r_i in range(3):
                for c_i in range(3):
                    e = e0 + (e1 - e0) * (c_i + 0.5) / 3
                    nn = n1 - (n1 - n0) * (r_i + 0.5) / 3
                    x, y = P(*(inv @ (e, nn)))
                    label = "Brahmasthan" if GRID[r_i][c_i] == "BRAHMASTHAN" else GRID[r_i][c_i]
                    out.append(f'<text x="{x:.1f}" y="{y:.1f}" fill="#b3a07a" text-anchor="middle">{label}</text>')
        for wall in room.walls:
            (x0, y0), (x1, y1) = P(*wall.start), P(*wall.end)
            out.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="#2d2a26" stroke-width="6" stroke-linecap="square"/>')
            # length label just outside the wall (CCW polygon: outward normal is to the right)
            tx, ty = x1 - x0, y1 - y0
            n = max((tx * tx + ty * ty) ** 0.5, 1e-9)
            ox, oy = -ty / n * 16, tx / n * 16  # svg y is flipped, so "right" becomes this
            mx, my = (x0 + x1) / 2 + ox, (y0 + y1) / 2 + oy + 4
            out.append(f'<text x="{mx:.1f}" y="{my:.1f}" text-anchor="middle" fill="#6b645a">{wall.length_m:.2f} m</text>')
        walls = {wl.id: wl for wl in room.walls}
        for op in room.openings:
            wl = walls[op.wall_id]
            a, b = np.asarray(wl.start), np.asarray(wl.end)
            t = (b - a) / wl.length_m
            s0, s1 = a + t * op.offset_m, a + t * (op.offset_m + op.width_m)
            (x0, y0), (x1, y1) = P(*s0), P(*s1)
            colour = "#d35400" if op.type == "door" else "#3498db"
            out.append(f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="{colour}" stroke-width="8"/>')
        for o in room.objects:
            x, y = P(o.position[0], o.position[1])
            colour = ELEMENT_COLOURS.get(o.element or "", "#555")
            out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{colour}"/>')
            zone = f" · {o.direction_16}" if o.direction_16 else ""
            out.append(f'<text x="{x + 10:.1f}" y="{y + 4:.1f}" fill="#2d2a26">{escape(o.label)}{zone}</text>')
        out.append(
            f'<text x="12" y="22" fill="#2d2a26" font-weight="600" font-size="14">'
            f"{escape(room.name)} · {room.floor_area_m2:.1f} m² · ceiling {room.ceiling_height_m:.2f} m</text>"
        )
    if model.north_deg is not None:
        # arrow pointing to true north: bearing 0 -> model-frame angle -north_deg from +Y
        n = math.radians(model.north_deg)
        dx, dy = -math.sin(n), math.cos(n)
        bx, by = w - 40, 50
        out.append(
            f'<line x1="{bx:.1f}" y1="{by:.1f}" x2="{bx + dx * 28:.1f}" y2="{by - dy * 28:.1f}" stroke="#c0392b" stroke-width="3"/>'
            f'<text x="{bx + dx * 40:.1f}" y="{by - dy * 40 + 4:.1f}" text-anchor="middle" fill="#c0392b" font-weight="700">N</text>'
        )
    out.append("</svg>")
    return "".join(out)
