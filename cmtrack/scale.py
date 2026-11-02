"""px -> mm on the column face (X to the right, Y upwards, origin at the lower-left corner LB).

Modes
  rectangle  : separate x / y scales from the four corners of a rectangle of known size
               (mean of the two widths and the two heights), axes taken along the image axes
  affine     : least-squares affine map from the four corners (also absorbs a slight camera roll)
  homography : perspective map from the four corners (camera not perpendicular to the face)
  spacing    : no corners needed; scales from the known marker pitch and the frame-0 marker positions,
               origin at marker c1r01
All corner coordinates are frame-0 image coordinates, i.e. the same frame the camera-corrected marker
coordinates refer to.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .layout import Grid

CORNER_KEYS = ("LB", "LT", "RB", "RT")


@dataclass
class PixelScale:
    mode: str
    M: np.ndarray            # 3 x 3 map from homogeneous px to homogeneous mm
    mm_per_px_x: float
    mm_per_px_y: float

    def to_mm(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        shp = xy.shape
        p = xy.reshape(-1, 2)
        h = np.hstack([p, np.ones((len(p), 1))]) @ self.M.T
        out = h[:, :2] / h[:, 2:3]
        return out.reshape(shp)

    def describe(self) -> dict:
        return {"mode": self.mode, "mm_per_px_x": self.mm_per_px_x, "mm_per_px_y": self.mm_per_px_y,
                "matrix_px_to_mm": self.M.tolist()}


def _corners(corners: dict) -> np.ndarray:
    try:
        return np.array([corners[k] for k in CORNER_KEYS], dtype=float)
    except (KeyError, TypeError):
        raise ValueError("scale.corners_px needs LB, LT, RB, RT (frame-0 px)") from None


def from_rectangle(corners: dict, width_mm: float, height_mm: float, mode: str = "rectangle") -> PixelScale:
    LB, LT, RB, RT = _corners(corners)
    w_px = 0.5 * (np.linalg.norm(RB - LB) + np.linalg.norm(RT - LT))
    h_px = 0.5 * (np.linalg.norm(LT - LB) + np.linalg.norm(RT - RB))
    sx, sy = width_mm / w_px, height_mm / h_px
    if mode == "rectangle":
        M = np.array([[sx, 0.0, -sx * LB[0]], [0.0, -sy, sy * LB[1]], [0.0, 0.0, 1.0]])
    else:
        src = np.array([LB, LT, RB, RT], dtype=np.float64)
        dst = np.array([[0.0, 0.0], [0.0, height_mm], [width_mm, 0.0], [width_mm, height_mm]])
        if mode == "affine":
            G = np.hstack([src, np.ones((4, 1))])
            sol, *_ = np.linalg.lstsq(G, dst, rcond=None)      # (3, 2)
            M = np.vstack([sol.T, [0.0, 0.0, 1.0]])
        elif mode == "homography":
            M = homography(src, dst)
        else:
            raise ValueError(mode)
    return PixelScale(mode, M, float(sx), float(sy))


def homography(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """3 x 3 perspective map from 4 point pairs (direct linear solution in float64)."""
    A, b = [], []
    for (x, y), (u, v) in zip(src, dst):
        A.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        b.append(u)
        A.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        b.append(v)
    h = np.linalg.solve(np.array(A, dtype=float), np.array(b, dtype=float))
    return np.append(h, 1.0).reshape(3, 3)


def from_spacing(xy0: np.ndarray, grid: Grid, spacing_mm) -> PixelScale:
    dx_mm, dy_mm = (float(s) for s in spacing_mm)
    dx = np.mean([np.linalg.norm(xy0[j] - xy0[i]) for i, j in grid.horizontal_pairs()])
    dy = np.mean([np.linalg.norm(xy0[j] - xy0[i]) for i, j in grid.vertical_pairs()])
    sx, sy = dx_mm / dx, dy_mm / dy
    o = xy0[grid.idx(1, 1)]
    M = np.array([[sx, 0.0, -sx * o[0]], [0.0, -sy, sy * o[1]], [0.0, 0.0, 1.0]])
    return PixelScale("spacing", M, float(sx), float(sy))


def from_config(cfg: dict, xy0: np.ndarray, grid: Grid) -> PixelScale:
    s = cfg["scale"]
    if s["mode"] == "spacing":
        return from_spacing(xy0, grid, s["spacing_mm"])
    return from_rectangle(s["corners_px"], float(s["width_mm"]), float(s["height_mm"]), s["mode"])
