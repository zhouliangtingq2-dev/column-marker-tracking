"""ZNCC template matching with sub-pixel peak refinement (quadratic surface through the 3 x 3 peak).

Conventions: pixel centres are at integer coordinates (x = column, y = row). A template of odd size
L = 2r + 1 cut around (cx, cy) with ``extract_patch`` is matched back at exactly (cx, cy).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

NAN = float("nan")


@dataclass
class Match:
    x: float
    y: float
    corr: float
    at_edge: bool = False   # peak on the border of the search window -> the true peak may lie outside

    @property
    def ok(self) -> bool:
        return bool(np.isfinite(self.corr)) and not self.at_edge


def odd(n) -> int:
    n = int(round(n))
    return max(3, n if n % 2 == 1 else n + 1)


def extract_patch(img: np.ndarray, center, size: int) -> np.ndarray:
    """size x size float32 patch centred on a sub-pixel position (bilinear; borders replicated)."""
    size = odd(size)
    return cv2.getRectSubPix(np.ascontiguousarray(img, dtype=np.float32), (size, size), (float(center[0]), float(center[1])))


def znorm(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    s = float(a.std())
    return (a - a.mean()) / s if s > 1e-9 else np.zeros_like(a)


def zncc_map(search: np.ndarray, template: np.ndarray) -> np.ndarray:
    r = cv2.matchTemplate(np.ascontiguousarray(search, dtype=np.float32), np.ascontiguousarray(template, dtype=np.float32),
                          cv2.TM_CCOEFF_NORMED)
    return np.nan_to_num(r, nan=-1.0, posinf=-1.0, neginf=-1.0)


def parabolic_offset(cm: float, c0: float, cp: float) -> float:
    """Vertex of the parabola through (-1, cm), (0, c0), (+1, cp), clipped to +-0.5."""
    den = cm - 2.0 * c0 + cp
    if den >= -1e-12:          # not a strict maximum
        return 0.0
    return float(np.clip(0.5 * (cm - cp) / den, -0.5, 0.5))


def subpixel_peak(R: np.ndarray, iy: int, ix: int):
    """(dy, dx) sub-pixel offsets of the correlation peak at integer (iy, ix).

    Vertex of the quadratic surface through the 3 x 3 neighbourhood, including the xy term
    (Newton step with central differences). Separate 1-D parabolas in x and y ignore that term and are
    biased by up to ~0.2 px when the peak is elongated obliquely (textures, corners); they are only used
    as a fallback (border of the map, or no proper maximum).
    """
    h, w = R.shape
    if 0 < ix < w - 1 and 0 < iy < h - 1:
        f = R[iy - 1:iy + 2, ix - 1:ix + 2].astype(np.float64)
        fx, fy = 0.5 * (f[1, 2] - f[1, 0]), 0.5 * (f[2, 1] - f[0, 1])
        fxx, fyy = f[1, 2] - 2 * f[1, 1] + f[1, 0], f[2, 1] - 2 * f[1, 1] + f[0, 1]
        fxy = 0.25 * (f[2, 2] - f[0, 2] - f[2, 0] + f[0, 0])
        det = fxx * fyy - fxy * fxy
        if fxx < 0 and det > 1e-12:
            dx = -(fyy * fx - fxy * fy) / det
            dy = -(fxx * fy - fxy * fx) / det
            if abs(dx) <= 1.0 and abs(dy) <= 1.0:
                return float(dy), float(dx)
    dx = parabolic_offset(R[iy, ix - 1], R[iy, ix], R[iy, ix + 1]) if 0 < ix < w - 1 else 0.0
    dy = parabolic_offset(R[iy - 1, ix], R[iy, ix], R[iy + 1, ix]) if 0 < iy < h - 1 else 0.0
    return dy, dx


def match_template(img: np.ndarray, template: np.ndarray, center, half_window: float) -> Match:
    """Search ``template`` within +-half_window px of ``center``; returns the sub-pixel centre and peak ZNCC."""
    if center is None or not np.all(np.isfinite(center)):
        return Match(NAN, NAN, NAN)
    th, tw = template.shape
    ry, rx = th // 2, tw // 2
    cx, cy = int(round(float(center[0]))), int(round(float(center[1])))
    hw = max(1, int(round(half_window)))
    x1, x2 = max(0, cx - hw - rx), min(img.shape[1], cx + hw + rx + 1)
    y1, y2 = max(0, cy - hw - ry), min(img.shape[0], cy + hw + ry + 1)
    if x2 - x1 < tw + 2 or y2 - y1 < th + 2:       # need at least a 3 x 3 correlation map
        return Match(NAN, NAN, NAN)
    R = zncc_map(img[y1:y2, x1:x2], template)
    iy, ix = np.unravel_index(int(np.argmax(R)), R.shape)
    dy, dx = subpixel_peak(R, iy, ix)
    at_edge = ix == 0 or iy == 0 or ix == R.shape[1] - 1 or iy == R.shape[0] - 1
    return Match(x1 + ix + dx + rx, y1 + iy + dy + ry, float(R[iy, ix]), bool(at_edge))
