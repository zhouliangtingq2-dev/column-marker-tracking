"""Deformation quantities from marker coordinates in mm (face coordinates: X right, Y up).

theta : member angle of a marker column = atan of the least-squares slope dX/dY of its markers,
        reported as the change from frame 0 (rad, positive when the top moves to +X)
dv    : axial deformation = change of the vertical distance between the mean of the top rows and the
        mean of the bottom rows of one marker column (mm, elongation positive by default)
eps_v : dv / initial gauge length
spacing_x / spacing_y : change of the X distance of horizontally adjacent markers and of the
        Y distance of vertically adjacent markers (mm)
"""
from __future__ import annotations

import numpy as np

from .layout import Grid


def fit_slope(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Per-frame least-squares slope dX/dY of the points (Y_i, X_i); X, Y: (T, n). NaN points are ignored."""
    w = np.isfinite(X) & np.isfinite(Y)
    n = w.sum(axis=1)
    Xz, Yz = np.where(w, X, 0.0), np.where(w, Y, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        xm = Xz.sum(axis=1) / n
        ym = Yz.sum(axis=1) / n
        dy = np.where(w, Y - ym[:, None], 0.0)
        dx = np.where(w, X - xm[:, None], 0.0)
        s = (dx * dy).sum(axis=1) / (dy * dy).sum(axis=1)
    s[n < 2] = np.nan
    return s


def theta_column(XY: np.ndarray, grid: Grid, c: int) -> np.ndarray:
    ids = grid.column(c)
    th = np.arctan(fit_slope(XY[:, ids, 0], XY[:, ids, 1]))
    return th - th[0]


def gauge_length(XY: np.ndarray, grid: Grid, c: int, top_rows, bottom_rows) -> np.ndarray:
    top = [grid.idx(c, r) for r in top_rows]
    bot = [grid.idx(c, r) for r in bottom_rows]
    return XY[:, top, 1].mean(axis=1) - XY[:, bot, 1].mean(axis=1)


def spacing(XY: np.ndarray, grid: Grid):
    """Changes of adjacent-marker distances: (dX of horizontal pairs, dY of vertical pairs, names...)."""
    names = grid.names()
    hp, vp = grid.horizontal_pairs(), grid.vertical_pairs()
    sx = np.stack([XY[:, j, 0] - XY[:, i, 0] for i, j in hp], axis=1)
    sy = np.stack([XY[:, j, 1] - XY[:, i, 1] for i, j in vp], axis=1)
    return sx - sx[0], sy - sy[0], [f"{names[i]}-{names[j]}" for i, j in hp], [f"{names[i]}-{names[j]}" for i, j in vp]


def compute(XY: np.ndarray, grid: Grid, cfg: dict) -> dict:
    d = cfg["deformation"]
    th_cols = np.stack([theta_column(XY, grid, c) for c in range(1, grid.cols + 1)], axis=1)
    if str(d["theta_column"]).lower() == "mean":
        theta = th_cols.mean(axis=1)
    else:
        theta = th_cols[:, int(d["theta_column"]) - 1]
    L = gauge_length(XY, grid, int(d["dv_column"]), d["dv_top_rows"], d["dv_bottom_rows"])
    dv = L - L[0]
    if d["dv_sign"] == "shortening":
        dv = -dv
    sx, sy, nx, ny = spacing(XY, grid)
    return {"theta": theta, "theta_cols": th_cols, "dv": dv, "eps_v": dv / L[0], "gauge_mm": float(L[0]),
            "spacing_x": sx, "spacing_y": sy, "spacing_x_names": nx, "spacing_y_names": ny}
