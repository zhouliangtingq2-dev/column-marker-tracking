"""Camera-motion estimation from fixed points (e.g. corners on the stub, which does not move).

Each fixed point keeps the template cut from frame 0 (never updated, so it cannot drift). Per frame the
points are found by ZNCC, and a transform A_k (frame-0 image coords -> frame-k image coords) is fitted:
  translation : median displacement of the points (1 point is enough)
  rigid       : least-squares rotation + translation (>= 2 points)
  similarity  : least-squares rotation + uniform scale + translation (>= 2 points; use it when the camera
                may move along its axis, the scale estimate adds a little noise to dv)
Marker coordinates are later mapped back with A_k^-1, so the images themselves are never resampled.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .csvio import read_table, write_table
from .frames import FrameSource, preprocess
from .zncc import extract_patch, match_template, odd


def apply_affine(A: np.ndarray, xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    return xy @ A[:, :2].T + A[:, 2]


def invert_affine(A: np.ndarray) -> np.ndarray:
    M = np.linalg.inv(A[:, :2])
    return np.hstack([M, -M @ A[:, 2:3]])


def fit_translation(P0: np.ndarray, P1: np.ndarray) -> np.ndarray:
    t = np.median(P1 - P0, axis=0)
    return np.array([[1.0, 0.0, t[0]], [0.0, 1.0, t[1]]])


def _fit_2d(P0: np.ndarray, P1: np.ndarray, with_scale: bool) -> np.ndarray:
    """Least-squares R (and s), t with P1 ~ s R P0 + t; closed form in 2-D, no reflection."""
    m0, m1 = P0.mean(axis=0), P1.mean(axis=0)
    X, Y = P0 - m0, P1 - m1
    sxx = float((X ** 2).sum())
    if sxx < 1e-12:
        return fit_translation(P0, P1)
    c = float((X[:, 0] * Y[:, 0] + X[:, 1] * Y[:, 1]).sum())
    s = float((X[:, 0] * Y[:, 1] - X[:, 1] * Y[:, 0]).sum())
    if with_scale:
        a, b = c / sxx, s / sxx
    else:
        ang = np.arctan2(s, c)
        a, b = np.cos(ang), np.sin(ang)
    M = np.array([[a, -b], [b, a]])
    return np.hstack([M, (m1 - M @ m0)[:, None]])


def fit_rigid(P0: np.ndarray, P1: np.ndarray) -> np.ndarray:
    """Rotation + translation (scale fixed to 1)."""
    return _fit_2d(P0, P1, with_scale=False)


def fit_similarity(P0: np.ndarray, P1: np.ndarray) -> np.ndarray:
    """Rotation + uniform scale + translation."""
    return _fit_2d(P0, P1, with_scale=True)


FITS = {"translation": (fit_translation, 1), "rigid": (fit_rigid, 2), "similarity": (fit_similarity, 2)}


def fit_robust(P0: np.ndarray, P1: np.ndarray, model: str, max_resid: float):
    """Fit, then drop the worst point while its residual exceeds ``max_resid``.

    Points are only dropped while at least one redundant point remains (2 for translation, 3 otherwise).
    """
    fit, need = FITS[model]
    keep = np.ones(len(P0), bool)
    while True:
        A = fit(P0[keep], P1[keep])
        res = np.linalg.norm(apply_affine(A, P0) - P1, axis=1)
        worst = int(np.argmax(np.where(keep, res, -1)))
        if res[worst] <= max_resid or keep.sum() <= need + 1:
            return A, keep, float(np.sqrt(np.mean(res[keep] ** 2)))
        keep[worst] = False


@dataclass
class CameraMotion:
    frame: np.ndarray      # (T,) source frame index
    t_s: np.ndarray        # (T,)
    A: np.ndarray          # (T, 2, 3) frame-0 image coords -> frame-k image coords
    n_used: np.ndarray     # (T,) number of fixed points used (0 = no estimate)
    resid_px: np.ndarray   # (T,) RMS residual of the used points
    min_corr: np.ndarray   # (T,)
    ref_xy: np.ndarray     # (2,) point where dx/dy are reported (centroid of the fixed points)

    @classmethod
    def identity(cls, frame, t_s, ref_xy=(0.0, 0.0)):
        T = len(frame)
        A = np.tile(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), (T, 1, 1))
        return cls(np.asarray(frame), np.asarray(t_s, float), A, np.full(T, -1), np.zeros(T), np.ones(T), np.asarray(ref_xy, float))

    @property
    def valid(self) -> np.ndarray:
        return self.n_used != 0

    def to_frame0(self, k: int, xy) -> np.ndarray:
        return apply_affine(invert_affine(self.A[k]), xy)

    def from_frame0(self, k: int, xy) -> np.ndarray:
        return apply_affine(self.A[k], xy)

    def params(self) -> dict:
        a, b = self.A[:, 0, 0], self.A[:, 1, 0]
        d = np.einsum("tij,j->ti", self.A[:, :, :2], self.ref_xy) + self.A[:, :, 2] - self.ref_xy
        return {"dx_px": d[:, 0], "dy_px": d[:, 1], "rot_deg": np.degrees(np.arctan2(b, a)), "scale": np.hypot(a, b)}

    def save_csv(self, path) -> None:
        p = self.params()
        write_table(path, {
            "frame": self.frame, "t_s": self.t_s,
            "a": self.A[:, 0, 0], "b": self.A[:, 1, 0], "tx": self.A[:, 0, 2], "ty": self.A[:, 1, 2],
            "dx_px": p["dx_px"], "dy_px": p["dy_px"], "rot_deg": p["rot_deg"], "scale": p["scale"],
            "n_used": self.n_used, "resid_px": self.resid_px, "min_corr": self.min_corr,
            "ref_x": np.full(len(self.frame), self.ref_xy[0]), "ref_y": np.full(len(self.frame), self.ref_xy[1]),
        })

    @classmethod
    def load_csv(cls, path) -> "CameraMotion":
        d = read_table(path)
        a, b = d["a"], d["b"]
        A = np.stack([np.stack([a, -b, d["tx"]], axis=1), np.stack([b, a, d["ty"]], axis=1)], axis=1)
        return cls(d["frame"].astype(int), d["t_s"], A, d["n_used"].astype(int), d["resid_px"], d["min_corr"],
                   np.array([d["ref_x"][0], d["ref_y"][0]]))


def estimate_camera_motion(source: FrameSource, cfg: dict, log=print):
    """Track the fixed points of ``cfg['camera']`` and fit A_k for every frame.

    Returns (CameraMotion, xy (T, N, 2) fixed-point positions, corr (T, N)).
    """
    cam = cfg["camera"]
    P0 = np.asarray(cam["points_px"], dtype=float).reshape(-1, 2)
    need = FITS[cam["model"]][1]
    if len(P0) < need:
        raise ValueError(f"camera.model={cam['model']} needs at least {need} fixed point(s) in camera.points_px")
    T, N = len(source), len(P0)
    tsz, search = odd(cam["template_px"]), float(cam["search_px"])
    xy = np.full((T, N, 2), np.nan)
    corr = np.full((T, N), np.nan)
    A = np.tile(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]), (T, 1, 1))
    n_used = np.zeros(T, int)
    resid = np.full(T, np.nan)
    min_corr = np.full(T, np.nan)
    templates = []
    A_last = A[0].copy()
    blur = cfg["preprocess"]["blur_sigma"]
    for k, img in source:
        g = preprocess(img, blur)
        if k == 0:
            templates = [extract_patch(g, p, tsz) for p in P0]
            xy[0], corr[0], n_used[0], resid[0], min_corr[0] = P0, 1.0, N, 0.0, 1.0
            continue
        pred = apply_affine(A_last, P0)
        for i in range(N):
            m = match_template(g, templates[i], pred[i], search)
            if not (m.ok and m.corr >= cam["min_corr"]):
                m2 = match_template(g, templates[i], pred[i], search * cam["rescue_scale"])
                if m2.ok and (not m.ok or m2.corr > m.corr):
                    m = m2
            corr[k, i] = m.corr
            if m.ok and m.corr >= cam["min_corr"]:
                xy[k, i] = (m.x, m.y)
        good = np.isfinite(xy[k, :, 0])
        if good.sum() >= need:
            Ak, keep, r = fit_robust(P0[good], xy[k, good], cam["model"], cam["max_resid_px"])
            A[k], A_last = Ak, Ak
            n_used[k], resid[k] = int(keep.sum()), r
            min_corr[k] = float(np.nanmin(corr[k, good]))
        else:
            A[k] = A_last              # carried over; frame is flagged by n_used = 0
            log(f"[camera] frame {source.indices[k]}: only {good.sum()} fixed point(s) found")
        if k % max(1, T // 10) == 0:
            log(f"[camera] {k}/{T - 1}")
    ref = P0.mean(axis=0)
    return CameraMotion(source.frame_index, source.times, A, n_used, resid, min_corr, ref), xy, corr
