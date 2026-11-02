"""Marker initialisation and frame-to-frame tracking.

Initialisation : approximate positions (4 corner markers -> bilinear grid, or all points) are refined by
                 ZNCC with the ideal quadrant template of each marker's checker pattern (A or B).
Tracking       : for every frame and marker
                 1. predict in camera-corrected (frame-0) coordinates: last position + gain * velocity,
                    then map into the current frame with the camera transform A_k
                 2. ZNCC with the marker's template inside a search window (smaller when the previous
                    match was good), sub-pixel parabolic peak
                 3. accept if ZNCC >= corr_accept and the step from the prediction is small
                 4. otherwise "rescue": the frame-0 (anchor) template in a window rescue_scale times larger;
                    accepted only if ZNCC >= corr_rescue, the checker pattern (A/B quadrant template) is
                    found at the match, and the vector to the vertical neighbour differs from frame 0 by
                    less than neighbor_tol x its length
                 5. otherwise the marker is lost in this frame (NaN, filled later)
                 Whole frames are dropped when many markers are lost, the mean ZNCC is low, the reference
                 row is lost, or the median step jumps (STEP_SPIKE).
"""
from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass

import numpy as np

from .camera import CameraMotion, apply_affine, invert_affine
from .csvio import read_table, unwide, wide, write_table
from .frames import FrameSource, preprocess
from .layout import Grid, grid_from_corners, patterns, quadrant_template
from .zncc import extract_patch, match_template, odd, znorm

OK, RESCUED, LOST, BAD_FRAME = 0, 1, 2, 3
STATUS_NAMES = {OK: "ok", RESCUED: "rescued", LOST: "lost", BAD_FRAME: "bad_frame"}


def grid_from_config(cfg: dict) -> Grid:
    return Grid(int(cfg["markers"]["cols"]), int(cfg["markers"]["rows"]))


def approximate_positions(cfg: dict, grid: Grid) -> np.ndarray:
    init = cfg["markers"]["init"]
    if init["mode"] == "points":
        pts = np.asarray(init["points_px"], dtype=float).reshape(-1, 2)
        if len(pts) != grid.n:
            raise ValueError(f"markers.init.points_px has {len(pts)} points, expected {grid.n}")
        return pts
    c = init["corners_px"]
    if not c:
        raise ValueError("markers.init.corners_px is not set (or use scripts/pick_points.py)")
    return grid_from_corners(grid, c["bottom_left"], c["top_left"], c["bottom_right"], c["top_right"])


def init_markers(g0: np.ndarray, grid: Grid, approx: np.ndarray, pats: list, cfg: dict):
    """Refine approximate positions on the (blurred) first frame. Returns (xy (N, 2), corr (N,))."""
    mk = cfg["markers"]
    tsz = odd(max(9, 0.7 * float(mk["marker_px"])))
    xy = np.full((grid.n, 2), np.nan)
    corr = np.full(grid.n, np.nan)
    for i in range(grid.n):
        m = match_template(g0, quadrant_template(tsz, pats[i]), approx[i], mk["init"]["search_px"])
        if m.ok:
            xy[i], corr[i] = (m.x, m.y), m.corr
    return xy, corr


def pattern_score(g: np.ndarray, xy, pattern: str, marker_px: float) -> float:
    tsz = odd(max(9, 0.7 * float(marker_px)))
    m = match_template(g, quadrant_template(tsz, pattern), xy, 2)
    return m.corr if np.isfinite(m.corr) else -1.0


@dataclass
class TrackResult:
    frame: np.ndarray       # (T,)
    t_s: np.ndarray         # (T,)
    xy: np.ndarray          # (T, N, 2) image coordinates in each frame (NaN = no position)
    corr: np.ndarray        # (T, N) ZNCC of the accepted / best match
    status: np.ndarray      # (T, N) OK / RESCUED / LOST / BAD_FRAME
    reason: list            # (T,) frame-level reason, '' if the frame is usable
    init_xy: np.ndarray     # (N, 2)
    init_corr: np.ndarray   # (N,)
    names: list

    def save(self, out_dir) -> None:
        write_table(out_dir / "markers_px.csv", {"frame": self.frame, "t_s": self.t_s, **wide(self.names, self.xy)})
        write_table(out_dir / "markers_corr.csv", {"frame": self.frame, "t_s": self.t_s, **wide(self.names, self.corr)})
        write_table(out_dir / "markers_status.csv", {"frame": self.frame, "t_s": self.t_s, **wide(self.names, self.status.astype(int))})
        c = np.where(self.status <= RESCUED, self.corr, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN rows of dropped frames
            mean_c, min_c = np.nanmean(c, axis=1), np.nanmin(c, axis=1)
        write_table(out_dir / "frames.csv", {
            "frame": self.frame, "t_s": self.t_s, "mean_corr": mean_c, "min_corr": min_c,
            "n_rescued": (self.status == RESCUED).sum(axis=1), "n_lost": (self.status == LOST).sum(axis=1),
            "reason": list(self.reason)})
        write_table(out_dir / "init_markers.csv", {"marker": self.names, "x_px": self.init_xy[:, 0], "y_px": self.init_xy[:, 1],
                                                   "init_corr": self.init_corr})

    @classmethod
    def load(cls, out_dir, grid: Grid) -> "TrackResult":
        names = grid.names()
        p = read_table(out_dir / "markers_px.csv")
        c = read_table(out_dir / "markers_corr.csv")
        s = read_table(out_dir / "markers_status.csv")
        f = read_table(out_dir / "frames.csv")
        i = read_table(out_dir / "init_markers.csv")
        reason = ["" if (isinstance(r, float) or r in ("nan", "")) else str(r) for r in f["reason"]]
        return cls(p["frame"].astype(int), p["t_s"], unwide(p, names), unwide(c, names, None),
                   unwide(s, names, None).astype(int), reason,
                   np.stack([i["x_px"], i["y_px"]], axis=1), i["init_corr"], names)


def track_markers(source: FrameSource, cfg: dict, camera: CameraMotion | None = None, log=print) -> TrackResult:
    grid = grid_from_config(cfg)
    N, T = grid.n, len(source)
    mk, tr = cfg["markers"], cfg["tracking"]
    pats = patterns(grid, mk["pattern_b_rows"])
    tsz, search = odd(mk["template_px"]), float(mk["search_px"])
    blur = cfg["preprocess"]["blur_sigma"]
    if camera is None:
        camera = CameraMotion.identity(source.frame_index, source.times)
    ref_ids = grid.row(int(cfg["deformation"]["reference_row"]))

    xy = np.full((T, N, 2), np.nan)
    corr = np.full((T, N), np.nan)
    status = np.full((T, N), LOST, dtype=np.int8)
    reason = [""] * T
    anchor, current = [], []
    q_last = np.zeros((N, 2))           # last accepted position, frame-0 coordinates
    last_k = np.zeros(N, int)
    vel = np.zeros((N, 2))
    nb = [grid.vertical_neighbor(i) for i in range(N)]
    v0 = np.zeros((N, 2))                # frame-0 vector to the vertical neighbour
    steps_hist = deque(maxlen=15)
    init_xy = init_corr = None

    for k, img in source:
        g = preprocess(img, blur)
        if k == 0:
            approx = approximate_positions(cfg, grid)
            init_xy, init_corr = init_markers(g, grid, approx, pats, cfg)
            bad = ~np.isfinite(init_corr) | (init_corr < mk["init"]["min_corr"])
            if bad.any():
                names = grid.names()
                raise RuntimeError("marker initialisation failed for " + ", ".join(names[i] for i in np.where(bad)[0])
                                   + " (check markers.init.* and pattern_b_rows)")
            anchor = [extract_patch(g, p, tsz) for p in init_xy]
            current = [znorm(a) for a in anchor]
            q_last[:] = camera.to_frame0(0, init_xy)
            xy[0], corr[0], status[0] = init_xy, 1.0, OK
            v0 = q_last[nb] - q_last
            log(f"[markers] initialised {N} markers, init ZNCC min {init_corr.min():.3f} / mean {init_corr.mean():.3f}")
            continue

        if not camera.valid[k]:
            status[k], reason[k] = BAD_FRAME, "CAMERA_LOST"
            continue
        A = camera.A[k]
        Ainv = invert_affine(A)
        q_new = np.full((N, 2), np.nan)
        for i in range(N):
            q_pred = q_last[i] + tr["velocity_gain"] * vel[i]
            p_pred = apply_affine(A, q_pred[None])[0]
            prev_ok = status[k - 1, i] in (OK, RESCUED)
            qprev = corr[k - 1, i] if prev_ok else 0.0
            hw = search * (0.6 if qprev > 0.9 else 1.0 if qprev > 0.7 else 1.5)
            m = match_template(g, current[i], p_pred, hw)
            q = apply_affine(Ainv, np.array([[m.x, m.y]]))[0] if m.ok else None
            if m.ok and m.corr >= tr["corr_accept"] and np.linalg.norm(q - q_pred) <= tr["max_step_px"]:
                st = OK
            else:
                st = LOST
                p_last = apply_affine(A, q_last[i][None])[0]
                m2 = match_template(g, anchor[i], p_last, search * tr["rescue_scale"])
                if m2.ok and m2.corr >= tr["corr_rescue"] and pattern_score(g, (m2.x, m2.y), pats[i], mk["marker_px"]) >= tr["pattern_min_corr"]:
                    q2 = apply_affine(Ainv, np.array([[m2.x, m2.y]]))[0]
                    j = nb[i]
                    qj = q_new[j] if np.isfinite(q_new[j, 0]) else q_last[j]
                    if np.linalg.norm((qj - q2) - v0[i]) <= tr["neighbor_tol"] * np.linalg.norm(v0[i]):
                        m, q, st = m2, q2, RESCUED
                if st == LOST:
                    corr[k, i] = m.corr if np.isfinite(m.corr) else np.nan
            status[k, i] = st
            if st != LOST:
                xy[k, i], corr[k, i], q_new[i] = (m.x, m.y), m.corr, q
                if tr["template_update"] == "ema" and st == OK and m.corr >= tr["update_min_corr"]:
                    patch = znorm(extract_patch(g, (m.x, m.y), tsz))
                    a = float(tr["ema_alpha"])
                    current[i] = znorm((1 - a) * current[i] + a * patch)

        # ---------------- frame-level checks
        acc = status[k] <= RESCUED
        n_lost = int((~acc).sum())
        mean_c = float(np.nanmean(corr[k, acc])) if acc.any() else 0.0
        steps = np.linalg.norm(q_new[acc] - q_last[acc], axis=1) if acc.any() else np.zeros(0)
        med_step = float(np.median(steps)) if steps.size else 0.0
        if n_lost > tr["frame_max_lost_ratio"] * N or mean_c < tr["frame_min_mean_corr"]:
            reason[k] = "LOW_QUALITY"
        elif not acc[ref_ids].any():
            reason[k] = "REF_LOST"
        elif steps_hist and med_step > max(tr["step_spike_min_px"], tr["step_spike_k"] * float(np.mean(steps_hist))):
            reason[k] = "STEP_SPIKE"
        if reason[k]:
            xy[k] = np.nan
            status[k] = BAD_FRAME
            log(f"[markers] frame {source.indices[k]}: dropped ({reason[k]}, lost {n_lost}, mean ZNCC {mean_c:.2f})")
            continue
        steps_hist.append(med_step)
        for i in np.where(acc)[0]:
            v = q_new[i] - q_last[i] if last_k[i] == k - 1 else np.zeros(2)
            nv = np.linalg.norm(v)
            vel[i] = v if nv <= tr["max_velocity_px"] else v * tr["max_velocity_px"] / nv
            q_last[i], last_k[i] = q_new[i], k
        if k % max(1, T // 10) == 0:
            log(f"[markers] {k}/{T - 1}  mean ZNCC {mean_c:.3f}  lost {n_lost}")
    return TrackResult(source.frame_index, source.times, xy, corr, status, reason, init_xy, init_corr, grid.names())
