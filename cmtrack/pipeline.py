"""The four processing steps (used by scripts/01-04 and scripts/run_pipeline.py).

step_camera : fixed points -> camera_motion.csv, fixed_points_px.csv
step_track  : marker initialisation + tracking -> markers_px.csv, markers_corr.csv, markers_status.csv,
              frames.csv, init_markers.csv
step_deform : camera correction -> jump detection -> PCHIP -> px->mm -> theta / dv / spacing
              -> markers_mm.csv, markers_filled.csv, displacement_mm.csv, spacing_*.csv, deformation.csv,
                 bad_frames.csv, summary.json
step_plot   : figures in <out>/fig
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import deform, plots
from .camera import CameraMotion, estimate_camera_motion
from .clean import jump_mask, pchip_fill
from .csvio import read_table, wide, write_table
from .frames import FrameSource
from .markers import RESCUED, TrackResult, grid_from_config, track_markers
from .scale import from_config as scale_from_config


def out_dir_of(cfg: dict, out=None) -> Path:
    d = Path(out or cfg["output"]["dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def step_camera(cfg: dict, out=None, log=print) -> CameraMotion:
    out = out_dir_of(cfg, out)
    src = FrameSource.from_config(cfg)
    t0 = time.time()
    if not cfg["camera"]["enabled"]:
        cam = CameraMotion.identity(src.frame_index, src.times)
        cam.save_csv(out / "camera_motion.csv")
        log("[camera] disabled -> identity transforms")
        return cam
    cam, xy, corr = estimate_camera_motion(src, cfg, log=log)
    cam.save_csv(out / "camera_motion.csv")
    names = [f"p{i + 1}" for i in range(xy.shape[1])]
    write_table(out / "fixed_points_px.csv", {"frame": cam.frame, "t_s": cam.t_s, **wide(names, xy),
                                              **{f"{n}_corr": corr[:, i] for i, n in enumerate(names)}})
    p = cam.params()
    log(f"[camera] {len(src)} frames in {time.time() - t0:.1f} s; |dx| max {np.nanmax(np.abs(p['dx_px'])):.2f} px, "
        f"|dy| max {np.nanmax(np.abs(p['dy_px'])):.2f} px, |rot| max {np.nanmax(np.abs(p['rot_deg'])):.3f} deg, "
        f"frames without estimate {(~cam.valid).sum()}")
    return cam


def step_track(cfg: dict, out=None, use_camera: bool = True, log=print) -> TrackResult:
    out = out_dir_of(cfg, out)
    src = FrameSource.from_config(cfg)
    cam = None
    if use_camera and cfg["camera"]["enabled"]:
        p = out / "camera_motion.csv"
        if not p.exists():
            raise FileNotFoundError(f"{p} not found: run scripts/01_camera_motion.py first (or set camera.enabled: false)")
        cam = CameraMotion.load_csv(p)
        if len(cam.frame) != len(src) or np.any(cam.frame != src.frame_index):
            raise ValueError("camera_motion.csv does not match the selected frames (input.start/stop/step changed?)")
    t0 = time.time()
    res = track_markers(src, cfg, cam, log=log)
    res.save(out)
    n_bad = sum(1 for r in res.reason if r)
    log(f"[markers] {len(src)} frames in {time.time() - t0:.1f} s; dropped frames {n_bad}; "
        f"rescued {int((res.status == RESCUED).sum())}, lost {int((res.status == 2).sum())} marker-frames")
    return res


def step_deform(cfg: dict, out=None, use_camera: bool = True, tracking_dir=None, camera_csv=None, log=print) -> dict:
    out = out_dir_of(cfg, out)
    tdir = Path(tracking_dir) if tracking_dir else out
    grid = grid_from_config(cfg)
    names = grid.names()
    trk = TrackResult.load(tdir, grid)
    T, N = trk.xy.shape[:2]
    if use_camera and cfg["camera"]["enabled"]:
        cam = CameraMotion.load_csv(camera_csv or (tdir / "camera_motion.csv"))
    else:
        cam = CameraMotion.identity(trk.frame, trk.t_s)
    cl = cfg["cleaning"]

    # 1. camera-corrected image coordinates (the frame-0 camera); weak matches are not used
    Q = np.full_like(trk.xy, np.nan)
    for k in range(T):
        good = np.isfinite(trk.xy[k, :, 0])
        if good.any():
            Q[k, good] = cam.to_frame0(k, trk.xy[k, good])
    low_corr = np.isfinite(Q[:, :, 0]) & (trk.corr < cl["min_corr"])
    low_corr[0] = False
    Q[low_corr] = np.nan
    dropped = {k: r for k, r in enumerate(trk.reason) if r}
    for t0, t1 in cl.get("drop_ranges_s") or []:
        for k in np.where((trk.t_s >= t0) & (trk.t_s <= t1))[0]:
            if k > 0:
                dropped.setdefault(int(k), "MANUAL")
    lost_markers = ~np.isfinite(trk.xy[:, :, 0])

    # 2. per-marker jumps relative to the running median; many jumps in one frame -> drop the frame
    for k in dropped:
        Q[k] = np.nan
    jm = jump_mask(Q.reshape(T, -1), cl["hampel_window"], cl["hampel_k"], cl["hampel_min_px"]).reshape(T, N, 2).any(axis=2)
    jm[0] = False
    for k in np.where(jm.sum(axis=1) >= cl["frame_jump_markers"])[0]:
        dropped.setdefault(int(k), f"MARKER_JUMPS({int(jm[k].sum())})")
        Q[k] = np.nan
    Q[jm] = np.nan

    scale = None
    for it in range(5):
        # 3. fill gaps (PCHIP), px -> mm, deformation
        Qf, interp = pchip_fill(Q.reshape(T, -1))
        Qf, interp = Qf.reshape(T, N, 2), interp.reshape(T, N, 2).any(axis=2)
        if scale is None:
            scale = scale_from_config(cfg, Qf[0], grid)
        XY = scale.to_mm(Qf)
        res = deform.compute(XY, grid, cfg)
        # 4. jumps of theta / dv relative to their running median -> drop those frames and recompute
        jt = jump_mask(res["theta"], cl["series_window"], cl["series_k"], cl["series_min_theta_rad"])
        jd = jump_mask(res["dv"], cl["series_window"], cl["series_k"], cl["series_min_dv_mm"])
        new = [int(k) for k in np.where(jt | jd)[0] if k > 0 and k not in dropped]
        if not new or it == 4:
            break
        for k in new:
            dropped[k] = "+".join(n for n, m in (("THETA_JUMP", jt[k]), ("DV_JUMP", jd[k])) if m)
            Q[k] = np.nan

    # ---------------- outputs
    is_dropped = np.zeros(T, bool)
    is_dropped[list(dropped)] = True
    n_interp = interp.sum(axis=1)
    disp = XY - XY[0]
    ref = grid.row(int(cfg["deformation"]["reference_row"]))
    disp_rel = disp - disp[:, ref].mean(axis=1, keepdims=True)
    write_table(out / "markers_mm.csv", {"frame": trk.frame, "t_s": trk.t_s, **wide(names, XY, ("X", "Y"))})
    write_table(out / "markers_filled.csv", {"frame": trk.frame, "t_s": trk.t_s, **wide(names, interp.astype(int), None)})
    write_table(out / "displacement_mm.csv", {"frame": trk.frame, "t_s": trk.t_s, **wide(names, disp, ("dX", "dY")),
                                              **wide(names, disp_rel, ("dX_rel", "dY_rel"))})
    write_table(out / "spacing_x_mm.csv", {"frame": trk.frame, "t_s": trk.t_s, **wide(res["spacing_x_names"], res["spacing_x"], None)})
    write_table(out / "spacing_y_mm.csv", {"frame": trk.frame, "t_s": trk.t_s, **wide(res["spacing_y_names"], res["spacing_y"], None)})
    cols = {"frame": trk.frame, "t_s": trk.t_s, "theta_rad": res["theta"]}
    for c in range(grid.cols):
        cols[f"theta_c{c + 1}_rad"] = res["theta_cols"][:, c]
    cols.update({"dv_mm": res["dv"], "eps_v": res["eps_v"], "dropped": is_dropped.astype(int), "n_interp": n_interp,
                 "reason": [dropped.get(k, "") for k in range(T)]})
    write_table(out / "deformation.csv", cols)
    bad = [k for k in range(T) if is_dropped[k] or n_interp[k] > 0]
    write_table(out / "bad_frames.csv", {
        "frame": trk.frame[bad], "t_s": trk.t_s[bad],
        "action": ["dropped" if is_dropped[k] else "markers_interpolated" for k in bad],
        "reason": [dropped.get(k, "") for k in bad],
        "n_markers_interpolated": n_interp[bad],
        "n_markers_lost_in_tracking": lost_markers[bad].sum(axis=1),
        "n_low_corr": low_corr[bad].sum(axis=1),
        "n_marker_jumps": jm[bad].sum(axis=1)})
    summary = {
        "frames": int(T), "fps": float(1.0 / np.median(np.diff(trk.t_s))) if T > 1 else None,
        "camera_correction": bool(use_camera and cfg["camera"]["enabled"]),
        "dropped_frames": int(is_dropped.sum()), "frames_with_interpolated_markers": int(((n_interp > 0) & ~is_dropped).sum()),
        "dropped_reasons": {r: sum(1 for v in dropped.values() if v == r) for r in sorted(set(dropped.values()))},
        "scale": scale.describe(), "gauge_length_mm": res["gauge_mm"],
        "theta_rad_range": [float(np.min(res["theta"])), float(np.max(res["theta"]))],
        "dv_mm_range": [float(np.min(res["dv"])), float(np.max(res["dv"]))],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"[deform] dropped {summary['dropped_frames']} of {T} frames {summary['dropped_reasons']}, "
        f"{summary['frames_with_interpolated_markers']} more frames with interpolated markers; "
        f"theta {summary['theta_rad_range'][0]:+.5f}..{summary['theta_rad_range'][1]:+.5f} rad, "
        f"dv {summary['dv_mm_range'][0]:+.3f}..{summary['dv_mm_range'][1]:+.3f} mm")
    return {"XY": XY, "deform": res, "dropped": is_dropped, "n_interp": n_interp, "summary": summary, "t_s": trk.t_s}


def step_plot(cfg: dict, out=None, log=print) -> list:
    out = out_dir_of(cfg, out)
    fig = out / "fig"
    fig.mkdir(exist_ok=True)
    grid = grid_from_config(cfg)
    d = read_table(out / "deformation.csv")
    f = read_table(out / "frames.csv")
    dropped = d["dropped"] > 0
    interp = (d["n_interp"] > 0) & ~dropped
    made = [plots.timeseries(fig / "timeseries.png", d["t_s"], d["theta_rad"], d["dv_mm"], f["mean_corr"], f["min_corr"], dropped, interp),
            plots.dv_theta(fig / "dv_theta.png", d["theta_rad"], d["dv_mm"], dropped)]
    cm = out / "camera_motion.csv"
    if cm.exists():
        c = read_table(cm)
        made.append(plots.camera(fig / "camera_motion.png", c["t_s"], c["dx_px"], c["dy_px"], c["rot_deg"]))
    disp = read_table(out / "displacement_mm.csv")
    c_mid = (grid.cols + 1) // 2
    ids = [grid.names()[i] for i in grid.column(c_mid)]
    made.append(plots.column_displacements(fig / "column_displacement.png", disp["t_s"],
                                           np.stack([disp[f"{n}_dX_rel"] for n in ids], 1),
                                           np.stack([disp[f"{n}_dY_rel"] for n in ids], 1), ids, dropped))
    log("[plot] " + ", ".join(str(p.name) for p in made))
    return made
