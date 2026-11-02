#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
End-to-end demonstration on a synthetic video with known ground truth (CPU, about one minute).

  00  synthetic video: column face with 3 x 11 checker markers, stub with fixed points, camera shake,
      known theta / dv, occlusion / glare / tearing / camera-bump frames
  01  camera motion from the fixed points          02  marker tracking
  03  deformation (theta, dv, spacing)             04  figures
  then, with the same tracking, the deformation is recomputed without camera correction and with the
  translation / similarity camera models, and everything is compared with the ground truth.

The numbered scripts are called exactly as a user would call them, so the demo is also a smoke test of
the whole workflow (CI runs it on every push).

Output: <root>/report/report.md (+ summary.json and figures); with --docs-img the figures used in the
README are refreshed.

Usage:
  python scripts/demo.py --root demo_runs
  python scripts/demo.py --root demo_runs --docs-img docs/img
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from cmtrack import plots  # noqa: E402
from cmtrack.camera import CameraMotion, apply_affine  # noqa: E402
from cmtrack.csvio import read_table, unwide  # noqa: E402
from cmtrack.layout import Grid  # noqa: E402

GRID = Grid(3, 11)
NAMES = GRID.names()


def run(cmd: list, log: list, label: str):
    cmd = [str(c) for c in cmd]
    print("\n[RUN] " + " ".join(cmd), flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True)
    log.append((label, time.time() - t0))


def rms(x) -> float:
    return float(np.sqrt(np.nanmean(np.square(x))))


def evaluate(run_dir: Path, syn: Path) -> dict:
    tr = read_table(syn / "truth.csv")
    d = read_table(run_dir / "deformation.csv")
    e_th, e_dv = d["theta_rad"] - tr["theta_rad"], d["dv_mm"] - tr["dv_mm"]
    XY = unwide(read_table(run_dir / "markers_mm.csv"), NAMES, ("X", "Y"))
    T = unwide(read_table(syn / "truth_markers_mm.csv"), NAMES, ("X", "Y"))
    e_disp = np.linalg.norm((XY - XY[0]) - (T - T[0]), axis=2)
    return {"theta_rms": rms(e_th), "theta_max": float(np.max(np.abs(e_th))), "dv_rms": rms(e_dv), "dv_max": float(np.max(np.abs(e_dv))),
            "disp_rms": rms(e_disp), "disp_max": float(np.max(e_disp)), "e_th": e_th, "e_dv": e_dv, "d": d}


def camera_error(run_dir: Path, syn: Path) -> dict:
    tr = read_table(syn / "truth.csv")
    cam = CameraMotion.load_csv(run_dir / "camera_motion.csv")
    P0 = unwide(read_table(syn / "truth_markers_px.csv"), NAMES)[0]
    A_true = np.stack([np.stack([tr["A_a"], -tr["A_b"], tr["A_tx"]], 1), np.stack([tr["A_b"], tr["A_a"], tr["A_ty"]], 1)], 1)
    err = np.array([np.linalg.norm(apply_affine(cam.A[k], P0) - apply_affine(A_true[k], P0), axis=1) for k in range(len(cam.A))])
    p = cam.params()
    return {"rms_px": rms(err), "max_px": float(err.max()), "params": p, "t": cam.t_s,
            "max_shift_px": float(np.max(np.hypot(tr["cam_dx_px"], tr["cam_dy_px"]))),
            "max_rot_deg": float(np.max(np.abs(tr["cam_rot_rel_deg"])))}


def frames_figure(syn: Path, run_dir: Path, out_png: Path, cfg_points, occl_k: int, drift_k: int):
    """frame 0 with fixed points / corners / markers | occlusion frame | largest-drift frame."""
    P = unwide(read_table(run_dir / "markers_px.csv"), NAMES)
    filled = unwide(read_table(run_dir / "markers_filled.csv"), NAMES, None)
    TP = unwide(read_table(syn / "truth_markers_px.csv"), NAMES)
    panels = []
    labels = (["frame 0", "fixed points (orange)", "face corners (cyan)", "markers (green)"],
              [f"frame {occl_k}: occlusion", "red: not used -> PCHIP"],
              [f"frame {drift_k}: largest drift", "theta = 1/50 cycle"])
    for j, k in enumerate((0, occl_k, drift_k)):
        g = cv2.imread(str(syn / "frames" / f"frame_{k:04d}.png"), cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        if j == 0:
            for x, y in cfg_points["fixed"]:
                cv2.rectangle(img, (int(x) - 20, int(y) - 20), (int(x) + 20, int(y) + 20), (0, 140, 255), 2)
            for x, y in cfg_points["corners"]:
                cv2.circle(img, (int(round(x)), int(round(y))), 7, (255, 200, 0), 2)
        for i in range(GRID.n):
            if filled[k, i] == 0:
                x, y = P[k, i]
                cv2.drawMarker(img, (int(round(x)), int(round(y))), (40, 200, 40), cv2.MARKER_CROSS, 18, 2)
            else:
                x, y = TP[k, i]
                cv2.circle(img, (int(round(x)), int(round(y))), 13, (40, 40, 230), 2)
        img = cv2.resize(img, (img.shape[1] * 3 // 5, img.shape[0] * 3 // 5), interpolation=cv2.INTER_AREA)
        band = np.full((76, img.shape[1], 3), 255, np.uint8)
        for n, text in enumerate(labels[j]):
            cv2.putText(band, text, (6, 16 + 17 * n), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 1, cv2.LINE_AA)
        img = np.concatenate([band, img], axis=0)
        panels.append(img)
        panels.append(np.full((img.shape[0], 4, 3), 255, np.uint8))
    sheet = np.concatenate(panels[:-1], axis=1)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85] if out_png.suffix == ".jpg" else [cv2.IMWRITE_PNG_COMPRESSION, 9])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("demo_runs"))
    ap.add_argument("--frames", type=int, default=180)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--docs-img", type=Path, default=None, help="also write the README figures here (e.g. docs/img)")
    args = ap.parse_args()
    root = args.root
    syn, run_dir, rep = root / "synthetic", root / "run", root / "report"
    rep.mkdir(parents=True, exist_ok=True)
    py, log, t_all = sys.executable, [], time.time()

    run([py, HERE / "00_make_synthetic.py", "--out", syn, "--frames", args.frames, "--seed", args.seed], log, "00 synthetic video")
    cfg = syn / "config.yaml"
    run([py, HERE / "01_camera_motion.py", "--config", cfg, "--out", run_dir], log, "01 camera motion")
    run([py, HERE / "02_track_markers.py", "--config", cfg, "--out", run_dir], log, "02 marker tracking")
    run([py, HERE / "03_deformation.py", "--config", cfg, "--out", run_dir], log, "03 deformation")
    run([py, HERE / "04_plot.py", "--config", cfg, "--out", run_dir], log, "04 figures")
    variants = {"rigid (default)": run_dir}
    for model in ("translation", "similarity"):
        vdir = root / f"camera_{model}"
        vdir.mkdir(parents=True, exist_ok=True)
        (vdir / "override.yaml").write_text(f"camera: {{model: {model}}}\n", encoding="utf-8")
        run([py, HERE / "01_camera_motion.py", "--config", cfg, "--config", vdir / "override.yaml", "--out", vdir], log, f"01 camera ({model})")
        run([py, HERE / "03_deformation.py", "--config", cfg, "--tracking-dir", run_dir, "--camera-csv", vdir / "camera_motion.csv",
             "--out", vdir], log, f"03 deformation ({model})")
        variants[model] = vdir
    none_dir = root / "camera_none"
    run([py, HERE / "03_deformation.py", "--config", cfg, "--tracking-dir", run_dir, "--no-camera", "--out", none_dir], log, "03 deformation (no correction)")
    variants["none"] = none_dir

    # ---------------- evaluation
    tr = read_table(syn / "truth.csv")
    ev = {name: evaluate(p, syn) for name, p in variants.items()}
    main_ev = ev["rigid (default)"]
    cam = camera_error(run_dir, syn)
    bad = read_table(run_dir / "bad_frames.csv")
    action = {int(f): (a, r, int(n)) for f, a, r, n in zip(bad["frame"], bad["action"], bad["reason"], bad["n_markers_interpolated"])}
    anomalies = {k: a for k, a in enumerate(tr["anomaly"]) if isinstance(a, str) and a}
    need = [k for k, a in anomalies.items() if a != "camera_bump"]
    handled = [k for k in need if k in action]
    false_flags = sorted(k for k in action if k not in anomalies)
    bump_ok = all(abs(main_ev["e_dv"][k]) < 4 * main_ev["dv_rms"] + 0.02 and abs(main_ev["e_th"][k]) < 4 * main_ev["theta_rms"] + 1e-5
                  for k, a in anomalies.items() if a == "camera_bump")
    summ = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

    # ---------------- figures
    figs = {}
    d = main_ev["d"]
    f = read_table(run_dir / "frames.csv")
    dropped = d["dropped"] > 0
    interp = (d["n_interp"] > 0) & ~dropped
    figs["demo_timeseries.png"] = plots.timeseries(rep / "demo_timeseries.png", d["t_s"], d["theta_rad"], d["dv_mm"], f["mean_corr"], f["min_corr"],
                                                   dropped, interp, tr["theta_rad"], tr["dv_mm"],
                                                   title="synthetic video: tracked vs ground truth")
    figs["demo_dv_theta.png"] = plots.dv_theta(rep / "demo_dv_theta.png", d["theta_rad"], d["dv_mm"], dropped, tr["theta_rad"], tr["dv_mm"],
                                               title="dv - theta (synthetic)")
    figs["demo_camera.png"] = shake_figure(rep / "demo_camera.png", cam, tr, ev)
    c = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    pts = {"fixed": c["camera"]["points_px"], "corners": list(c["scale"]["corners_px"].values())}
    occl = [k for k, a in anomalies.items() if a == "occlusion"]
    occl_k = occl[min(2, len(occl) - 1)] if occl else len(tr["theta_rad"]) // 3
    drift_k = int(np.argmax(np.abs(tr["theta_rad"])))
    figs["demo_frames.jpg"] = rep / "demo_frames.jpg"
    frames_figure(syn, run_dir, figs["demo_frames.jpg"], pts, occl_k, drift_k)

    # ---------------- report
    th_true, dv_true = tr["theta_rad"], tr["dv_mm"]
    L = ["# Synthetic demo report", "",
         f"synthetic video: {len(th_true)} frames, 600 x 1000 px, 30 fps, seed {args.seed}, 0.75 px/mm (1.33 mm/px), markers 30 px; "
         f"camera shake up to {cam['max_shift_px']:.1f} px / {cam['max_rot_deg']:.3f} deg; camera model rigid (default)", "",
         "## Accuracy against the ground truth (all frames)", "",
         "| quantity | RMS error | max abs error | true range |", "|---|---|---|---|",
         f"| member angle theta (rad) | {main_ev['theta_rms']:.2e} | {main_ev['theta_max']:.2e} | {th_true.min():+.4f} .. {th_true.max():+.4f} |",
         f"| axial deformation dv (mm) | {main_ev['dv_rms']:.3f} | {main_ev['dv_max']:.3f} | {dv_true.min():+.2f} .. {dv_true.max():+.2f} |",
         f"| marker displacement (mm) | {main_ev['disp_rms']:.3f} | {main_ev['disp_max']:.3f} | |",
         f"| camera motion at the markers (px) | {cam['rms_px']:.3f} | {cam['max_px']:.3f} | |", "",
         "theta and dv are compared with the same definitions applied to the true marker positions "
         "(least-squares line through the middle column; rows 10-11 minus rows 1-2 of the middle column).", "",
         "## Effect of the camera-shake correction (same tracking, only the camera model changes)", "",
         "| camera correction | theta RMS (rad) | dv RMS (mm) | displacement RMS (mm) |", "|---|---|---|---|"]
    for name in ("none", "translation", "rigid (default)", "similarity"):
        e = ev[name]
        L.append(f"| {name} | {e['theta_rms']:.2e} | {e['dv_rms']:.3f} | {e['disp_rms']:.3f} |")
    L += ["", "## Injected anomalies", "",
          "| frame | injected | pipeline | theta error (rad) | dv error (mm) |", "|---|---|---|---|---|"]
    for k, a in anomalies.items():
        if k in action:
            act, rsn, n = action[k]
            what = f"frame dropped ({rsn})" if act == "dropped" else f"{n} marker(s) not used -> PCHIP"
        else:
            what = "nothing flagged (absorbed by the camera correction)" if a == "camera_bump" else "NOT flagged"
        L.append(f"| {k} | {a} | {what} | {main_ev['e_th'][k]:+.1e} | {main_ev['e_dv'][k]:+.3f} |")
    L += ["", f"frames flagged without an injected anomaly: {false_flags if false_flags else 'none'}", "",
          "## Checks (parsed by CI)", "", "| check | value |", "|---|---|",
          f"| theta_rms_rad | {main_ev['theta_rms']:.3e} |",
          f"| dv_rms_mm | {main_ev['dv_rms']:.4f} |",
          f"| disp_rms_mm | {main_ev['disp_rms']:.4f} |",
          f"| theta_rms_no_correction_rad | {ev['none']['theta_rms']:.3e} |",
          f"| disp_rms_no_correction_mm | {ev['none']['disp_rms']:.4f} |",
          f"| anomalies_handled | {len(handled)}/{len(need)} |",
          f"| camera_bump_absorbed | {'yes' if bump_ok else 'no'} |",
          f"| false_flags | {len(false_flags)} |",
          f"| frames_dropped | {summ['dropped_frames']} |", "",
          "![frames](demo_frames.jpg)", "", "![time histories](demo_timeseries.png)", "", "![dv-theta](demo_dv_theta.png)", "",
          "![camera](demo_camera.png)", "",
          "## Timing", "", "| step | seconds |", "|---|---|"] + [f"| {n} | {t:.1f} |" for n, t in log] + \
         ["", f"total {time.time() - t_all:.0f} s"]
    (rep / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (rep / "summary.json").write_text(json.dumps({
        "theta_rms_rad": main_ev["theta_rms"], "theta_max_rad": main_ev["theta_max"], "dv_rms_mm": main_ev["dv_rms"], "dv_max_mm": main_ev["dv_max"],
        "disp_rms_mm": main_ev["disp_rms"], "camera_rms_px": cam["rms_px"],
        "by_camera_model": {n: {"theta_rms_rad": e["theta_rms"], "dv_rms_mm": e["dv_rms"], "disp_rms_mm": e["disp_rms"]} for n, e in ev.items()},
        "anomalies": {str(k): a for k, a in anomalies.items()}, "handled": handled, "false_flags": false_flags,
        "timing_s": {n: round(t, 2) for n, t in log}}, indent=2), encoding="utf-8")
    if args.docs_img:
        args.docs_img.mkdir(parents=True, exist_ok=True)
        for name, p in figs.items():
            (args.docs_img / name).write_bytes(Path(p).read_bytes())
    print("\n".join(L[:30]))
    print("\nreport:", rep / "report.md")


def shake_figure(path, cam: dict, tr: dict, ev: dict):
    import matplotlib.pyplot as plt
    t = cam["t"]
    p = cam["params"]
    fig, axs = plt.subplots(3, 1, figsize=(8.0, 6.0), sharex=True)
    axs[0].plot(t, tr["cam_dx_px"], color=plots.C_TRUE, lw=2.4, alpha=0.7, label="true dx")
    axs[0].plot(t, p["dx_px"], color=plots.C_EST, lw=1.0, label="estimated dx")
    axs[0].plot(t, tr["cam_dy_px"], color=plots.C_TRUE, lw=2.4, alpha=0.35, label="true dy")
    axs[0].plot(t, p["dy_px"], color=plots.C_EST, lw=1.0, ls="--", label="estimated dy")
    axs[0].set_ylabel("camera shift (px)")
    axs[0].legend(fontsize=7, ncol=4, loc="upper left")
    axs[1].plot(t, tr["cam_rot_rel_deg"], color=plots.C_TRUE, lw=2.4, alpha=0.7, label="true")
    axs[1].plot(t, p["rot_deg"], color=plots.C_EST, lw=1.0, label="estimated (rigid)")
    axs[1].set_ylabel("camera roll (deg)")
    axs[1].legend(fontsize=7, loc="upper left")
    axs[2].plot(t, ev["none"]["e_th"] * 1e3, color=plots.C_RAW, lw=1.0, label="no correction")
    axs[2].plot(t, ev["translation"]["e_th"] * 1e3, color="#b58900", lw=1.0, label="translation only")
    axs[2].plot(t, ev["rigid (default)"]["e_th"] * 1e3, color=plots.C_EST, lw=1.2, label="rigid (default)")
    axs[2].set_ylabel("theta error (10$^{-3}$ rad)")
    axs[2].set_xlabel("time (s)")
    axs[2].legend(fontsize=7, ncol=3, loc="upper left")
    for ax in axs:
        ax.grid(alpha=0.3)
    axs[0].set_title("camera motion from the fixed points and its effect on theta (synthetic)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=plots.DPI)
    plt.close(fig)
    return path


if __name__ == "__main__":
    main()
