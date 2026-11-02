#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 3: camera correction -> jump detection -> PCHIP -> px->mm -> deformation.

  1. marker coordinates are mapped back to the frame-0 camera with the inverse camera transform
  2. per-marker jumps relative to a running median are removed; many jumps in one frame drop the frame
  3. gaps (lost markers, dropped frames) are filled by PCHIP, coordinates converted to mm on the face
  4. theta (least-squares line through one marker column), dv (top rows - bottom rows) and the spacing of
     adjacent markers are computed; frames where theta or dv jump relative to their running median are
     dropped and everything is recomputed

Output: markers_mm.csv, displacement_mm.csv, spacing_x_mm.csv, spacing_y_mm.csv, deformation.csv,
        bad_frames.csv, summary.json

Usage:
  python scripts/03_deformation.py --config demo_runs/synthetic/config.yaml
  python scripts/03_deformation.py --config cfg.yaml --no-camera --tracking-dir runs/a --out runs/a_nocam
  python scripts/03_deformation.py --config cfg.yaml --tracking-dir runs/a --camera-csv runs/b/camera_motion.csv --out runs/b
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cmtrack.config import load_config  # noqa: E402
from cmtrack.pipeline import step_deform  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tracking-dir", type=Path, default=None, help="folder with the step-2 results (default: --out)")
    ap.add_argument("--camera-csv", type=Path, default=None, help="camera_motion.csv to use (default: the one in --tracking-dir)")
    ap.add_argument("--no-camera", action="store_true", help="skip the camera correction (to see its effect)")
    args = ap.parse_args()
    step_deform(load_config(args.config), args.out, use_camera=not args.no_camera, tracking_dir=args.tracking_dir,
                camera_csv=args.camera_csv)


if __name__ == "__main__":
    main()
