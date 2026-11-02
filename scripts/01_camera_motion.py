#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 1: camera motion from fixed points (corners / features on the stub or other parts that do not move).

Each fixed point keeps its frame-0 template (no update, no drift); per frame the points are found by ZNCC
(sub-pixel) and a translation or similarity transform frame 0 -> frame k is fitted.

Output (in output.dir or --out): camera_motion.csv (a, b, tx, ty, dx/dy at the fixed-point centroid,
rotation, scale, points used, residual), fixed_points_px.csv

Usage:
  python scripts/01_camera_motion.py --config demo_runs/synthetic/config.yaml
  python scripts/01_camera_motion.py --config my.yaml --config picked.yaml --out runs/test1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cmtrack.config import load_config  # noqa: E402
from cmtrack.pipeline import step_camera  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, action="append", required=True, help="YAML/JSON; repeat to merge several files")
    ap.add_argument("--out", type=Path, default=None, help="output folder (default: output.dir of the config)")
    args = ap.parse_args()
    step_camera(load_config(args.config), args.out)


if __name__ == "__main__":
    main()
