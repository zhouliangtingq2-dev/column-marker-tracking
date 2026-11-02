#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate a synthetic test video of an RC column face (PNG frames), its ground truth and a ready-to-run
config.yaml. No real experiment data is used.

Contents: textured column face 300 x 1100 mm with 3 x 11 checker markers (two quadrant patterns), a stub
with corners and anchor nuts that do not move (fixed points), a loading beam; known lateral (double
curvature) and axial deformation; camera shake (translation + small roll, one bump with motion blur);
PSF blur, illumination flicker, vignetting and noise; occlusion / glare / tearing frames.

Output (relative to --out):
  frames/frame_0000.png ...    8-bit grayscale frames
  truth.csv                    per-frame ground truth: theta, dv, camera motion, injected anomaly
  truth_markers_px.csv         true marker centres in each frame (px)
  truth_markers_mm.csv         true marker positions on the face (mm)
  config.yaml                  settings for scripts/01-04 (the points a user would click on frame 0)

Usage:
  python scripts/00_make_synthetic.py --out demo_runs/synthetic --frames 180 --seed 0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cmtrack.synth import SynthConfig, generate  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("demo_runs/synthetic"))
    ap.add_argument("--frames", type=int, default=180)
    ap.add_argument("--width", type=int, default=600)
    ap.add_argument("--height", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-anomalies", action="store_true", help="no occlusion / glare / tearing / camera bump")
    args = ap.parse_args()
    t0 = time.time()
    cfg = SynthConfig(width=args.width, height=args.height, n_frames=args.frames, seed=args.seed, anomalies=not args.no_anomalies)
    res = generate(args.out, cfg)
    print(f"anomalies: {res['anomalies']}")
    print(f"config: {res['config']}  ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
