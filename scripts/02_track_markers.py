#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 2: initialise and track the 3 x 11 checker markers.

Initialisation refines the clicked / interpolated positions with the ideal quadrant template of each
marker (pattern A or B). Tracking predicts each marker in camera-corrected coordinates, matches it by ZNCC
in a search window with a sub-pixel peak, rescues lost markers in a larger window, and drops frames that
are unusable as a whole (LOW_QUALITY / REF_LOST / STEP_SPIKE / CAMERA_LOST).

Output: markers_px.csv (raw image px, NaN = lost), markers_corr.csv, markers_status.csv
        (0 ok / 1 rescued / 2 lost / 3 dropped frame), frames.csv, init_markers.csv

Usage:
  python scripts/02_track_markers.py --config demo_runs/synthetic/config.yaml
  python scripts/02_track_markers.py --config my.yaml --no-camera      # without camera_motion.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cmtrack.config import load_config  # noqa: E402
from cmtrack.pipeline import step_track  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-camera", action="store_true", help="ignore camera_motion.csv (identity transforms)")
    args = ap.parse_args()
    step_track(load_config(args.config), args.out, use_camera=not args.no_camera)


if __name__ == "__main__":
    main()
