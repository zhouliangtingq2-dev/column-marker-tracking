#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run steps 1-4 in one go (camera motion -> marker tracking -> deformation -> figures).

Usage:
  python scripts/run_pipeline.py --config my_test.yaml
  python scripts/run_pipeline.py --config my_test.yaml --config picked.yaml --out runs/test1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cmtrack.config import load_config  # noqa: E402
from cmtrack.pipeline import step_camera, step_deform, step_plot, step_track  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    t0 = time.time()
    step_camera(cfg, args.out)
    step_track(cfg, args.out)
    step_deform(cfg, args.out)
    if not args.no_plots and cfg["output"].get("plots", True):
        step_plot(cfg, args.out)
    print(f"done in {time.time() - t0:.1f} s -> {args.out or cfg['output']['dir']}")


if __name__ == "__main__":
    main()
