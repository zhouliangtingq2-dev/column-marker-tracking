#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Step 4: figures in <out>/fig: time histories of theta / dv / ZNCC with dropped and interpolated frames,
dv-theta, camera motion, and the displacements of the middle marker column.

Usage:
  python scripts/04_plot.py --config demo_runs/synthetic/config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cmtrack.config import load_config  # noqa: E402
from cmtrack.pipeline import step_plot  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, action="append", required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    step_plot(load_config(args.config), args.out)


if __name__ == "__main__":
    main()
