#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Optional interactive helper: click the points that a config needs on the first frame (OpenCV window).
Not used by the demo or CI; you can also type the coordinates into the YAML by hand.

  --what fixed    : fixed points for the camera-motion estimate (corners on the stub, bolts ...), any number
  --what corners  : the four corners of the column face, in the order LB, LT, RB, RT
  --what markers  : the four corner MARKERS, in the order bottom-left, top-left, bottom-right, top-right

Keys: left click = add, u = undo, Enter = done, Esc = cancel. A magnified view follows the cursor.
The points are written (merged) into --out, which can then be passed as a second --config:

  python scripts/pick_points.py --frames video.mp4 --what fixed   --out picked.yaml
  python scripts/pick_points.py --frames video.mp4 --what corners --out picked.yaml
  python scripts/pick_points.py --frames video.mp4 --what markers --out picked.yaml
  python scripts/run_pipeline.py --config my_test.yaml --config picked.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cmtrack.config import deep_merge  # noqa: E402
from cmtrack.frames import FrameSource  # noqa: E402

ORDER = {"corners": ["LB", "LT", "RB", "RT"], "markers": ["bottom_left", "top_left", "bottom_right", "top_right"]}


def pick(img: np.ndarray, what: str, max_h: int = 950) -> list:
    g = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    s = min(1.0, max_h / g.shape[0])
    view = cv2.resize(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR), None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    pts, cursor = [], [0, 0]
    need = len(ORDER[what]) if what in ORDER else None
    win = f"pick {what} (click, u = undo, Enter = done, Esc = cancel)"

    def on_mouse(event, x, y, flags, param):
        cursor[:] = [x, y]
        if event == cv2.EVENT_LBUTTONDOWN and (need is None or len(pts) < need):
            pts.append((x / s, y / s))

    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        canvas = view.copy()
        for i, (x, y) in enumerate(pts):
            p = (int(round(x * s)), int(round(y * s)))
            cv2.drawMarker(canvas, p, (0, 140, 255), cv2.MARKER_CROSS, 16, 2)
            name = ORDER[what][i] if what in ORDER else str(i + 1)
            cv2.putText(canvas, name, (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1, cv2.LINE_AA)
        nxt = ORDER[what][len(pts)] if what in ORDER and len(pts) < need else ("Enter" if need else "next point / Enter")
        cv2.putText(canvas, f"next: {nxt}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
        cx, cy = int(cursor[0] / s), int(cursor[1] / s)            # magnifier (x8 around the cursor)
        r = 12
        patch = cv2.getRectSubPix(g, (2 * r + 1, 2 * r + 1), (float(cx), float(cy)))
        mag = cv2.resize(cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR), None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
        cv2.drawMarker(mag, (mag.shape[1] // 2, mag.shape[0] // 2), (0, 0, 255), cv2.MARKER_CROSS, 24, 1)
        h, w = mag.shape[:2]
        if canvas.shape[0] > h and canvas.shape[1] > w:
            canvas[-h:, -w:] = mag
        cv2.imshow(win, canvas)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10) and (need is None or len(pts) == need) and pts:
            break
        if key == 27:
            cv2.destroyAllWindows()
            raise SystemExit("cancelled")
        if key == ord("u") and pts:
            pts.pop()
    cv2.destroyAllWindows()
    return [[round(float(x), 1), round(float(y), 1)] for x, y in pts]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=Path, required=True, help="video file, image folder or .npy (frame 0 is shown)")
    ap.add_argument("--what", choices=["fixed", "corners", "markers"], required=True)
    ap.add_argument("--rotate", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("picked.yaml"))
    args = ap.parse_args()
    img = FrameSource(args.frames, rotate=args.rotate).read(0)
    pts = pick(img, args.what)
    if args.what == "fixed":
        new = {"camera": {"points_px": pts}}
    elif args.what == "corners":
        new = {"scale": {"corners_px": dict(zip(ORDER["corners"], pts))}}
    else:
        new = {"markers": {"init": {"mode": "corners", "corners_px": dict(zip(ORDER["markers"], pts))}}}
    old = yaml.safe_load(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    args.out.write_text(yaml.safe_dump(deep_merge(old or {}, new), sort_keys=False, default_flow_style=None), encoding="utf-8")
    print(f"{len(pts)} point(s) written to {args.out}")


if __name__ == "__main__":
    main()
