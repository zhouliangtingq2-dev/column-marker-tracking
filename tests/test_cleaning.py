# -*- coding: utf-8 -*-
"""Jump detection, PCHIP gap filling, and the frame-level handling of step 3 on a fake tracking result."""
import json

import numpy as np

from cmtrack.clean import jump_mask, pchip_fill, running_nanmedian
from cmtrack.config import load_config
from cmtrack.layout import Grid
from cmtrack.markers import LOST, OK, TrackResult
from cmtrack.pipeline import step_deform

GRID = Grid(3, 11)


def stretch_of(T):
    return 1 + 4e-5 * np.arange(T)


def test_running_median_and_jump_mask_on_a_smooth_signal():
    rng = np.random.default_rng(0)
    t = np.arange(300)
    x = 8 * np.sin(2 * np.pi * t / 120) + rng.normal(0, 0.03, t.size)
    assert np.allclose(running_nanmedian(np.arange(10.0), 5)[2:-2], np.arange(2.0, 8.0))
    assert not jump_mask(x, 7, 6.0, 0.3).any(), "smooth motion must not be flagged"
    y = x.copy()
    y[[50, 51, 200]] += [2.0, 2.0, -1.0]          # a two-frame glitch and a single spike
    y[120] = np.nan
    m = jump_mask(y, 7, 6.0, 0.3)
    assert set(np.where(m)[0]) == {50, 51, 200}


def test_pchip_fill_keeps_shape_and_holds_the_ends():
    t = np.arange(40, dtype=float)
    x = np.stack([0.01 * t ** 2, np.sin(t / 6)], axis=1)
    y = x.copy()
    y[[0, 1, 17, 18, 19, 39], 0] = np.nan
    y[[10, 11], 1] = np.nan
    f, was = pchip_fill(y)
    assert was.sum() == 8 and np.isfinite(f).all()
    assert np.allclose(f[17:20, 0], x[17:20, 0], atol=0.02)
    assert np.allclose(f[10:12, 1], x[10:12, 1], atol=0.02)          # PCHIP flattens a little at a peak
    assert f[10, 1] <= max(y[9, 1], y[12, 1]) + 1e-12                 # ... and never overshoots the data
    assert f[0, 0] == f[2, 0] and f[39, 0] == f[38, 0]


def _fake_tracking(tmp_path, T=60):
    """Markers of a column (75 px pitch) that leans and stretches slowly; frame 30 has 6 jumped markers,
    frame 40 two lost ones, frame 45 one weak match. Returns the exact theta (slope dX/dY)."""
    ref = np.array([(130.0 + 75 * (c - 1), 860.0 - 75 * (r - 1)) for c in range(1, 4) for r in range(1, 12)])
    lean = 0.004 * np.sin(np.linspace(0, 2 * np.pi, T))
    stretch = stretch_of(T)
    xy = np.zeros((T, GRID.n, 2))
    h = 860.0 - ref[:, 1]
    for k in range(T):
        xy[k, :, 0] = ref[:, 0] + lean[k] * h
        xy[k, :, 1] = 860.0 - stretch[k] * h
    th = np.arctan(lean / stretch)
    status = np.full((T, GRID.n), OK)
    corr = np.full((T, GRID.n), 0.99)
    xy[30, :6, 0] += 9.0                               # tearing-like jump of six markers
    xy[40, [GRID.idx(2, 5), GRID.idx(3, 9)]] = np.nan  # two markers lost in tracking
    status[40, [GRID.idx(2, 5), GRID.idx(3, 9)]] = LOST
    corr[45, GRID.idx(1, 3)] = 0.6                      # a weak match is not used
    trk = TrackResult(np.arange(T), np.arange(T) / 30.0, xy, corr, status, [""] * T, xy[0], np.ones(GRID.n), GRID.names())
    trk.save(tmp_path)
    return xy, th


def test_step3_drops_jump_frames_and_fills_lost_markers(tmp_path):
    xy, th = _fake_tracking(tmp_path)
    cfg = load_config(overrides={"camera": {"enabled": False}, "scale": {"mode": "spacing", "spacing_mm": [100.0, 100.0]}})
    res = step_deform(cfg, tmp_path, log=lambda *a: None)
    s = json.loads((tmp_path / "summary.json").read_text())
    assert res["dropped"][30] and s["dropped_frames"] == 1 and "MARKER_JUMPS" in list(s["dropped_reasons"])[0]
    assert res["n_interp"][40] == 2 and res["n_interp"][45] == 1 and not res["dropped"][40]
    clean = np.ones(len(th), bool)
    clean[[30, 40, 45]] = False
    exact = th - th[0]
    assert np.allclose(res["deform"]["theta"][clean], exact[clean], atol=1e-9)
    assert abs(res["deform"]["theta"][30] - exact[30]) < 2e-5          # PCHIP across the dropped frame
    gauge = 75.0 * 9 * 100 / 75                                          # rows 10-11 minus rows 1-2, in mm
    assert np.allclose(res["deform"]["dv"][clean], (gauge * (stretch_of(len(th)) - 1))[clean], atol=1e-6)
    bad = (tmp_path / "bad_frames.csv").read_text().splitlines()
    assert len(bad) == 1 + 3                                            # header + frames 30, 40, 45
