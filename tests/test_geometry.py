# -*- coding: utf-8 -*-
"""Marker grid, px -> mm conversion and the deformation quantities (theta, dv, spacing)."""
import numpy as np
import pytest

from cmtrack import deform
from cmtrack.config import load_config
from cmtrack.layout import Grid, grid_from_corners, patterns
from cmtrack.scale import from_rectangle, from_spacing

GRID = Grid(3, 11)


def test_grid_ids_and_corner_interpolation():
    assert GRID.n == 33 and GRID.idx(1, 1) == 0 and GRID.idx(2, 1) == 11 and GRID.idx(3, 11) == 32
    assert GRID.names()[11] == "c2r01" and GRID.cr(21) == (2, 11)
    assert GRID.vertical_neighbor(GRID.idx(2, 11)) == GRID.idx(2, 10)
    p = grid_from_corners(GRID, (100, 900), (110, 100), (300, 906), (310, 108))
    assert np.allclose(p[GRID.idx(1, 1)], (100, 900)) and np.allclose(p[GRID.idx(3, 11)], (310, 108))
    assert np.allclose(p[GRID.idx(2, 6)], (205, 503.5))
    assert patterns(GRID, [6, 7, 8, 9, 10, 11]) == (["A"] * 5 + ["B"] * 6) * 3


def test_px_to_mm_from_rectangle_and_spacing():
    corners = {"LB": (100.0, 900.0), "LT": (100.0, 75.0), "RB": (325.0, 900.0), "RT": (325.0, 75.0)}
    for mode in ("rectangle", "affine", "homography"):
        s = from_rectangle(corners, 300.0, 1100.0, mode)
        assert s.mm_per_px_x == pytest.approx(300 / 225) and s.mm_per_px_y == pytest.approx(1100 / 825)
        got = s.to_mm(np.array([corners[k] for k in ("LB", "LT", "RB", "RT")]))
        assert np.allclose(got, [[0, 0], [0, 1100], [300, 0], [300, 1100]], atol=1e-6), mode
    # a slightly rolled camera: affine / homography still map the corners exactly
    a = np.radians(0.5)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    rolled = {k: tuple(R @ (np.array(v) - (200, 500)) + (200, 500)) for k, v in corners.items()}
    for mode in ("affine", "homography"):
        got = from_rectangle(rolled, 300.0, 1100.0, mode).to_mm(np.array([rolled["RT"], rolled["LB"]]))
        assert np.allclose(got, [[300, 1100], [0, 0]], atol=1e-6)
    xy0 = grid_from_corners(GRID, (130, 860), (130, 110), (280, 860), (280, 110))     # 75 px pitch both ways
    s = from_spacing(xy0, GRID, (100.0, 100.0))
    assert s.mm_per_px_x == pytest.approx(100 / 75) and s.mm_per_px_y == pytest.approx(100 / 75)
    assert np.allclose(s.to_mm(xy0[[0, 32]]), [[0, 0], [200, 1000]])


def column_points(theta=0.0, strain=0.0, shift=(0.0, 0.0)):
    """Marker positions (mm) of a column rotated rigidly by theta about its base and stretched vertically."""
    ref = np.array([(50.0 + 100 * (c - 1), 50.0 + 100 * (r - 1)) for c in range(1, 4) for r in range(1, 12)])
    p = ref.copy()
    p[:, 1] *= 1 + strain
    R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])   # +theta: top moves to +X
    return p @ R.T + np.asarray(shift)


def test_theta_dv_and_spacing_on_known_motion():
    th, eps = 1 / 150, -0.0012
    XY = np.stack([column_points(), column_points(th, eps, (3.0, -1.0))])
    cfg = load_config()
    res = deform.compute(XY, GRID, cfg)
    assert res["theta"][1] == pytest.approx(th, abs=1e-12)
    assert np.allclose(res["theta_cols"][1], th, atol=1e-12)
    gauge = 1000.0 - 100.0                     # mean(rows 10, 11) - mean(rows 1, 2) of the middle column
    assert res["gauge_mm"] == pytest.approx(gauge)
    expected_dv = gauge * (1 + eps) * np.cos(th) - gauge
    assert res["dv"][1] == pytest.approx(expected_dv, abs=1e-9)
    assert res["eps_v"][1] == pytest.approx(expected_dv / gauge)
    assert res["spacing_y"].shape == (2, 30) and res["spacing_x"].shape == (2, 22)
    assert np.allclose(res["spacing_y"][1], 100 * (1 + eps) * np.cos(th) - 100, atol=1e-9)
    cfg["deformation"]["dv_sign"] = "shortening"
    assert deform.compute(XY, GRID, cfg)["dv"][1] == pytest.approx(-expected_dv)


def test_example_config_loads_and_merges(tmp_path):
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    over = tmp_path / "picked.yaml"
    over.write_text("camera: {points_px: [[1, 2], [3, 4]]}\noutput: {dir: out}\n", encoding="utf-8")
    cfg = load_config([root / "configs" / "example.yaml", over])
    assert cfg["camera"]["points_px"] == [[1, 2], [3, 4]] and cfg["camera"]["model"] == "rigid"
    assert cfg["input"]["frames"].endswith("path/to/video.mp4".replace("/", __import__("os").sep))
    assert cfg["output"]["dir"] == str((tmp_path / "out").resolve())
    assert cfg["markers"]["rows"] == 11 and cfg["cleaning"]["min_corr"] == 0.85


def test_theta_fit_ignores_missing_markers_and_translation():
    XY = np.stack([column_points(), column_points(0.004, 0.0, (12.0, 5.0))])
    XY[1, GRID.idx(2, 4)] = np.nan
    assert deform.theta_column(XY, GRID, 2)[1] == pytest.approx(0.004, abs=1e-12)
