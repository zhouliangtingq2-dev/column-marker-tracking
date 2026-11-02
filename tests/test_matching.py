# -*- coding: utf-8 -*-
"""ZNCC sub-pixel matching, quadrant templates and camera-motion estimation.  Run: pytest -q"""
import cv2
import numpy as np
import pytest

from cmtrack.camera import apply_affine, estimate_camera_motion, fit_rigid, fit_robust, fit_similarity, invert_affine
from cmtrack.config import load_config
from cmtrack.frames import FrameSource
from cmtrack.layout import quadrant_template
from cmtrack.synth import checker_image
from cmtrack.zncc import extract_patch, match_template, parabolic_offset


def test_parabolic_offset_is_exact_on_a_parabola():
    for x0 in (-0.4, -0.1, 0.0, 0.27, 0.45):
        f = [-(x - x0) ** 2 for x in (-1, 0, 1)]
        assert parabolic_offset(*f) == pytest.approx(x0, abs=1e-12)
    assert parabolic_offset(1.0, 0.5, 1.0) == 0.0          # a minimum is not refined


def test_subpixel_peak_handles_an_oblique_peak():
    """A correlation peak elongated along a diagonal: the 3 x 3 quadratic (with xy term) finds the vertex,
    separate x / y parabolas would not."""
    from cmtrack.zncc import subpixel_peak
    x0, y0 = 0.31, -0.22
    yy, xx = np.mgrid[-2:3, -2:3].astype(float)
    R = 1 - 0.30 * (xx - x0) ** 2 - 0.30 * (yy - y0) ** 2 - 0.45 * (xx - x0) * (yy - y0)
    dy, dx = subpixel_peak(R, 2, 2)
    assert (dx, dy) == pytest.approx((x0, y0), abs=1e-9)
    sep_dx = parabolic_offset(R[2, 1], R[2, 2], R[2, 3])
    assert abs(sep_dx - x0) > 0.1


def test_zncc_subpixel_on_area_integrated_checker():
    """A template cut from one rendering finds the same checker shifted by sub-pixel amounts (independent renderer)."""
    rng = np.random.default_rng(1)
    ref = checker_image((81, 81), (40.0, 40.0), 12, "A", sticker_px=34)
    tmpl = extract_patch(cv2.GaussianBlur(ref, (0, 0), 1.0), (40.0, 40.0), 31)
    errs = []
    for dx in np.linspace(-0.5, 0.5, 6):
        for dy in np.linspace(-0.45, 0.55, 5):
            img = checker_image((81, 81), (40.0 + dx, 40.0 + dy), 12, "A", sticker_px=34)
            img = img + rng.normal(0, 1.5, img.shape).astype(np.float32)
            m = match_template(cv2.GaussianBlur(img, (0, 0), 1.0), tmpl, (41, 39), 6)
            assert m.ok and m.corr > 0.98
            errs.append(np.hypot(m.x - 40.0 - dx, m.y - 40.0 - dy))
    assert np.sqrt(np.mean(np.square(errs))) < 0.02, f"RMS {np.sqrt(np.mean(np.square(errs))):.3f} px"
    assert max(errs) < 0.05


def test_quadrant_template_finds_the_x_junction_and_the_pattern():
    for pattern, other in (("A", "B"), ("B", "A")):
        img = cv2.GaussianBlur(checker_image((61, 61), (30.3, 29.6), 10, pattern, angle_deg=1.5, sticker_px=27), (0, 0), 1.0)
        m = match_template(img, quadrant_template(15, pattern), (32, 28), 5)
        assert m.ok and m.corr > 0.9
        assert np.hypot(m.x - 30.3, m.y - 29.6) < 0.1
        wrong = match_template(img, quadrant_template(15, other), (30, 30), 1)
        assert wrong.corr < 0.0, "the other checker pattern must not match at the centre"


def test_match_flags_peak_on_the_window_border():
    img = cv2.GaussianBlur(checker_image((81, 81), (52.0, 40.0), 10, "A", sticker_px=27), (0, 0), 1.0)
    tmpl = quadrant_template(15, "A")
    assert not match_template(img, tmpl, (40, 40), 4).ok        # true centre 12 px away, window +-4 px
    assert match_template(img, tmpl, (40, 40), 16).ok


def _rot(deg, t, about=(0.0, 0.0)):
    a = np.radians(deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    c = np.asarray(about)
    return np.hstack([R, (c - R @ c + np.asarray(t))[:, None]])


def test_rigid_and_similarity_fits_recover_known_transforms():
    rng = np.random.default_rng(0)
    P0 = rng.uniform(0, 500, (5, 2))
    A = _rot(0.12, (3.2, -1.7), about=(250, 250))
    assert np.allclose(fit_rigid(P0, apply_affine(A, P0)), A, atol=1e-9)
    S = A.copy()
    S[:, :2] *= 1.0004
    assert np.allclose(fit_similarity(P0, apply_affine(S, P0)), S, atol=1e-9)
    P1 = apply_affine(A, P0)
    P1[2] += (4.0, -3.0)                                            # one fixed point was hit by something
    A2, keep, resid = fit_robust(P0, P1, "rigid", max_resid=0.5)
    assert not keep[2] and keep.sum() == 4 and resid < 1e-6
    assert np.allclose(invert_affine(A) @ np.append(apply_affine(A, P0[:1])[0], 1.0), P0[0])


def blob_scene(A, blobs, shape=(240, 320)):
    """Smooth analytic image (sum of Gaussian blobs) seen through the transform A: evaluated exactly, no resampling."""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(float)
    src = apply_affine(invert_affine(A), np.stack([xx.ravel(), yy.ravel()], 1))
    sx, sy = src[:, 0].reshape(shape), src[:, 1].reshape(shape)
    img = np.full(shape, 110.0)
    for cx, cy, s, a in blobs:
        img += a * np.exp(-((sx - cx) ** 2 + (sy - cy) ** 2) / (2 * s * s))
    return img.astype(np.float32)


def speckle_blobs(seed=3):
    rng = np.random.default_rng(seed)
    fixed = [(60, 60), (260, 70), (70, 190), (250, 180)]
    blobs = [(x + rng.uniform(-9, 9), y + rng.uniform(-9, 9), rng.uniform(2.5, 4.5), rng.choice([-1, 1]) * rng.uniform(40, 90))
             for x, y in fixed for _ in range(6)]
    blobs += [(rng.uniform(0, 320), rng.uniform(0, 240), rng.uniform(3, 8), rng.uniform(-40, 40)) for _ in range(60)]
    return fixed, blobs, rng


def test_quadratic_surface_beats_separate_parabolas_on_a_speckle_pattern():
    """Shift a speckle pattern by 0.5 px in x only: separate x / y parabolas move y as well, the 3 x 3 quadratic does not."""
    from cmtrack.zncc import subpixel_peak, zncc_map
    fixed, blobs, _ = speckle_blobs()
    f0 = cv2.GaussianBlur(blob_scene(_rot(0, (0, 0)), blobs), (0, 0), 1.0)
    f1 = cv2.GaussianBlur(blob_scene(_rot(0, (0.5, 0)), blobs), (0, 0), 1.0)
    sep_err, quad_err = [], []
    for x, y in fixed:
        R = zncc_map(f1[y - 23:y + 24, x - 23:x + 24], extract_patch(f0, (x, y), 31))
        iy, ix = np.unravel_index(int(np.argmax(R)), R.shape)
        sep = (parabolic_offset(R[iy - 1, ix], R[iy, ix], R[iy + 1, ix]), parabolic_offset(R[iy, ix - 1], R[iy, ix], R[iy, ix + 1]))
        quad = subpixel_peak(R, iy, ix)
        truth = (8 - iy, 8.5 - ix)                      # 8 = search half-window: the true offset is (0, +0.5)
        sep_err.append(np.hypot(sep[0] - truth[0], sep[1] - truth[1]))
        quad_err.append(np.hypot(quad[0] - truth[0], quad[1] - truth[1]))
    assert max(sep_err) > 0.1, sep_err
    assert max(quad_err) < 0.01, quad_err


def test_camera_motion_from_fixed_points_on_known_transforms(tmp_path):
    """Frames show one scene through known small rotations + translations; the fixed points recover them."""
    fixed, blobs, rng = speckle_blobs()
    truth =[_rot(0, (0, 0))] + [_rot(r, t, about=(160, 120)) for r, t in
                                 ((0.05, (1.3, -0.4)), (-0.10, (2.7, 1.9)), (0.20, (-3.1, 2.2)), (0.0, (0.45, -2.6)))]
    frames = np.stack([blob_scene(A, blobs) + rng.normal(0, 0.5, (240, 320)).astype(np.float32) for A in truth])
    np.save(tmp_path / "frames.npy", frames.astype(np.float32))
    cfg = load_config(overrides={"input": {"frames": str(tmp_path / "frames.npy")}, "preprocess": {"blur_sigma": 1.0},
                                 "camera": {"model": "rigid", "points_px": [list(p) for p in fixed], "template_px": 31, "search_px": 8}})
    cam, xy, corr = estimate_camera_motion(FrameSource.from_config(cfg), cfg, log=lambda *a: None)
    probe = np.array([[30.0, 30.0], [300.0, 220.0], [160.0, 120.0]])
    for k, A in enumerate(truth):
        err = np.linalg.norm(apply_affine(cam.A[k], probe) - apply_affine(A, probe), axis=1)
        assert err.max() < 0.05, f"frame {k}: {err}"
    assert (cam.n_used == 4).all() and np.nanmin(corr) > 0.95
