"""Synthetic test video of an RC column face with 3 x 11 checker markers and known ground truth.

No real experiment data is used. Scene in face coordinates (mm, X right, Y up, origin = lower-left
corner of the column face):
  * column face 300 x 1100 mm, painted, faint 100 mm grid, 33 checker markers (40 mm, on 52 mm stickers);
    rows 1-5 pattern A (upper-right / lower-left black), rows 6-11 pattern B
  * stub (base) below, wider than the column, with two anchor nuts: the fixed points
  * loading beam above (moves with the column top); dark textured background
Column deformation (H = 1100 mm, s = Y / H, Xc = 150 mm):
  lateral  u = delta(t) (3 s^2 - 2 s^3)                 double curvature, drift angle R = delta / H
  vertical v = a(t) (s + 0.5 sin(2 pi s) / (2 pi)) - (X - Xc) phi,  phi = du/dY
  R(t): one sine cycle each at 1/400, 1/200, 1/100, 1/50 rad;
  a(t) = preload + c2 R^2 + c1 R + c_res max|R| (residual elongation grows with the largest drift so far)
Camera: orthographic (0.75 px/mm), per-frame translation (random walk + 2 Hz vibration) and a small roll
  about the image centre, plus one "bump" (permanent jump, motion-blurred frame).
Rendering: textures are rasterised at 2 texels/mm and pre-blurred with the camera PSF; every frame samples
  them through the exact inverse of camera + deformation (fixed-point iteration). Illumination flicker,
  vignetting and signal-dependent noise are added and the frame is quantised to 8 bit.
Injected anomalies: occlusion (4 frames), camera bump (1), glare (2), tearing of a band of rows (1).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml
from scipy.special import erf

from . import deform
from .csvio import wide, write_table
from .layout import Grid

H_COL, W_COL, XC = 1100.0, 300.0, 150.0
STUB_X = (-150.0, 450.0)
BEAM_X = (-150.0, 450.0)
NUTS = ((-100.0, -60.0), (400.0, -60.0))
GAMMA = 0.5


@dataclass
class SynthConfig:
    width: int = 600
    height: int = 1000
    n_frames: int = 180
    fps: float = 30.0
    seed: int = 0
    px_per_mm: float = 0.75
    tex_per_mm: float = 2.0
    psf_sigma_px: float = 0.8
    marker_cols_mm: tuple = (50.0, 150.0, 250.0)
    marker_rows_mm: tuple = tuple(50.0 + 100.0 * i for i in range(11))
    marker_mm: float = 40.0
    sticker_mm: float = 52.0
    pattern_b_rows: tuple = (6, 7, 8, 9, 10, 11)
    theta_levels: tuple = (1 / 400, 1 / 200, 1 / 100, 1 / 50)
    axial_preload_mm: float = -1.0
    axial_c2_mm: float = 4000.0            # elongation ~ R^2 (crack opening)
    axial_c1_mm: float = 15.0              # ~ R (e.g. varying axial force)
    axial_res_mm: float = 30.0             # residual elongation ~ largest |R| so far
    shake_px: float = 1.2
    vibration_px: float = 0.5
    roll_deg: float = 0.02
    bump: tuple = (5.0, -3.0, 0.08)        # dx px, dy px, roll deg
    noise_read: float = 1.5
    noise_shot: float = 0.02
    anomalies: bool = True
    anomaly_frac: dict = field(default_factory=lambda: {"occlusion": 0.345, "camera_bump": 0.528, "glare": 0.656, "tearing": 0.778})


# ----------------------------------------------------------------------------- kinematics
def smoothstep(u):
    v = np.clip(u + 0.5, 0.0, 1.0)
    return v * v * (3 - 2 * v)


def history(cfg: SynthConfig):
    """Drift angle R(t), lateral delta(t) [mm] and axial a(t) [mm] for every frame.

    One sine cycle per amplitude level; the amplitude envelope changes smoothly around the zero crossings
    between cycles, so the motion has no velocity jumps (like a slowly loaded specimen, only compressed in time).
    """
    n = cfg.n_frames
    k = np.arange(n, dtype=float)
    ramp_end = max(2, round(0.06 * n))
    k0 = round(0.08 * n)
    levels = list(cfg.theta_levels)
    L = (n - k0 - round(0.07 * n)) / len(levels)
    tau = 0.3 * L
    env = np.zeros(n)
    for j, (a_prev, a_new) in enumerate(zip([0.0] + levels, levels + [0.0])):
        env += (a_new - a_prev) * smoothstep((k - k0 - j * L) / tau)
    R = env * np.sin(2 * np.pi * (k - k0) / L)
    R[np.abs(env) < 1e-12] = 0.0
    ramp = 0.5 * (1 - np.cos(np.pi * np.clip(k / ramp_end, 0, 1)))
    a = cfg.axial_preload_mm * ramp + cfg.axial_c2_mm * R ** 2 + cfg.axial_c1_mm * R + cfg.axial_res_mm * np.maximum.accumulate(np.abs(R))
    return R, R * H_COL, a


def displacement(X0, Y0, delta: float, a: float):
    s = np.clip(Y0 / H_COL, 0.0, 1.0)
    u = delta * (3 * s ** 2 - 2 * s ** 3)
    phi = delta / H_COL * 6 * s * (1 - s)
    v = a * (s + GAMMA * np.sin(2 * np.pi * s) / (2 * np.pi)) - (X0 - XC) * phi
    return u, v


def inverse_deformation(X, Y, delta: float, a: float, iters: int = 5):
    X0, Y0 = X.copy(), Y.copy()
    for _ in range(iters):
        u, v = displacement(X0, Y0, delta, a)
        X0, Y0 = X - u, Y - v
    return X0, Y0


def camera_history(cfg: SynthConfig, rng):
    n = cfg.n_frames
    t = np.arange(n) / cfg.fps

    def ou(sigma, rho):
        x = np.zeros(n)
        x[0] = sigma * rng.standard_normal()
        for i in range(1, n):
            x[i] = rho * x[i - 1] + sigma * math.sqrt(1 - rho ** 2) * rng.standard_normal()
        return x
    ph = rng.uniform(0, 2 * np.pi, 2)
    tx = ou(cfg.shake_px, 0.92) + cfg.vibration_px * np.sin(2 * np.pi * 2.0 * t + ph[0])
    ty = ou(cfg.shake_px, 0.92) + cfg.vibration_px * np.sin(2 * np.pi * 2.0 * t + ph[1])
    rot = ou(cfg.roll_deg, 0.97)
    kb = anomaly_frames(cfg).get("camera_bump", [None])[0]
    if kb is not None:
        tx[kb:] += cfg.bump[0]
        ty[kb:] += cfg.bump[1]
        rot[kb:] += cfg.bump[2]
    return tx, ty, rot


def anomaly_frames(cfg: SynthConfig) -> dict:
    if not cfg.anomalies:
        return {}
    n, f = cfg.n_frames, cfg.anomaly_frac
    k = {name: int(round(v * n)) for name, v in f.items()}
    return {"occlusion": [k["occlusion"] + i for i in range(4)], "camera_bump": [k["camera_bump"]],
            "glare": [k["glare"], k["glare"] + 1], "tearing": [k["tearing"]]}


# ----------------------------------------------------------------------------- scene
class Raster:
    def __init__(self, x0, x1, y0, y1, res):
        self.x0, self.y1, self.res = x0, y1, res
        self.w, self.h = int(round((x1 - x0) * res)), int(round((y1 - y0) * res))
        self.X = (x0 + (np.arange(self.w) + 0.5) / res)[None, :].astype(np.float32)
        self.Y = (y1 - (np.arange(self.h) + 0.5) / res)[:, None].astype(np.float32)
        self.img = np.zeros((self.h, self.w), np.float32)

    def sample(self, X, Y):
        mx = ((X - self.x0) * self.res - 0.5).astype(np.float32)
        my = ((self.y1 - Y) * self.res - 0.5).astype(np.float32)
        return cv2.remap(self.img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def smooth_noise(shape, rng, cell: float) -> np.ndarray:
    h, w = shape
    gh, gw = max(2, int(h / cell) + 2), max(2, int(w / cell) + 2)
    g = rng.standard_normal((gh, gw)).astype(np.float32)
    n = cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
    return n / (n.std() + 1e-9)


def fine_noise(shape, rng, sigma: float) -> np.ndarray:
    n = rng.standard_normal(shape).astype(np.float32)
    n = cv2.GaussianBlur(n, (0, 0), sigma)
    return n / (n.std() + 1e-9)


class Scene:
    def __init__(self, cfg: SynthConfig, rng):
        self.cfg = cfg
        r, s = cfg.tex_per_mm, cfg.px_per_mm
        W, H = cfg.width, cfg.height
        self.u0 = W / 2 - s * XC
        self.v0 = H / 2 + s * (H_COL / 2)
        self.c = np.array([(W - 1) / 2, (H - 1) / 2])
        blur = cfg.psf_sigma_px * r / s

        st = Raster(-330, 630, -200, 1300, r)
        X, Y = st.X, st.Y
        st.img[:] = 70 + 8 * smooth_noise(st.img.shape, rng, 200) + 3 * smooth_noise(st.img.shape, rng, 20) + 2 * fine_noise(st.img.shape, rng, 1.0)
        stub = np.broadcast_to((X >= STUB_X[0]) & (X < STUB_X[1]) & (Y < 0), st.img.shape)
        tex = 140 + 7 * smooth_noise(st.img.shape, rng, 60) + 5 * fine_noise(st.img.shape, rng, 1.2)
        st.img[stub] = tex[stub]
        for nx, ny in NUTS:
            washer = (X - nx) ** 2 + (Y - ny) ** 2 < 24.0 ** 2
            st.img[washer] = 178.0
            nut = (np.abs(X - nx) < 15.0) & (np.abs(Y - ny) < 15.0)
            st.img[nut] = 48.0
        st.img[:] = cv2.GaussianBlur(st.img, (0, 0), blur)
        self.static = st

        sp = Raster(-200, 500, -40, 1300, r)
        X, Y = sp.X, sp.Y
        sp.img[:] = 192 + 4 * smooth_noise(sp.img.shape, rng, 80) + 3 * fine_noise(sp.img.shape, rng, 1.0)
        col = (X >= 0) & (X < W_COL) & (Y >= 0) & (Y < H_COL)
        lines = np.zeros(sp.img.shape, bool)
        for xl in (100.0, 200.0):
            lines |= np.broadcast_to(np.abs(X - xl) < 0.5, sp.img.shape)
        for yl in np.arange(100.0, 1001.0, 100.0):
            lines |= np.broadcast_to(np.abs(Y - yl) < 0.5, sp.img.shape)
        sp.img[lines & col] -= 35.0
        grid = Grid(len(cfg.marker_cols_mm), len(cfg.marker_rows_mm))
        hs, hm = cfg.sticker_mm / 2, cfg.marker_mm / 2
        for i in range(grid.n):
            c, rr = grid.cr(i)
            xm, ym = cfg.marker_cols_mm[c - 1], cfg.marker_rows_mm[rr - 1]
            dx, dy = X - xm, Y - ym
            sticker = (np.abs(dx) < hs) & (np.abs(dy) < hs)
            sp.img[sticker] = 232.0
            sq = (np.abs(dx) < hm) & (np.abs(dy) < hm)
            black = sq & ((dx * dy < 0) if rr in cfg.pattern_b_rows else (dx * dy > 0))   # face Y is up
            sp.img[black] = 30.0
        beam = np.broadcast_to(Y >= H_COL, sp.img.shape)
        btex = 128 + 7 * smooth_noise(sp.img.shape, rng, 60) + 4 * fine_noise(sp.img.shape, rng, 1.2)
        sp.img[beam] = btex[beam]
        sp.img[np.broadcast_to((Y >= H_COL) & (Y < H_COL + 4), sp.img.shape)] -= 30.0
        sp.img[:] = cv2.GaussianBlur(sp.img, (0, 0), blur)
        self.spec = sp

        step = 4
        self.step = step
        self.uc = np.arange(0, W - 1 + step, step, dtype=np.float64)
        self.vc = np.arange(0, H - 1 + step, step, dtype=np.float64)
        self.fx = np.broadcast_to((np.arange(W) / step).astype(np.float32)[None, :], (H, W)).copy()
        self.fy = np.broadcast_to((np.arange(H) / step).astype(np.float32)[:, None], (H, W)).copy()
        uu, vv = np.meshgrid(np.arange(W), np.arange(H))
        r2 = ((uu - self.c[0]) ** 2 + (vv - self.c[1]) ** 2) / (self.c[0] ** 2 + self.c[1] ** 2)
        self.vignette = (1.0 - 0.12 * r2).astype(np.float32)

    # camera: p = R(rot) (p_nom - c) + c + t
    def to_image(self, X, Y, tx, ty, rot_deg):
        s = self.cfg.px_per_mm
        un, vn = self.u0 + s * X, self.v0 - s * Y
        ca, sa = math.cos(math.radians(rot_deg)), math.sin(math.radians(rot_deg))
        du, dv = un - self.c[0], vn - self.c[1]
        return ca * du - sa * dv + self.c[0] + tx, sa * du + ca * dv + self.c[1] + ty

    def camera_matrix(self, tx, ty, rot_deg) -> np.ndarray:
        """3 x 3 matrix of p_nominal -> p_image."""
        ca, sa = math.cos(math.radians(rot_deg)), math.sin(math.radians(rot_deg))
        R = np.array([[ca, -sa], [sa, ca]])
        M = np.eye(3)
        M[:2, :2] = R
        M[:2, 2] = self.c - R @ self.c + np.array([tx, ty])
        return M

    def render(self, tx, ty, rot_deg, delta, a) -> np.ndarray:
        s, cfg = self.cfg.px_per_mm, self.cfg
        U, V = np.meshgrid(self.uc, self.vc)
        ca, sa = math.cos(math.radians(rot_deg)), math.sin(math.radians(rot_deg))
        du, dv = U - self.c[0] - tx, V - self.c[1] - ty
        un, vn = ca * du + sa * dv + self.c[0], -sa * du + ca * dv + self.c[1]
        X, Y = (un - self.u0) / s, (self.v0 - vn) / s
        X0, Y0 = inverse_deformation(X, Y, delta, a)
        up = [cv2.remap(m.astype(np.float32), self.fx, self.fy, cv2.INTER_LINEAR) for m in (X, Y, X0, Y0)]
        X, Y, X0, Y0 = up
        bg = self.static.sample(X, Y)
        fg = self.spec.sample(X0, Y0)
        d1 = np.minimum(np.minimum(X0, W_COL - X0), np.minimum(Y0, H_COL + 50.0 - Y0))
        d2 = np.minimum(np.minimum(X0 - BEAM_X[0], BEAM_X[1] - X0), Y0 - H_COL)
        dist_px = np.maximum(d1, d2) * s
        alpha = 0.5 * (1.0 + erf(dist_px / (math.sqrt(2.0) * cfg.psf_sigma_px)))
        return (alpha * fg + (1.0 - alpha) * bg).astype(np.float32)


def marker_reference(cfg: SynthConfig, grid: Grid) -> np.ndarray:
    return np.array([(cfg.marker_cols_mm[grid.cr(i)[0] - 1], cfg.marker_rows_mm[grid.cr(i)[1] - 1]) for i in range(grid.n)])


def capsule_mask(shape, p1, p2, radius) -> np.ndarray:
    h, w = shape
    vv, uu = np.mgrid[0:h, 0:w].astype(np.float32)
    p1, p2 = np.asarray(p1, np.float32), np.asarray(p2, np.float32)
    d = p2 - p1
    tt = np.clip(((uu - p1[0]) * d[0] + (vv - p1[1]) * d[1]) / float(d @ d), 0, 1)
    dist = np.hypot(uu - (p1[0] + tt * d[0]), vv - (p1[1] + tt * d[1]))
    return np.clip(radius - dist + 0.5, 0, 1)


# ----------------------------------------------------------------------------- generator
def generate(out_dir, cfg: SynthConfig | None = None, log=print) -> dict:
    cfg = cfg or SynthConfig()
    out = Path(out_dir)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    for old in (out / "frames").glob("*.png"):
        old.unlink()
    rng = np.random.default_rng(cfg.seed)
    grid = Grid(len(cfg.marker_cols_mm), len(cfg.marker_rows_mm))
    names = grid.names()
    scene = Scene(cfg, rng)
    R, delta, a = history(cfg)
    tx, ty, rot = camera_history(cfg, rng)
    anom = anomaly_frames(cfg)
    anom_of = {k: name for name, ks in anom.items() for k in ks}
    n, W, H = cfg.n_frames, cfg.width, cfg.height
    ref_mm = marker_reference(cfg, grid)

    mk_mm = np.zeros((n, grid.n, 2))
    mk_px = np.zeros((n, grid.n, 2))
    cam_used = np.zeros((n, 3))
    t = np.arange(n) / cfg.fps
    for k in range(n):
        u, v = displacement(ref_mm[:, 0], ref_mm[:, 1], delta[k], a[k])
        mk_mm[k] = np.stack([ref_mm[:, 0] + u, ref_mm[:, 1] + v], axis=1)
        if anom_of.get(k) == "camera_bump" and k > 0:       # motion blur: average of sub-frames along the jump
            ws = np.linspace(0.1, 0.9, 5)
            subs = [((1 - w) * tx[k - 1] + w * tx[k], (1 - w) * ty[k - 1] + w * ty[k], (1 - w) * rot[k - 1] + w * rot[k]) for w in ws]
        else:
            subs = [(tx[k], ty[k], rot[k])]
        img = np.mean([scene.render(*cp, delta[k], a[k]) for cp in subs], axis=0)
        pts = np.mean([np.stack(scene.to_image(mk_mm[k, :, 0], mk_mm[k, :, 1], *cp), axis=1) for cp in subs], axis=0)
        mk_px[k] = pts
        cam_used[k] = np.mean(subs, axis=0)

        gain = 1.0 + 0.03 * math.sin(2 * math.pi * t[k] / 4.0) + 0.012 * rng.standard_normal()
        img = img * scene.vignette * gain
        kind = anom_of.get(k)
        if kind == "occlusion":
            j = anom["occlusion"].index(k)
            uc = 200 + 28 * j
            m = capsule_mask((H, W), (uc - 40, 0.74 * H), (uc + 10, 0.33 * H), 26)
            img = img * (1 - m) + (58 + 8 * fine_noise((H, W), rng, 2.0)) * m
        elif kind == "glare":
            c = mk_px[k, grid.idx(3, 9)] + np.array([8.0, -6.0])
            vv, uu = np.mgrid[0:H, 0:W]
            img = img + 300.0 * np.exp(-0.5 * (((uu - c[0]) / 26.0) ** 2 + ((vv - c[1]) / 16.0) ** 2))
        elif kind == "tearing":
            a0, a1, sh = int(0.38 * H), int(0.64 * H), 9
            band = img[a0:a1].copy()
            img[a0:a1, sh:] = band[:, :-sh]
            img[a0:a1, :sh] = band[:, :1]
        sigma = np.sqrt(cfg.noise_read ** 2 + cfg.noise_shot * np.clip(img, 0, None))
        img = img + sigma * rng.standard_normal(img.shape).astype(np.float32)
        cv2.imwrite(str(out / "frames" / f"frame_{k:04d}.png"), np.clip(np.rint(img), 0, 255).astype(np.uint8),
                    [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if k % max(1, n // 6) == 0:
            log(f"[synth] frame {k}/{n - 1}")

    # ---------------- ground truth
    M0inv = np.linalg.inv(scene.camera_matrix(*cam_used[0]))
    rel = np.array([scene.camera_matrix(*cam_used[k]) @ M0inv for k in range(n)])
    fixed_mm = np.array([[STUB_X[0], 0.0], [STUB_X[1], 0.0], list(NUTS[0]), list(NUTS[1])])
    fixed0 = np.stack(scene.to_image(fixed_mm[:, 0], fixed_mm[:, 1], *cam_used[0]), axis=1)
    ref = fixed0.mean(axis=0)
    d_ref = np.einsum("kij,j->ki", rel[:, :2, :2], ref) + rel[:, :2, 2] - ref
    th = deform.theta_column(mk_mm, grid, 2)
    L = deform.gauge_length(mk_mm, grid, 2, (10, 11), (1, 2))
    write_table(out / "truth.csv", {
        "frame": np.arange(n), "t_s": t, "drift_R_rad": R, "theta_rad": th, "dv_mm": L - L[0],
        "delta_mm": delta, "axial_a_mm": a,
        "cam_tx_px": cam_used[:, 0], "cam_ty_px": cam_used[:, 1], "cam_rot_deg": cam_used[:, 2],
        "cam_dx_px": d_ref[:, 0], "cam_dy_px": d_ref[:, 1], "cam_rot_rel_deg": cam_used[:, 2] - cam_used[0, 2],
        "A_a": rel[:, 0, 0], "A_b": rel[:, 1, 0], "A_tx": rel[:, 0, 2], "A_ty": rel[:, 1, 2],
        "anomaly": [anom_of.get(k, "") for k in range(n)]})
    write_table(out / "truth_markers_px.csv", {"frame": np.arange(n), "t_s": t, **wide(names, mk_px)})
    write_table(out / "truth_markers_mm.csv", {"frame": np.arange(n), "t_s": t, **wide(names, mk_mm, ("X", "Y"))})

    # ---------------- config for the pipeline: what a user would click on frame 0
    corners_mm = {"LB": (0.0, 0.0), "LT": (0.0, H_COL), "RB": (W_COL, 0.0), "RT": (W_COL, H_COL)}
    corners = {kk: [round(float(c), 1) for c in scene.to_image(np.array(x), np.array(y), *cam_used[0])] for kk, (x, y) in corners_mm.items()}
    click_err = {"bottom_left": (2.6, -1.8), "top_left": (-2.2, 2.4), "bottom_right": (1.9, 2.2), "top_right": (-2.7, -1.5)}
    which = {"bottom_left": (1, 1), "top_left": (1, grid.rows), "bottom_right": (grid.cols, 1), "top_right": (grid.cols, grid.rows)}
    mcorners = {kk: [float(round(mk_px[0, grid.idx(*which[kk]), j] + click_err[kk][j])) for j in range(2)] for kk in which}
    config = {
        "input": {"frames": "frames", "fps": cfg.fps},
        "preprocess": {"blur_sigma": 1.0},
        "camera": {"enabled": True, "model": "rigid", "template_px": 41, "search_px": 16,
                   "points_px": [[float(round(x)), float(round(y))] for x, y in fixed0]},
        "scale": {"mode": "affine", "width_mm": W_COL, "height_mm": H_COL, "corners_px": corners},
        "markers": {"cols": grid.cols, "rows": grid.rows, "pattern_b_rows": list(cfg.pattern_b_rows),
                    "marker_px": int(round(cfg.marker_mm * cfg.px_per_mm)), "template_px": 41, "search_px": 12,
                    "init": {"mode": "corners", "corners_px": mcorners, "search_px": 10}},
        "output": {"dir": "../run"},
    }
    head = ("# Generated by scripts/00_make_synthetic.py for the synthetic demo video.\n"
            "# The points are what a user would click on frame 0 (fixed points on the stub, column-face corners,\n"
            "# the four corner markers); see configs/example.yaml for every available key.\n")
    (out / "config.yaml").write_text(head + yaml.safe_dump(config, sort_keys=False, default_flow_style=None), encoding="utf-8")
    log(f"[synth] wrote {n} frames ({W}x{H}) + truth to {out}")
    return {"config": out / "config.yaml", "anomalies": anom}


# ----------------------------------------------------------------------------- small test images
def _coverage(c: np.ndarray, a: float, b: float) -> np.ndarray:
    """Overlap of the pixels [c - 0.5, c + 0.5] with the interval [a, b]."""
    return np.clip(np.minimum(c + 0.5, b) - np.maximum(c - 0.5, a), 0.0, 1.0)


def checker_image(shape, center, square_px: float, pattern: str = "A", angle_deg: float = 0.0, psf_sigma: float = 0.8,
                  ss: int = 16, bg: float = 190.0, black: float = 30.0, white: float = 230.0, sticker_px: float | None = None):
    """One 2 x 2 checker at a sub-pixel position, integrated over each pixel area, then a Gaussian PSF.

    Image coordinates (y down); pattern A = upper-right / lower-left black. Exact pixel coverage when the
    checker is axis-aligned, ss x ss super-sampling when it is rotated. Independent of ``Scene`` so that the
    unit tests check the matcher against a different renderer.
    """
    h, w = shape
    if angle_deg == 0.0:
        cx, cy, s = float(center[0]), float(center[1]), float(square_px)
        xs, ys = np.arange(w, dtype=float), np.arange(h, dtype=float)

        def rect(x0, x1, y0, y1):
            return _coverage(ys, y0, y1)[:, None] * _coverage(xs, x0, x1)[None, :]
        half = (sticker_px or 2 * s) / 2
        img = bg + (white - bg) * rect(cx - half, cx + half, cy - half, cy + half)
        if pattern == "A":
            blk = rect(cx, cx + s, cy - s, cy) + rect(cx - s, cx, cy, cy + s)
        else:
            blk = rect(cx - s, cx, cy - s, cy) + rect(cx, cx + s, cy, cy + s)
        img = (img + (black - white) * blk).astype(np.float32)
        return cv2.GaussianBlur(img, (0, 0), psf_sigma) if psf_sigma > 0 else img
    ys = (np.arange(h * ss) + 0.5) / ss - 0.5
    xs = (np.arange(w * ss) + 0.5) / ss - 0.5
    X, Y = np.meshgrid(xs, ys)
    ca, sa = math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))
    dx, dy = X - center[0], Y - center[1]
    lx, ly = ca * dx + sa * dy, -sa * dx + ca * dy
    img = np.full(X.shape, bg, np.float32)
    if sticker_px:
        img[(np.abs(lx) < sticker_px / 2) & (np.abs(ly) < sticker_px / 2)] = white
    sq = (np.abs(lx) < square_px) & (np.abs(ly) < square_px)
    img[sq] = white
    img[sq & ((lx * ly < 0) if pattern == "A" else (lx * ly > 0))] = black
    img = img.reshape(h, ss, w, ss).mean(axis=(1, 3))
    return cv2.GaussianBlur(img, (0, 0), psf_sigma) if psf_sigma > 0 else img
