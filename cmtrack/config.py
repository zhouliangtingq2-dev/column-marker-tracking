"""Settings: defaults + one or more YAML/JSON files merged on top of each other.

Relative paths (``input.frames``, ``output.dir``) are resolved against the folder of the file
that sets them, so a config file can live next to the frames it describes.
"""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "input": {
        "frames": None,       # folder of images | .npy stack (T,H,W) | video file
        "fps": 30.0,          # used for image sequences / .npy (videos report their own fps)
        "start": 0,
        "stop": None,
        "step": 1,
        "rotate": 0,          # 0 / 90 / 180 / 270 degrees clockwise, applied on read
    },
    "preprocess": {
        "blur_sigma": 1.0,    # Gaussian blur (px) before matching; templates are cut from the blurred frame 0
    },
    "camera": {
        "enabled": True,
        "model": "rigid",        # translation (>= 1 point) | rigid: rotation + translation | similarity: + scale
        "points_px": [],         # fixed points in frame 0 (corners on the stub etc.), [[x, y], ...]
        "template_px": 41,
        "search_px": 16,
        "rescue_scale": 3.0,
        "min_corr": 0.6,
        "max_resid_px": 1.0,
    },
    "scale": {
        "mode": "affine",        # rectangle | affine | homography | spacing
        "width_mm": 300.0,       # known rectangle (column face) for rectangle / affine / homography
        "height_mm": 1100.0,
        "corners_px": None,      # {LB: [x, y], LT: [...], RB: [...], RT: [...]} in frame 0
        "spacing_mm": [100.0, 100.0],   # spacing mode: marker pitch (horizontal, vertical)
    },
    "markers": {
        "cols": 3,
        "rows": 11,
        "pattern_b_rows": [6, 7, 8, 9, 10, 11],   # rows (bottom = 1) whose upper-left / lower-right squares are black
        "marker_px": 30,          # approximate printed marker size (2 x 2 squares) in px
        "template_px": 41,        # tracking template (odd)
        "search_px": 12,          # half width of the search window around the prediction
        "init": {
            "mode": "corners",    # corners (4 corner markers, grid interpolated) | points (all markers)
            "corners_px": None,   # {bottom_left: [x, y], top_left: [...], bottom_right: [...], top_right: [...]}
            "points_px": None,    # [[x, y], ...] column-major from bottom-left (c1r01, c1r02, ...)
            "search_px": 10,
            "min_corr": 0.5,
        },
    },
    "tracking": {
        "corr_accept": 0.70,
        "corr_rescue": 0.50,
        "rescue_scale": 3.0,
        "max_step_px": 10.0,      # allowed distance between prediction and match (camera-corrected)
        "velocity_gain": 0.7,
        "max_velocity_px": 10.0,
        "template_update": "none",  # none (frame-0 template, no drift) | ema (slow update for changing appearance)
        "ema_alpha": 0.03,
        "update_min_corr": 0.90,
        "pattern_min_corr": 0.7,  # rescued matches must show the expected checker pattern at the match
        "neighbor_tol": 0.2,      # rescued matches: vector to the vertical neighbour within 20 % of its frame-0 length
        "frame_max_lost_ratio": 0.4,
        "frame_min_mean_corr": 0.5,
        "step_spike_k": 5.0,
        "step_spike_min_px": 5.0,
    },
    "cleaning": {
        "min_corr": 0.85,         # matches below this ZNCC are not used (filled by PCHIP like lost markers)
        "hampel_window": 7,       # running-median window for marker coordinates (frames, odd)
        "hampel_k": 6.0,
        "hampel_min_px": 1.5,
        "frame_jump_markers": 4,  # this many marker jumps in one frame -> drop the whole frame
        "series_window": 5,       # running-median window for theta / dv
        "series_k": 6.0,
        "series_min_theta_rad": 0.002,
        "series_min_dv_mm": 0.2,
        "drop_ranges_s": [],      # [[t0, t1], ...] intervals to drop by hand (e.g. someone in front of the camera)
    },
    "deformation": {
        "theta_column": 2,        # 1..cols, or "mean"
        "dv_column": 2,
        "dv_top_rows": [10, 11],
        "dv_bottom_rows": [1, 2],
        "reference_row": 1,       # displacements are also reported relative to this row
        "dv_sign": "elongation",  # elongation (+) | shortening (+)
    },
    "output": {
        "dir": "runs/out",
        "plots": True,
    },
}

PATH_KEYS = [("input", "frames"), ("output", "dir")]


def deep_merge(base: dict, new: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (new or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _resolve_paths(d: dict, folder: Path) -> dict:
    for sec, key in PATH_KEYS:
        v = d.get(sec, {}).get(key) if isinstance(d.get(sec), dict) else None
        if v:
            p = Path(v).expanduser()
            d[sec][key] = str(p if p.is_absolute() else (folder / p).resolve())
    return d


def load_config(paths=(), overrides: dict | None = None) -> dict:
    """Defaults <- each file in ``paths`` (YAML or JSON) <- ``overrides``."""
    cfg = copy.deepcopy(DEFAULTS)
    for p in paths:
        p = Path(p)
        with p.open(encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        cfg = deep_merge(cfg, _resolve_paths(d, p.resolve().parent))
    if overrides:
        cfg = deep_merge(cfg, overrides)
    check(cfg)
    return cfg


def check(cfg: dict) -> None:
    if cfg["camera"]["model"] not in ("translation", "rigid", "similarity"):
        raise ValueError("camera.model must be 'translation', 'rigid' or 'similarity'")
    if cfg["scale"]["mode"] not in ("rectangle", "affine", "homography", "spacing"):
        raise ValueError("scale.mode must be rectangle / affine / homography / spacing")
    if cfg["markers"]["init"]["mode"] not in ("corners", "points"):
        raise ValueError("markers.init.mode must be 'corners' or 'points'")
    if cfg["tracking"]["template_update"] not in ("ema", "none"):
        raise ValueError("tracking.template_update must be 'ema' or 'none'")
    if cfg["deformation"]["dv_sign"] not in ("elongation", "shortening"):
        raise ValueError("deformation.dv_sign must be 'elongation' or 'shortening'")


def dump(cfg: dict, path: Path) -> None:
    Path(path).write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
