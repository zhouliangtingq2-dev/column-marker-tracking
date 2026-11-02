"""Jump detection and gap filling for tracked series.

jump_mask : residual from a NaN-aware running median (trend-compensated, see running_level); a sample is
            a jump when |x - level| > max(k * sigma, min_abs), sigma = 1.4826 * MAD of the residuals of the
            whole series (a smoothly moving marker follows its running median, a one- or two-frame glitch
            does not)
pchip_fill: shape-preserving cubic (PCHIP) interpolation over NaN gaps; ends are held at the nearest value
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.interpolate import PchipInterpolator


def running_nanmedian(x: np.ndarray, window: int) -> np.ndarray:
    """NaN-aware centred running median along axis 0 (window forced odd).

    Near the ends the window shrinks symmetrically, so a linear trend is followed exactly up to the ends.
    """
    x = np.asarray(x, dtype=float)
    w = int(window) | 1
    h, T = w // 2, len(x)
    out = np.array(x, dtype=float, copy=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if T >= w:
            win = np.lib.stride_tricks.sliding_window_view(x, w, axis=0)
            out[h:T - h] = np.nanmedian(win, axis=-1)
        for k in list(range(min(h, T))) + list(range(max(h, T - h), T)):
            r = min(k, T - 1 - k, h)
            out[k] = np.nanmedian(x[k - r:k + r + 1], axis=0)
    return out


def running_level(x: np.ndarray, window: int) -> np.ndarray:
    """Trend-compensated running median along axis 0.

    For frame k: s = median of the first differences inside the window, then
    level_k = median_j (x_j + s (k - j)) over the window. On a linear trend this equals x_k even when samples
    are missing on one side, and a glitch inside the window neither moves the slope nor the level.
    """
    x = np.asarray(x, dtype=float)
    T = len(x)
    h = (int(window) | 1) // 2
    d = np.diff(x, axis=0)
    out = np.empty_like(x)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for k in range(T):
            a, b = max(0, k - h), min(T, k + h + 1)
            s = np.nan_to_num(np.nanmedian(d[a:b - 1], axis=0)) if b - a > 1 else np.zeros(x.shape[1:])
            lag = (k - np.arange(a, b)).reshape((-1,) + (1,) * (x.ndim - 1))
            out[k] = np.nanmedian(x[a:b] + s * lag, axis=0)
    return out


def robust_sigma(r: np.ndarray, axis=0) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(r, axis=axis, keepdims=True)
        return 1.4826 * np.nanmedian(np.abs(r - med), axis=axis)


def jump_mask(x: np.ndarray, window: int, k: float, min_abs: float) -> np.ndarray:
    """Samples that jump away from the trend-compensated running median (each column separately)."""
    x = np.asarray(x, dtype=float)
    r = x - running_level(x, window)
    thr = np.maximum(k * robust_sigma(r), min_abs)
    with np.errstate(invalid="ignore"):
        return np.abs(r) > thr


def pchip_fill(x: np.ndarray):
    """Fill NaNs along axis 0 (each column separately). Returns (filled, was_nan)."""
    x = np.asarray(x, dtype=float)
    flat = x.reshape(len(x), -1).copy()
    t = np.arange(len(x))
    for j in range(flat.shape[1]):
        y = flat[:, j]
        good = np.isfinite(y)
        if good.all() or good.sum() == 0:
            continue
        if good.sum() == 1:
            y[~good] = y[good][0]
            continue
        f = PchipInterpolator(t[good], y[good], extrapolate=False)
        y[~good] = f(t[~good])
        first, last = np.argmax(good), len(y) - 1 - np.argmax(good[::-1])
        y[:first], y[last + 1:] = y[first], y[last]
    return flat.reshape(x.shape), ~np.isfinite(x)
