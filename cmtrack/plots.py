"""Figures (matplotlib, Agg backend). Labels are in English so that no CJK font is needed."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DPI = 110
C_EST, C_TRUE, C_RAW = "#1f5aa6", "#e07b00", "#9a9a9a"


def _spans(ax, t, mask, color, alpha, label=None):
    """Shade the frames where mask is True (one span per run of consecutive frames)."""
    if mask is None or not np.any(mask):
        return
    dt = np.median(np.diff(t)) if len(t) > 1 else 1.0
    idx = np.where(mask)[0]
    runs = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    for j, r in enumerate(runs):
        ax.axvspan(t[r[0]] - dt / 2, t[r[-1]] + dt / 2, color=color, alpha=alpha, lw=0, label=label if j == 0 else None)


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def timeseries(path, t, theta, dv, mean_corr=None, min_corr=None, dropped=None, interp=None,
               truth_theta=None, truth_dv=None, title=None):
    n = 3 if mean_corr is not None else 2
    fig, axs = plt.subplots(n, 1, figsize=(8.0, 2.1 * n + 0.4), sharex=True)
    for ax, y, yt, lab in ((axs[0], theta * 1e3, None if truth_theta is None else truth_theta * 1e3, "theta (10$^{-3}$ rad)"),
                           (axs[1], dv, truth_dv, "dv (mm)")):
        _spans(ax, t, interp, "#f2c14e", 0.35, "markers interpolated")
        _spans(ax, t, dropped, "#c0392b", 0.25, "frame dropped")
        if yt is not None:
            ax.plot(t, yt, color=C_TRUE, lw=2.4, alpha=0.7, label="ground truth")
        ax.plot(t, y, color=C_EST, lw=1.1, label="tracked")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
    axs[0].legend(loc="upper left", fontsize=8, ncol=4)
    if n == 3:
        ax = axs[2]
        _spans(ax, t, dropped, "#c0392b", 0.25)
        ax.plot(t, mean_corr, color=C_EST, lw=1.0, label="mean ZNCC")
        ax.plot(t, min_corr, color=C_RAW, lw=1.0, label="min ZNCC")
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("ZNCC")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left", fontsize=8, ncol=2)
    axs[-1].set_xlabel("time (s)")
    if title:
        axs[0].set_title(title, fontsize=10)
    return _save(fig, path)


def dv_theta(path, theta, dv, dropped=None, truth_theta=None, truth_dv=None, title=None):
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    if truth_theta is not None:
        ax.plot(truth_theta * 1e3, truth_dv, color=C_TRUE, lw=2.6, alpha=0.6, label="ground truth")
    ax.plot(theta * 1e3, dv, color=C_EST, lw=1.0, label="tracked")
    if dropped is not None and np.any(dropped):
        ax.plot(theta[dropped] * 1e3, dv[dropped], "o", ms=3.5, mfc="none", color="#c0392b", label="dropped -> PCHIP")
    ax.axhline(0, color="k", lw=0.5)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("member angle theta (10$^{-3}$ rad)")
    ax.set_ylabel("axial deformation dv (mm, elongation +)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    return _save(fig, path)


def camera(path, t, dx, dy, rot_deg, truth=None, title=None):
    fig, axs = plt.subplots(3, 1, figsize=(8.0, 5.6), sharex=True)
    for ax, y, key, lab in ((axs[0], dx, "dx_px", "dx (px)"), (axs[1], dy, "dy_px", "dy (px)"), (axs[2], rot_deg, "rot_deg", "rotation (deg)")):
        if truth is not None:
            ax.plot(t, truth[key], color=C_TRUE, lw=2.4, alpha=0.7, label="ground truth")
        ax.plot(t, y, color=C_EST, lw=1.0, label="estimated from fixed points")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=8, loc="upper left")
    axs[-1].set_xlabel("time (s)")
    if title:
        axs[0].set_title(title, fontsize=10)
    return _save(fig, path)


def column_displacements(path, t, dX, dY, names, dropped=None):
    fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.6), sharex=True)
    cmap = plt.get_cmap("viridis")
    for j, n in enumerate(names):
        col = cmap(j / max(1, len(names) - 1))
        axs[0].plot(t, dX[:, j], color=col, lw=0.9, label=n)
        axs[1].plot(t, dY[:, j], color=col, lw=0.9)
    for ax, lab in ((axs[0], "dX rel. to reference row (mm)"), (axs[1], "dY rel. to reference row (mm)")):
        _spans(ax, t, dropped, "#c0392b", 0.2)
        ax.set_xlabel("time (s)")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=6, ncol=2, loc="upper left")
    return _save(fig, path)
