"""Small CSV helpers (header row + columns; NaN written as 'nan'). No pandas needed."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def _fmt(v) -> str:
    if isinstance(v, (str, bytes)):
        return v if isinstance(v, str) else v.decode()
    if isinstance(v, (bool, np.bool_)):
        return str(int(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    v = float(v)
    return "nan" if not np.isfinite(v) else f"{v:.10g}"


def write_table(path, columns: dict) -> None:
    """``columns``: {name: 1-D sequence}; all columns must have the same length."""
    names = list(columns)
    cols = [np.asarray(columns[n]) if not isinstance(columns[n], list) else columns[n] for n in names]
    n = len(cols[0]) if cols else 0
    for name, c in zip(names, cols):
        if len(c) != n:
            raise ValueError(f"column {name} has length {len(c)}, expected {n}")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(names)
        for i in range(n):
            w.writerow([_fmt(c[i]) for c in cols])


def read_table(path) -> dict:
    """Returns {name: np.ndarray}; numeric columns as float, others as str arrays."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    names, body = rows[0], rows[1:]
    out = {}
    for j, name in enumerate(names):
        col = [r[j] if j < len(r) else "" for r in body]
        try:
            out[name] = np.array([float(v) if v != "" else np.nan for v in col])
        except ValueError:
            out[name] = np.array(col, dtype=object)
    return out


def wide(prefix_names: list, arr: np.ndarray, suffixes=("x", "y")) -> dict:
    """(T, N, 2) -> {name_x: (T,), name_y: (T,), ...}; (T, N) -> {name: (T,)}."""
    out = {}
    if arr.ndim == 3:
        for i, n in enumerate(prefix_names):
            for j, s in enumerate(suffixes):
                out[f"{n}_{s}"] = arr[:, i, j]
    else:
        for i, n in enumerate(prefix_names):
            out[n] = arr[:, i]
    return out


def unwide(table: dict, names: list, suffixes=("x", "y")) -> np.ndarray:
    if suffixes:
        return np.stack([np.stack([table[f"{n}_{s}"] for s in suffixes], axis=1) for n in names], axis=1)
    return np.stack([table[n] for n in names], axis=1)
