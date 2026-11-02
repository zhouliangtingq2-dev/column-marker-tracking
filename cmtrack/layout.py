"""Marker grid (3 columns x 11 rows by default), marker IDs and quadrant (2 x 2 checker) templates.

IDs are column-major from the bottom-left marker: c1r01, c1r02, ..., c1r11, c2r01, ..., c3r11
(index = (c - 1) * rows + (r - 1); rows are counted from the bottom).

Two checker patterns are used on the column face:
  A : upper-right and lower-left squares black
  B : upper-left and lower-right squares black
The quadrant template has its exact centre on the X-junction, so matching it gives the marker centre.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Grid:
    cols: int = 3
    rows: int = 11

    @property
    def n(self) -> int:
        return self.cols * self.rows

    def idx(self, c: int, r: int) -> int:
        """1-based column / row (bottom = row 1) -> 0-based marker index."""
        return (c - 1) * self.rows + (r - 1)

    def cr(self, i: int):
        return i // self.rows + 1, i % self.rows + 1

    def names(self) -> list:
        return [f"c{c}r{r:02d}" for c in range(1, self.cols + 1) for r in range(1, self.rows + 1)]

    def column(self, c: int) -> list:
        return [self.idx(c, r) for r in range(1, self.rows + 1)]

    def row(self, r: int) -> list:
        return [self.idx(c, r) for c in range(1, self.cols + 1)]

    def vertical_neighbor(self, i: int) -> int:
        c, r = self.cr(i)
        return self.idx(c, r + 1) if r < self.rows else self.idx(c, r - 1)

    def horizontal_pairs(self) -> list:
        return [(self.idx(c, r), self.idx(c + 1, r)) for r in range(1, self.rows + 1) for c in range(1, self.cols)]

    def vertical_pairs(self) -> list:
        return [(self.idx(c, r), self.idx(c, r + 1)) for c in range(1, self.cols + 1) for r in range(1, self.rows)]


def patterns(grid: Grid, b_rows) -> list:
    b = set(int(r) for r in (b_rows or []))
    return ["B" if grid.cr(i)[1] in b else "A" for i in range(grid.n)]


def grid_from_corners(grid: Grid, bottom_left, top_left, bottom_right, top_right) -> np.ndarray:
    """Approximate positions of all markers by bilinear interpolation of the four corner markers."""
    bl, tl, br, tr = (np.asarray(p, dtype=float) for p in (bottom_left, top_left, bottom_right, top_right))
    out = np.zeros((grid.n, 2))
    for c in range(1, grid.cols + 1):
        u = (c - 1) / max(1, grid.cols - 1)
        bottom = bl + u * (br - bl)
        top = tl + u * (tr - tl)
        for r in range(1, grid.rows + 1):
            v = (r - 1) / max(1, grid.rows - 1)
            out[grid.idx(c, r)] = bottom + v * (top - bottom)
    return out


def quadrant_template(size: int, pattern: str, edge_px: float = 0.8) -> np.ndarray:
    """Ideal 2 x 2 checker (white = +1, black = -1) with soft edges, centred on the X-junction."""
    r = size // 2
    a = np.arange(-r, r + 1, dtype=np.float32)
    X, Y = np.meshgrid(a, a)
    v = np.tanh(X / edge_px) * np.tanh(Y / edge_px)     # image y points down: x*y < 0 -> upper-right / lower-left
    if pattern == "A":       # upper-right & lower-left black
        return v.astype(np.float32)
    if pattern == "B":       # upper-left & lower-right black
        return (-v).astype(np.float32)
    raise ValueError("pattern must be 'A' or 'B'")
