"""Frame sources: a folder of images (natural sort), a .npy stack, or a video file (OpenCV).

Frames are returned as float32 grayscale. Colour images are converted with the usual luma weights;
the absolute grey level does not matter because all matching is ZNCC.
"""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
ROTATIONS = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[..., :3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.astype(np.float32)


def preprocess(gray: np.ndarray, blur_sigma: float) -> np.ndarray:
    if blur_sigma and blur_sigma > 0:
        return cv2.GaussianBlur(gray, (0, 0), float(blur_sigma))
    return gray


class FrameSource:
    """Iterate over (k, frame) with k = 0..len-1 (position in the selected sub-sequence)."""

    def __init__(self, path, fps: float = 30.0, start: int = 0, stop=None, step: int = 1, rotate: int = 0):
        if path is None:
            raise ValueError("input.frames is not set")
        self.path = Path(path)
        if rotate not in ROTATIONS:
            raise ValueError("input.rotate must be 0, 90, 180 or 270")
        self.rotate = ROTATIONS[rotate]
        self.fps = float(fps)
        self._cap = None
        if self.path.is_dir():
            self.kind = "images"
            self.files = sorted((p for p in self.path.iterdir() if p.suffix.lower() in IMAGE_EXTS), key=lambda p: natural_key(p.name))
            n = len(self.files)
        elif self.path.suffix.lower() == ".npy":
            self.kind = "npy"
            self.array = np.load(self.path, mmap_mode="r")
            n = self.array.shape[0]
        elif self.path.is_file():
            self.kind = "video"
            cap = cv2.VideoCapture(str(self.path))
            if not cap.isOpened():
                raise IOError(f"cannot open video {self.path}")
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            vfps = cap.get(cv2.CAP_PROP_FPS)
            if vfps and vfps > 0:
                self.fps = float(vfps)
            cap.release()
        else:
            raise FileNotFoundError(self.path)
        if n == 0:
            raise ValueError(f"no frames found in {self.path}")
        stop = n if stop is None else min(int(stop), n)
        self.indices = list(range(int(start), stop, int(step)))

    @classmethod
    def from_config(cls, cfg: dict) -> "FrameSource":
        c = cfg["input"]
        return cls(c["frames"], c.get("fps", 30.0), c.get("start", 0), c.get("stop"), c.get("step", 1), c.get("rotate", 0))

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def frame_index(self) -> np.ndarray:
        return np.asarray(self.indices, dtype=int)

    @property
    def times(self) -> np.ndarray:
        return self.frame_index / self.fps

    def _finish(self, img: np.ndarray) -> np.ndarray:
        img = to_gray(np.asarray(img))
        if self.rotate is not None:
            img = cv2.rotate(img, self.rotate)
        return img

    def read(self, k: int) -> np.ndarray:
        idx = self.indices[k]
        if self.kind == "images":
            img = cv2.imread(str(self.files[idx]), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise IOError(f"cannot read {self.files[idx]}")
            return self._finish(img)
        if self.kind == "npy":
            a = np.asarray(self.array[idx])
            if a.ndim == 3:            # stored as RGB
                a = a[..., ::-1]
            return self._finish(a)
        cap = cv2.VideoCapture(str(self.path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, img = cap.read()
        cap.release()
        if not ok:
            raise IOError(f"cannot read frame {idx} of {self.path}")
        return self._finish(img)

    def __iter__(self):
        if self.kind != "video":
            for k in range(len(self)):
                yield k, self.read(k)
            return
        cap = cv2.VideoCapture(str(self.path))
        wanted = set(self.indices)
        pos, k = 0, 0
        last = self.indices[-1]
        while pos <= last:
            ok = cap.grab()
            if not ok:
                break
            if pos in wanted:
                ok, img = cap.retrieve()
                if not ok:
                    break
                yield k, self._finish(img)
                k += 1
            pos += 1
        cap.release()
        if k < len(self):
            raise IOError(f"video ended early: got {k} of {len(self)} frames")
