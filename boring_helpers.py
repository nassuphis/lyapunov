#!/usr/bin/env python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyvips


@dataclass(frozen=True)
class ImageGray:
    gray01: np.ndarray          # float32, shape (H,W), values in [0,1]
    width: int
    height: int
    path: str


def read_image_rgb_u8(path: str | Path, *, access: str = "sequential") -> pyvips.Image:
    """
    Read an image via pyvips and return uchar RGB (3 bands), dropping alpha.
    """
    p = str(path)
    im = pyvips.Image.new_from_file(p, access=access)

    # Ensure 3-band RGB
    if im.bands >= 3:
        im = im.extract_band(0, n=3)
    elif im.bands == 1:
        im = im.bandjoin([im, im, im])
    else:
        raise ValueError(f"Unsupported band count: {im.bands} for {p}")

    if im.format != "uchar":
        im = im.cast("uchar")

    return im


def vips_rgb_u8_to_gray01_np(im_rgb_u8: pyvips.Image) -> np.ndarray:
    """
    Convert uchar RGB pyvips image -> grayscale float32 numpy in [0,1].
    Uses Rec.601 luma (same weights you’ve used elsewhere).
    """
    mem = im_rgb_u8.write_to_memory()
    arr = np.frombuffer(mem, dtype=np.uint8).reshape(im_rgb_u8.height, im_rgb_u8.width, 3)

    # 0..255 -> 0..1
    r = arr[..., 0].astype(np.float32) / 255.0
    g = arr[..., 1].astype(np.float32) / 255.0
    b = arr[..., 2].astype(np.float32) / 255.0
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray


def read_gray01(path: str | Path, *, access: str = "sequential") -> ImageGray:
    im = read_image_rgb_u8(path, access=access)
    gray01 = vips_rgb_u8_to_gray01_np(im)
    return ImageGray(gray01=gray01, width=im.width, height=im.height, path=str(path))


def safe_log1p(x: float) -> float:
    # For stability when x is huge or tiny
    return float(np.log1p(np.maximum(x, 0.0)))


def entropy_from_weights(w: np.ndarray, *, eps: float = 1e-12) -> float:
    """
    Entropy of a nonnegative vector treated as weights.
    """
    w = np.asarray(w, dtype=np.float64)
    s = float(w.sum())
    if s <= 0:
        return 0.0
    p = w / (s + eps)
    p = p[p > 0]
    return float(-(p * np.log(p + eps)).sum())

def shannon_entropy_counts(counts: np.ndarray, *, eps: float = 1e-12) -> float:
    """
    Discrete Shannon entropy of histogram counts (>=0). Always >= 0.
    """
    c = np.asarray(counts, dtype=np.float64)
    s = float(c.sum())
    if s <= 0:
        return 0.0
    p = c / (s + eps)
    p = p[p > 0]
    return float(-(p * np.log(p + eps)).sum())


def shannon_entropy01_counts(counts: np.ndarray, *, eps: float = 1e-12) -> float:
    """
    Shannon entropy normalized to [0,1] by dividing by log(K),
    where K is number of bins (nonzero length).
    """
    c = np.asarray(counts, dtype=np.float64)
    k = int(c.size)
    if k <= 1:
        return 0.0
    h = shannon_entropy_counts(c, eps=eps)
    return float(h / (np.log(k) + eps))


