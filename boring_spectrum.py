#!/usr/bin/env python
"""
boring_spectrum.py

Spectrum-based boredom score with downsample pyramid.

Core idea:
- Convert to grayscale [0,1]
- For each pyramid level L (downsample factor = 2**L):
    - compute radially-averaged power spectrum
    - extract:
        * spectral entropy (radial power distribution)
        * high-frequency energy ratio
        * optional spectral slope (log P vs log r)
    - combine into "spectral complexity"
    - map to boredom_index in (0,1], where 1 = very boring

CLI:
  ./boring_spectrum.py imgs/*.jpg --levels 0,1,2,3
  ./boring_spectrum.py imgs/*.jpg --bins 64 --hi-frac 0.30

Deps:
  pip install pyvips numpy scipy
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.fft import fft2, fftshift

import boring_helpers as bh


@dataclass(frozen=True)
class SpectrumParams:
    n_bins: int = 64
    hi_frac: float = 0.30
    levels: list[int] = (0, 1, 2)  # downsample exponents; factor = 2**L


def parse_int_list(s: str) -> list[int]:
    try:
        return [int(x) for x in s.split(",") if x.strip() != ""]
    except ValueError:
        raise ValueError(f"Invalid list: {s}")


def _entropy01_from_probs(p: np.ndarray, *, eps: float = 1e-12) -> float:
    """
    Normalized Shannon entropy in [0,1] for a probability vector.
    """
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]
    if p.size <= 1:
        return 0.0
    h = float(-(p * np.log(p + eps)).sum())
    return float(h / (np.log(float(p.size)) + eps))


def _radial_power_spectrum(gray01: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Radially averaged power spectrum (mean power per radius bin).
    """
    h, w = gray01.shape
    if h < 8 or w < 8:
        return np.zeros(int(n_bins), dtype=np.float64)

    fy = np.fft.fftfreq(h)
    fx = np.fft.fftfreq(w)
    FX, FY = np.meshgrid(fx, fy)
    R = np.sqrt(FX * FX + FY * FY)

    F = fftshift(fft2(gray01))
    P = np.abs(F) ** 2

    r = R.ravel()
    p = P.ravel()

    rmax = float(r.max())
    if rmax <= 0:
        return np.zeros(int(n_bins), dtype=np.float64)

    edges = np.linspace(0.0, rmax, int(n_bins) + 1)
    out = np.zeros(int(n_bins), dtype=np.float64)

    for i in range(int(n_bins)):
        m = (r >= edges[i]) & (r < edges[i + 1])
        if m.any():
            out[i] = float(p[m].mean())

    return out


def spectral_features(gray01: np.ndarray, p: SpectrumParams) -> dict[str, float]:
    ps = _radial_power_spectrum(gray01, p.n_bins)
    s = float(ps.sum())
    if s <= 0:
        return {"spec_entropy": 0.0, "hi_ratio": 0.0, "spec_slope": 0.0}

    w = ps / s  # probability-like distribution over radii bins

    spec_entropy = _entropy01_from_probs(w)

    # high-frequency ratio: last hi_frac of bins
    k = int(max(0, min(p.n_bins - 1, round((1.0 - p.hi_frac) * p.n_bins))))
    hi_ratio = float(w[k:].sum())

    # spectral slope on log-log (ignore DC + zeros)
    idx = np.arange(1, p.n_bins, dtype=np.float64)
    y = ps[1:]
    m = y > 0
    if int(m.sum()) >= 6:
        xlog = np.log(idx[m])
        ylog = np.log(y[m])
        spec_slope = float(np.polyfit(xlog, ylog, 1)[0])
    else:
        spec_slope = 0.0

    return {
        "spec_entropy": float(spec_entropy),
        "hi_ratio": float(hi_ratio),
        "spec_slope": float(spec_slope),
    }


def complexity_from_spectral_features(f: dict[str, float]) -> float:
    # Keep everything nonnegative and bounded-ish.
    # - entropy & hi_ratio increase complexity
    # - penalize extremely steep slopes slightly (via exp(-|slope|))
    c = (
        1.0 * f["spec_entropy"]
        + 1.2 * f["hi_ratio"]
        + 0.3 * float(np.exp(-abs(f["spec_slope"])))
    )
    return float(max(c, 0.0))


def boredom_index_from_complexity(c: float) -> float:
    return float(1.0 / (1.0 + float(c)))


def gray01_at_level(im_rgb_u8: "bh.pyvips.Image", level: int) -> np.ndarray:
    """
    Downsample via pyvips resize; level is exponent: factor = 2**level.
    """
    level = int(level)
    if level <= 0:
        return bh.vips_rgb_u8_to_gray01_np(im_rgb_u8)

    scale = 1.0 / (2.0 ** level)
    im_small = im_rgb_u8.resize(scale)
    return bh.vips_rgb_u8_to_gray01_np(im_small)


def details(path: str | Path, p: SpectrumParams) -> dict[str, Any]:
    im_rgb = bh.read_image_rgb_u8(path, access="random")

    per_level = []
    boredom = []
    complexity = []

    for L in p.levels:
        g = gray01_at_level(im_rgb, L)
        f = spectral_features(g, p)
        c = complexity_from_spectral_features(f)
        b = boredom_index_from_complexity(c)
        boredom.append(b)
        complexity.append(c)
        per_level.append({"level": int(L), **f, "complexity": c, "boredom_index": b})

    return {
        "file": str(path),
        "levels": list(p.levels),
        "boredom": boredom,
        "complexity": complexity,
        "per_level": per_level,
        "params": {"n_bins": p.n_bins, "hi_frac": p.hi_frac},
    }


def score(path: str | Path, p: SpectrumParams = SpectrumParams()) -> float:
    # default: level 0 boredom
    d = details(path, p)
    return float(d["boredom"][0]) if d["boredom"] else 1.0


def fmt_header(levels: list[int]) -> str:
    cols = ["file"] + [f"b_p{L}" for L in levels]
    if len(levels) >= 2:
        cols.append("slope")
    return f"{cols[0]:<40} " + " ".join(f"{c:>10}" for c in cols[1:])


def fmt_row(path: str, scores: list[float], slope: float | None) -> str:
    s = f"{path:<40} "
    s += " ".join(f"{v:10.3f}" for v in scores)
    if slope is not None:
        s += f" {slope:10.5f}"
    return s


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute spectrum-based boredom (with pyramid levels).")
    ap.add_argument("images", nargs="+", help="Image paths or globs (expanded by shell).")
    ap.add_argument("--bins", type=int, default=SpectrumParams.n_bins)
    ap.add_argument("--hi-frac", type=float, default=SpectrumParams.hi_frac)
    ap.add_argument(
        "--levels",
        default="0,1,2",
        help="Comma-separated pyramid levels (downsample exponent; factor=2**L). Example: 0,1,2,3",
    )
    ap.add_argument("--json", action="store_true", help="Print JSON per image.")
    args = ap.parse_args()

    levels = parse_int_list(args.levels)
    p = SpectrumParams(n_bins=int(args.bins), hi_frac=float(args.hi_frac), levels=levels)

    header_printed = False
    rc = 0

    for img in args.images:
        path = Path(img)
        if not path.exists():
            print(f"{img}\tERROR:not_found")
            rc = 1
            continue

        out = details(path, p)
        bs = list(out["boredom"])

        slope = None
        if len(bs) >= 2:
            xs = np.array(levels, dtype=np.float64)  # slope vs pyramid level
            ys = np.array(bs, dtype=np.float64)
            slope = float(np.polyfit(xs, ys, 1)[0])

        if args.json:
            print(json.dumps(out))
        else:
            if not header_printed:
                print(fmt_header(levels))
                header_printed = True
            print(fmt_row(str(path), bs, slope))

    return rc


if __name__ == "__main__":
    raise SystemExit(main())

