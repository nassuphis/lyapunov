#!/usr/bin/env python
"""
boring_edges.py

Edge/gradient-based boredom score.
Standalone CLI and importable score/details.

Core idea:
- Convert image to grayscale [0,1]
- For each blur scale sigma:
    - compute gradients (Sobel)
    - compute edge mask (Canny on the pre-blurred image)
    - extract:
        * edge_density
        * gradient-magnitude entropy on edge pixels
        * orientation entropy on edge pixels (orientation, modulo pi)
    - combine into an "edge complexity"
    - map to boredom_index in (0,1], where 1 = very boring
- With multiple sigmas, the slope across sigmas is often more informative.

Deps:
  pip install pyvips numpy scikit-image scipy
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import feature, filters

import boring_helpers as bh


@dataclass(frozen=True)
class EdgeParams:
    sigma: float = 1.0
    canny_low: float = 0.10
    canny_high: float = 0.30
    n_orient_bins: int = 12


def parse_sigma_list(s: str) -> list[float]:
    try:
        return [float(x) for x in s.split(",") if x.strip() != ""]
    except ValueError:
        raise ValueError(f"Invalid --sigma value: {s}")


def edge_features(gray01: np.ndarray, p: EdgeParams) -> dict[str, float]:
    """
    Features at a single blur scale.
    """
    g = gaussian_filter(gray01, float(p.sigma)) if p.sigma > 0 else gray01

    gx = filters.sobel_h(g)
    gy = filters.sobel_v(g)
    gmag = np.hypot(gx, gy)

    # Use canny on already-blurred image (sigma=0 inside canny)
    edges = feature.canny(
        g,
        sigma=0.0,
        low_threshold=float(p.canny_low),
        high_threshold=float(p.canny_high),
    )

    edge_density = float(edges.mean())

    mags = gmag[edges]
    if mags.size:
        counts, _ = np.histogram(mags, bins=64)  # counts, not density
        grad_entropy = bh.shannon_entropy01_counts(counts)  # 0..1
        mean_gmag = float(mags.mean())  # strictly positive if edges exist
    else:
        grad_entropy = 0.0
        mean_gmag = 0.0

    ang = np.arctan2(gy, gx)[edges]
    if ang.size:
        ang = np.mod(ang, np.pi)
        counts, _ = np.histogram(ang, bins=int(p.n_orient_bins), range=(0.0, np.pi))
        orient_entropy = bh.shannon_entropy01_counts(counts)  # 0..1
    else:
        orient_entropy = 0.0

    return {
        "edge_density": edge_density,
        "mean_gmag": mean_gmag,
        "grad_entropy": grad_entropy,
        "orient_entropy": orient_entropy,
    }


def complexity_from_edge_features(f: dict[str, float]) -> float:
    # edge_density in [0,1]
    # grad_entropy, orient_entropy in [0,1]
    # mean_gmag in ~[0, ~0.5] typically (depends on content)
    c = (
        1.5 * f["edge_density"]
        + 1.0 * f["mean_gmag"]
        + 0.8 * f["grad_entropy"]
        + 0.7 * f["orient_entropy"]
    )
    return float(max(c, 0.0))


def boredom_index_from_complexity(c: float) -> float:
    return float(1.0 / (1.0 + float(c)))


def details(path: str | Path, p: EdgeParams = EdgeParams()) -> dict[str, Any]:
    im = bh.read_gray01(path)
    f = edge_features(im.gray01, p)
    c = complexity_from_edge_features(f)
    b = boredom_index_from_complexity(c)

    return {
        "boredom_index": b,
        "complexity": c,
        **f,
        "params": {
            "sigma": p.sigma,
            "canny_low": p.canny_low,
            "canny_high": p.canny_high,
            "n_orient_bins": p.n_orient_bins,
        },
        "image": {
            "path": im.path,
            "width": im.width,
            "height": im.height,
        },
    }


def score(path: str | Path, p: EdgeParams = EdgeParams()) -> float:
    return float(details(path, p)["boredom_index"])


def fmt_header(sigmas: list[float], *, with_c: bool) -> str:
    cols = ["file"] + [f"b_s{str(s).replace('.','p')}" for s in sigmas]
    if with_c:
        cols += [f"c_s{str(s).replace('.','p')}" for s in sigmas]
    if len(sigmas) >= 2:
        cols.append("slope")
    return f"{cols[0]:<40} " + " ".join(f"{c:>10}" for c in cols[1:])

def fmt_row(path: str, scores: list[float], complexities: list[float], slope: float | None, *, with_c: bool) -> str:
    s = f"{path:<40} "
    s += " ".join(f"{v:10.3f}" for v in scores)
    if with_c:
        s += " " + " ".join(f"{v:10.5f}" for v in complexities)
    if slope is not None:
        s += f" {slope:10.5f}"
    return s

def main() -> int:
    ap = argparse.ArgumentParser(description="Compute an edge-based boredom index.")
    ap.add_argument("images", nargs="+", help="Image paths or globs (expanded by shell).")

    ap.add_argument(
        "--sigma",
        default=str(EdgeParams.sigma),
        help="Blur sigma or comma-separated list (e.g. 0.5,1,2).",
    )
    ap.add_argument("--canny-low", type=float, default=EdgeParams.canny_low)
    ap.add_argument("--canny-high", type=float, default=EdgeParams.canny_high)
    ap.add_argument("--orient-bins", type=int, default=EdgeParams.n_orient_bins)
    ap.add_argument("--with-c", action="store_true", help="Also print complexity columns (more resolution).")

    ap.add_argument("--json", action="store_true", help="Print JSON per image.")
    args = ap.parse_args()

    sigmas = parse_sigma_list(args.sigma)
    header_printed = False
    rc = 0

    for img in args.images:
        path = Path(img)
        if not path.exists():
            print(f"{img}\tERROR:not_found")
            rc = 1
            continue

        scores: list[float] = []
        complexities: list[float] = []
        for sgm in sigmas:
            p = EdgeParams(
                sigma=float(sgm),
                canny_low=float(args.canny_low),
                canny_high=float(args.canny_high),
                n_orient_bins=int(args.orient_bins),
            )
            out = details(path, p)
            scores.append(float(out["boredom_index"]))
            complexities.append(float(out["complexity"]))

        slope = None
        if len(scores) >= 2:
            xs = np.array(sigmas, dtype=np.float64)
            ys = np.array(scores, dtype=np.float64)
            slope = float(np.polyfit(xs, ys, 1)[0])

        if not header_printed and not args.json:
            print(fmt_header(sigmas, with_c=args.with_c))
            header_printed = True

        if args.json:
            print(
                json.dumps(
                    {
                        "file": str(path),
                        "sigmas": sigmas,
                        "boredom": scores,
                        "params": {
                            "canny_low": float(args.canny_low),
                            "canny_high": float(args.canny_high),
                            "orient_bins": int(args.orient_bins),
                        },
                    }
                )
            )
        else:
            print(fmt_row(str(path), scores, complexities, slope, with_c=args.with_c))

    return rc


if __name__ == "__main__":
    raise SystemExit(main())

