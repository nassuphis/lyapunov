#!/usr/bin/env python
"""
boring_shapes.py

Shape/morphology-based boredom score.
Standalone CLI and importable score/details.

Core idea:
- Build a "near-black" mask (analog of your -fuzz/-opaque black step).
- Morphology OPEN with a disk to remove speckle and keep coarse structure.
- Compute shape complexity features (components, entropy of component areas,
  boundary convolutedness, Euler number).
- Map complexity -> boredom_index in (0, 1], where 1 = very boring.

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
from skimage import measure, morphology

import boring_helpers as bh


@dataclass(frozen=True)
class ShapeParams:
    black_thresh: float = 0.04
    open_disk: int = 2
    # Optional smoothing of the mask boundary; helps stabilize perimeters a bit
    mask_blur_sigma: float = 0.0

def parse_disk_list(s: str) -> list[int]:
    try:
        return [int(x) for x in s.split(",") if x.strip() != ""]
    except ValueError:
        raise ValueError(f"Invalid --open-disk value: {s}")

def make_shape_mask(gray01: np.ndarray, p: ShapeParams) -> np.ndarray:
    """
    Build a boolean mask for 'dark structure' and clean it with morphology opening.
    """
    m = gray01 <= float(p.black_thresh)

    if p.open_disk > 0:
        se = morphology.disk(int(p.open_disk))
        m = morphology.opening(m, se)

    if p.mask_blur_sigma and p.mask_blur_sigma > 0:
        # blur mask in float then re-threshold; stabilizes tiny jaggies
        mf = gaussian_filter(m.astype(np.float32), float(p.mask_blur_sigma))
        m = mf >= 0.5

    return m


def shape_features(mask: np.ndarray) -> dict[str, float]:
    """
    Extract stable, interpretable shape features.
    """
    labels = measure.label(mask, connectivity=2)
    props = measure.regionprops(labels)

    n_components = float(len(props))
    euler = float(measure.euler_number(mask, connectivity=2))

    if not props:
        return {
            "n_components": 0.0,
            "area_entropy": 0.0,
            "mean_p2_over_a": 0.0,
            "euler_number": euler,
        }

    areas = np.array([p.area for p in props], dtype=np.float64)

    # Prefer Crofton perimeter if present (more stable); else fallback
    perims = []
    for p in props:
        pc = getattr(p, "perimeter_crofton", None)
        if pc is None:
            pc = float(p.perimeter)
        perims.append(float(pc))
    perims = np.array(perims, dtype=np.float64)

    area_entropy = bh.entropy_from_weights(areas)

    mean_p2_over_a = float(np.mean((perims * perims) / (areas + 1e-12)))

    return {
        "n_components": n_components,
        "area_entropy": float(area_entropy),
        "mean_p2_over_a": mean_p2_over_a,
        "euler_number": euler,
    }


def complexity_from_features(f: dict[str, float]) -> float:
    """
    Combine features into a single complexity scalar.
    Higher complexity => less boring.
    """
    # Weighting: tuned for your “coarse dark shape” representation.
    # (We can re-tune after you look at distributions on a few hundred images.)
    n = f["n_components"]
    ae = f["area_entropy"]
    p2a = f["mean_p2_over_a"]
    eu = f["euler_number"]

    # More holes (lower Euler) tends to be more complex => add (-euler)
    c = (
        0.40 * n
        + 0.30 * ae
        + 0.30 * bh.safe_log1p(p2a)
        + 0.10 * (-eu)
    )
    return float(max(c, 0.0))


def boredom_index_from_complexity(c: float) -> float:
    """
    Map complexity -> boredom_index in (0,1], where 1 is most boring.
    """
    return float(1.0 / (1.0 + float(c)))


def details(path: str | Path, p: ShapeParams = ShapeParams()) -> dict[str, Any]:
    im = bh.read_gray01(path)
    mask = make_shape_mask(im.gray01, p)
    f = shape_features(mask)
    c = complexity_from_features(f)
    b = boredom_index_from_complexity(c)

    return {
        "boredom_index": b,
        "complexity": c,
        **f,
        "params": {
            "black_thresh": p.black_thresh,
            "open_disk": p.open_disk,
            "mask_blur_sigma": p.mask_blur_sigma,
        },
        "image": {
            "path": im.path,
            "width": im.width,
            "height": im.height,
        },
    }


def score(path: str | Path, p: ShapeParams = ShapeParams()) -> float:
    return float(details(path, p)["boredom_index"])

def fmt_header(disks):
    cols = ["file"] + [f"b_d{d}" for d in disks]
    if len(disks) >= 2:
        cols.append("slope")
    return f"{cols[0]:<40} " + " ".join(f"{c:>10}" for c in cols[1:])


def fmt_row(path, scores, slope):
    s = f"{path:<40} "
    s += " ".join(f"{v:10.3f}" for v in scores)
    if slope is not None:
        s += f" {slope:10.5f}"
    return s

def main() -> int:
    ap = argparse.ArgumentParser(description="Compute a morphology/shape-based boredom index.")
    ap.add_argument("images",nargs="+",help="Image paths or globs (expanded by shell).")
    ap.add_argument("--black-thresh", type=float, default=ShapeParams.black_thresh)
    ap.add_argument("--open-disk",default=str(ShapeParams.open_disk),help="Disk radius or comma-separated list (e.g. 1,3,5).")
    ap.add_argument("--mask-blur-sigma", type=float, default=ShapeParams.mask_blur_sigma)
    ap.add_argument("--json", action="store_true", help="Print JSON with intermediate features.")
    args = ap.parse_args()

    p = ShapeParams(
        black_thresh=args.black_thresh,
        open_disk=args.open_disk,
        mask_blur_sigma=args.mask_blur_sigma,
    )

    disks = parse_disk_list(args.open_disk)
    header_printed = False
    rc = 0
    for img in args.images:
        path = Path(img)
        if not path.exists():
            print(f"{img}\tERROR:not_found")
            rc = 1
            continue

        row = [str(path)]
        scores = []

        for d in disks:
            p = ShapeParams(
                black_thresh=args.black_thresh,
                open_disk=d,
                mask_blur_sigma=args.mask_blur_sigma,
            )
            out = details(path, p)
            b = out["boredom_index"]
            scores.append(b)
            row.append(f"{b:.6f}")

        # optional slope (linear fit in disk space)
        if len(scores) >= 2:
            xs = np.array(disks, dtype=np.float64)
            ys = np.array(scores, dtype=np.float64)
            slope = float(np.polyfit(xs, ys, 1)[0])
            row.append(f"{slope:.6f}")

        # print header once
        if not header_printed and not args.json:
            print(fmt_header(disks))
            header_printed = True

        if args.json:
            print(json.dumps({
                "file": str(path),
                "disks": disks,
                "boredom": scores,
            }))
        else:
            print(fmt_row(str(path), scores, slope if len(scores) >= 2 else None))

    return rc


if __name__ == "__main__":
    raise SystemExit(main())

