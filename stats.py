#!/usr/bin/env python
"""
stats.py — simple per-channel image stats (pyvips) + optional save transform

Examples:
  ./stats.py image.jpg --min --max
  ./stats.py image.jpg --q 1 --q 99 --save qstretch
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyvips


def _hist_u8_band(im_band: pyvips.Image) -> np.ndarray:
    h = im_band.hist_find()
    mem = h.write_to_memory()
    arr = np.frombuffer(mem, dtype=np.uint32)
    if arr.size != 256:
        w = h.width
        if w > 0 and arr.size % w == 0:
            arr = arr.reshape((-1, w)).sum(axis=0)
        else:
            raise RuntimeError(f"Unexpected histogram size: {arr.size} (expected 256)")
    return arr.astype(np.int64)


def _quantile_from_hist(hist: np.ndarray, q: float) -> float:
    if not (0.0 <= q <= 100.0):
        raise ValueError("Quantile must be between 0 and 100.")
    total = hist.sum()
    if total <= 0:
        return float("nan")
    target = (q / 100.0) * total
    cdf = np.cumsum(hist)
    v = int(np.searchsorted(cdf, target, side="left"))
    return float(min(255, max(0, v)))


def _load_rgb_u8(path: str) -> pyvips.Image:
    # For stats / histograms, use random access (sequential can break on jpeg)
    im = pyvips.Image.new_from_file(path, access="random")
    if im.bands < 3:
        raise ValueError(f"Expected at least 3 bands (RGB). Got {im.bands}.")
    if im.bands != 3:
        im = im.extract_band(0, n=3)
    if im.format != "uchar":
        im = im.cast("uchar")
    return im


def _print_block(title: str, values_rgb: tuple[float, float, float], fmt: str) -> None:
    print(title)
    r, g, b = values_rgb
    print(f"R: {fmt.format(r)}")
    print(f"G: {fmt.format(g)}")
    print(f"B: {fmt.format(b)}")


def _out_path_for_stats(in_path: str) -> str:
    p = Path(in_path)
    return str(p.with_name(f"{p.stem}_stats{p.suffix}"))

def _band_hist_u8(im_band: pyvips.Image) -> np.ndarray:
    # expects single-band uchar
    return _hist_u8_band(im_band)


def _median_u8_from_band(im_band_u8: pyvips.Image) -> float:
    h = _band_hist_u8(im_band_u8)
    return float(_quantile_from_hist(h, 50.0))


def _sigmoid_norm(im01: pyvips.Image, k: float, x0: float) -> pyvips.Image:
    """
    Apply logistic sigmoid to an image with values in [0,1], centered at x0 in [0,1],
    with steepness k, and renormalize so 0->0 and 1->1.
    """
    if k <= 0.0: return im01
    # y = 1/(1+exp(-k*(x-x0)))
    t = (im01 - x0) * (-k)
    y = 1.0 / (1.0 + t.exp())

    # renormalize endpoints
    y0 = 1.0 / (1.0 + np.exp(-k * (0.0 - x0)))
    y1 = 1.0 / (1.0 + np.exp(-k * (1.0 - x0)))
    a = 1.0 / (y1 - y0)
    b = -y0 * a
    return y.linear(a, b)


def _winsorize_per_channel(im: pyvips.Image, hists: list[np.ndarray], q_lo: float, q_hi: float) -> tuple[pyvips.Image, list[float], list[float]]:
    qlo_rgb = [float(_quantile_from_hist(h, q_lo)) for h in hists]
    qhi_rgb = [float(_quantile_from_hist(h, q_hi)) for h in hists]

    r = im[0].clamp(min=qlo_rgb[0], max=qhi_rgb[0])
    g = im[1].clamp(min=qlo_rgb[1], max=qhi_rgb[1])
    b = im[2].clamp(min=qlo_rgb[2], max=qhi_rgb[2])
    return r.bandjoin([g, b]), qlo_rgb, qhi_rgb


def sat_boost_rgb(im_u8: pyvips.Image, s_mul: float) -> pyvips.Image:
    """
    Multiply saturation in HSV by s_mul, keep H and V unchanged.
    Input/output are uchar RGB (sRGB-ish).
    """
    if s_mul == 1.0:
        return im_u8

    hsv = im_u8.colourspace("hsv").cast("float")   # H,S,V as bands
    h = hsv[0]
    s = (hsv[1] * float(s_mul)).clamp(min=0.0, max=255.0)
    v = hsv[2]
    out = h.bandjoin([s, v]).cast("uchar").colourspace("srgb")
    return out

def sat_pull_rgb(im_u8: pyvips.Image, alpha: float) -> pyvips.Image:
    """
    Increase saturation by pulling toward max:
      S' = S + alpha * (255 - S)
    alpha in [0,1]
    """
    if alpha <= 0.0:
        return im_u8

    hsv = im_u8.colourspace("hsv").cast("float")
    h = hsv[0]
    s = hsv[1]
    v = hsv[2]

    s2 = s + alpha * (255.0 - s)
    s2 = s2.clamp(min=0.0, max=255.0)

    out = h.bandjoin([s2, v]).cast("uchar").colourspace("srgb")
    return out

def bright_pull_rgb(im_u8: pyvips.Image, beta: float) -> pyvips.Image:
    """
    Increase brightness by pulling toward max:
      V' = V + beta * (255 - V)
    beta should be small (0.01–0.08)
    """
    if beta <= 0.0:
        return im_u8

    hsv = im_u8.colourspace("hsv").cast("float")
    h = hsv[0]
    s = hsv[1]
    v = hsv[2]

    v2 = v + beta * (255.0 - v)
    v2 = v2.clamp(min=0.0, max=255.0)

    out = h.bandjoin([s, v2]).cast("uchar").colourspace("srgb")
    return out

def sigmod(im: pyvips.Image, hists: list[np.ndarray], q_lo: float, q_hi: float, *, k: float = 5.0) -> pyvips.Image:
    """
    Winsorize per-channel to [q_lo,q_hi], then apply a luma sigmoid via RGB gain:
      Y = 0.299R+0.587G+0.114B
      Y' = sigmoid_norm(Y)
      RGB' = RGB * (Y'/max(Y,eps))
    This preserves hue/sat much better than Lab round-trips.
    """
    clipped, _, _ = _winsorize_per_channel(im, hists, q_lo, q_hi)

    # Work in float 0..255
    f = clipped.cast("float")
    r, g, b = f[0], f[1], f[2]

    # Luma in 0..255
    Y = r * 0.299 + g * 0.587 + b * 0.114

    # Median (midpoint) from luma histogram (uchar)
    Y_u8 = Y.cast("uchar")
    y0_u8 = _median_u8_from_band(Y_u8)          # 0..255
    x0 = y0_u8 / 255.0

    # Normalize to 0..1 for sigmoid
    Y01 = (Y / 255.0)

    Y01s = _sigmoid_norm(Y01, k=k, x0=x0)
    Ys = (Y01s * 255.0)

    # Gain (avoid div0)
    eps = 1e-6
    gain = Ys / (Y + eps)

    out = (f * gain).clamp(min=0.0, max=255.0).cast("uchar")
    return out

def sigmod_rgb(clipped_u8: pyvips.Image, *, k: float = 5.0, mid: float = 0.5) -> pyvips.Image:
    """
    Luma sigmoid via RGB gain, centered at fixed midpoint `mid` in [0,1].
    If k<=0: no-op.
    """
    if k <= 0.0:
        return clipped_u8

    # clamp midpoint
    if mid < 0.0: mid = 0.0
    if mid > 1.0: mid = 1.0

    f = clipped_u8.cast("float")
    r, g, b = f[0], f[1], f[2]
    Y = r * 0.299 + g * 0.587 + b * 0.114  # 0..255

    Y01 = Y / 255.0
    Y01s = _sigmoid_norm(Y01, k=k, x0=mid)
    Ys = Y01s * 255.0

    eps = 1e-6
    gain = Ys / (Y + eps)
    return (f * gain).clamp(min=0.0, max=255.0).cast("uchar")


def autolevels(
        im: pyvips.Image, hists: list[np.ndarray], q_lo: float, q_hi: float, *, k: float = 5.0, mid: float = 0.5
    ) -> tuple[pyvips.Image, float, float]:
    """
    Composite "autolevels":
      1) winsorize per-channel to [q_lo,q_hi]
      2) luma sigmoid via RGB gain (preserve hue/sat)
      3) global stretch lo=min(qlo_rgb), hi=max(qhi_rgb)
    """
    clipped, qlo_rgb, qhi_rgb = _winsorize_per_channel(im, hists, q_lo, q_hi)

    # 2) sigmoid (hue/sat-preserving)
    shaped = sigmod_rgb(clipped, k=k, mid=mid)

    # 3) global stretch
    lo = float(min(qlo_rgb))
    hi = float(max(qhi_rgb))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError(f"Invalid stretch range lo={lo} hi={hi} from q{q_lo:g}, q{q_hi:g}.")

    a = 255.0 / (hi - lo)
    b0 = -lo * a
    out = shaped.linear(a, b0).clamp(min=0.0, max=255.0).cast("uchar")
    return out, lo, hi


def qstretch(im: pyvips.Image, hists: list[np.ndarray], q_lo: float, q_hi: float) -> tuple[pyvips.Image, float, float]:
    """
    Winsorize per-channel to [q_lo, q_hi] (each channel has its own bounds),
    then global stretch using:
      lo = min(qlo_rgb)
      hi = max(qhi_rgb)
    mapping lo->0, hi->255, applied to all channels.
    """
    qlo_rgb = [float(_quantile_from_hist(h, q_lo)) for h in hists]
    qhi_rgb = [float(_quantile_from_hist(h, q_hi)) for h in hists]

    lo = float(min(qlo_rgb))
    hi = float(max(qhi_rgb))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError(f"Invalid stretch range lo={lo} hi={hi} from q{q_lo:g}, q{q_hi:g}.")

    # 1) per-channel winsorize (clip) to its own quantile band
    r = im[0].clamp(min=qlo_rgb[0], max=qhi_rgb[0])
    g = im[1].clamp(min=qlo_rgb[1], max=qhi_rgb[1])
    b = im[2].clamp(min=qlo_rgb[2], max=qhi_rgb[2])
    clipped = r.bandjoin([g, b])

    # 2) global stretch lo->0, hi->255
    a = 255.0 / (hi - lo)
    b0 = -lo * a
    out = clipped.cast("float").linear(a, b0).clamp(min=0.0, max=255.0).cast("uchar")
    return out, lo, hi

def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Per-channel stats for RGB images (pyvips).")
    p.add_argument("image", help="Input image (e.g. .jpg)")
    p.add_argument("--min", dest="do_min", action="store_true", help="Show per-channel min")
    p.add_argument("--max", dest="do_max", action="store_true", help="Show per-channel max")
    p.add_argument("--mean", dest="do_mean", action="store_true", help="Show per-channel mean")
    p.add_argument("--median", dest="do_median", action="store_true", help="Show per-channel median (q50)")
    p.add_argument(
        "--q",
        dest="qs",
        action="append",
        type=float,
        default=[],
        help="Quantile in percent (0..100). Repeatable, e.g. --q 1 --q 99",
    )
    p.add_argument("--save", choices=["qstretch", "sigmod", "autolevels"], help="Write a derived image.")
    p.add_argument("--k", type=float, default=5.0, help="Sigmoid steepness (default 5.0)")
    p.add_argument("--mid", type=float, default=0.5,
               help="Sigmoid midpoint in [0,1] (0.5 = ImageMagick-like)")
    p.add_argument("--sat", type=float, default=0.0,
               help="Saturation pull alpha (0..1), e.g. 0.15")
    p.add_argument("--bright", type=float, default=0.0,
               help="Brightness pull beta (0..1), e.g. 0.03")

    args = p.parse_args(argv)

    if not (args.do_min or args.do_max or args.do_mean or args.do_median or args.qs or args.save):
        p.error("Select at least one stat flag or --save")

    try:
        im = _load_rgb_u8(args.image)
    except Exception as e:
        print(f"Error: {repr(e)}", file=sys.stderr)
        return 5

    bands = [im[0], im[1], im[2]]

    # Simple stats
    if args.do_min:
        vals = tuple(float(b.min()) for b in bands)
        _print_block("min", vals, fmt="{:.0f}")
        print()

    if args.do_max:
        vals = tuple(float(b.max()) for b in bands)
        _print_block("max", vals, fmt="{:.0f}")
        print()

    if args.do_mean:
        vals = tuple(float(b.avg()) for b in bands)
        _print_block("mean", vals, fmt="{:.6g}")
        print()

    # Histogram-based
    need_hist = args.do_median or bool(args.qs) or (args.save == "qstretch")
    hists = None
    if need_hist:
        try:
            hists = [_hist_u8_band(b) for b in bands]
        except Exception as e:
            print(f"Error computing histogram: {e}", file=sys.stderr)
            return 3

    if args.do_median:
        vals = tuple(_quantile_from_hist(h, 50.0) for h in hists)  # type: ignore[arg-type]
        _print_block("median", vals, fmt="{:.0f}")
        print()

    for q in args.qs:
        title = f"q{q:g}"
        vals = tuple(_quantile_from_hist(h, q) for h in hists)  # type: ignore[arg-type]
        _print_block(title, vals, fmt="{:.0f}")
        print()

    # Save modes
    if args.save:
        if len(args.qs) < 2:
            print("Error: --save requires at least two --q values (uses the first two).", file=sys.stderr)
            return 4

        q_lo = min(float(args.qs[0]), float(args.qs[1]))
        q_hi = max(float(args.qs[0]), float(args.qs[1]))

        try:
            if args.save == "qstretch":
                out, lo, hi = qstretch(im, hists, q_lo, q_hi)
                msg = f"(lo={lo:.0f}, hi={hi:.0f}, q{q_lo:g}/q{q_hi:g})"
            elif args.save == "sigmod":
                out = sigmod(im, hists, q_lo, q_hi, k=args.k)
                msg = f"(sigmoid k={args.k:g}, q{q_lo:g}/q{q_hi:g})"
            elif args.save == "autolevels":
                out, lo, hi = autolevels(im, hists, q_lo, q_hi, k=args.k,mid=args.mid)
                out = sat_pull_rgb(out, args.sat)
                out = bright_pull_rgb(out, args.bright)
                msg = f"(sigmoid k={args.k:g}, lo={lo:.0f}, hi={hi:.0f}, q{q_lo:g}/q{q_hi:g})"
            else:
                raise ValueError(f"Unknown save mode: {args.save}")
        except Exception as e:
            print(f"Error: {repr(e)}", file=sys.stderr)
            return 5

        out_path = _out_path_for_stats(args.image)
        out.write_to_file(out_path, Q=95, strip=False, interlace=True)
        print(f"saved {args.save}: {out_path}  {msg}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

