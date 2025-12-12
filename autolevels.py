#!/usr/bin/env python
"""
autolevels.py

Preview-like Auto Levels:
- per-channel percentile clip (lo/hi) via vips histogram
- linear stretch to [0,255]
- optional mild gamma

Examples:
  python autolevels.py in.jpg
  python autolevels.py in.jpg out.jpg
  python autolevels.py in.jpg --lo 1 --hi 99 --gamma 0.98
  python autolevels.py in.png --keep-alpha
"""

import argparse
from pathlib import Path

import numpy as np
import pyvips as vips


def _percentile_from_hist(hist: vips.Image, pct: float) -> int:
    """
    hist: output of ch.hist_find() for an 8-bit band (256 bins).
    pct: 0..100
    returns intensity in [0,255]
    """
    # hist is tiny (256 bins), safe to pull to memory
    h = np.frombuffer(hist.write_to_memory(), dtype=np.uint32)
    total = int(h.sum())
    if total <= 0:
        return 0

    target = (pct / 100.0) * (total - 1)
    cdf = np.cumsum(h, dtype=np.uint64)
    idx = int(np.searchsorted(cdf, target, side="left"))
    return max(0, min(255, idx))


def _bandjoin3(bands: list[vips.Image]) -> vips.Image:
    if len(bands) != 3:
        raise ValueError("expected exactly 3 bands")
    out = bands[0].bandjoin(bands[1])
    out = out.bandjoin(bands[2])
    return out

def gamma_to_preserve_mid(mid_in_0_1: float, mid_target: float = 0.5) -> float:
    # want: (mid_in_0_1 ** gamma) == mid_target  => gamma = log(mid_target)/log(mid_in)
    eps = 1e-6
    x = min(max(mid_in_0_1, eps), 1.0 - eps)
    return float(np.log(mid_target) / np.log(x))

def autolevels_rgb_old(img: vips.Image, lo_pct: float, hi_pct: float, gamma: float) -> vips.Image:
    if img.format != "uchar":
        img = img.cast("uchar")
    if img.bands < 3:
        raise ValueError("Need RGB (>=3 bands).")

    out_bands: list[vips.Image] = []

    for b in range(3):
        ch = img[b]  # one uchar band

        hist = ch.hist_find()
        lo = _percentile_from_hist(hist, lo_pct)
        hi = _percentile_from_hist(hist, hi_pct)

        if hi <= lo:
            out_bands.append(ch)
            continue

        scale = 255.0 / float(hi - lo)
        chf = ch.cast("float").linear(scale, -scale * float(lo))

        mid = _percentile_from_hist(hist, 50.0)
        mid01 = (mid - lo) / float(hi - lo)
        auto_g = gamma_to_preserve_mid(mid01, 0.5)
        # combine with user gamma (multiply is a reasonable composition here)
        g = auto_g * float(gamma)

        chf = (chf / 255.0) ** g * 255.0

        # clamp (no clip op in your vips)
        chf = (chf < 0.0).ifthenelse(0.0, chf)
        chf = (chf > 255.0).ifthenelse(255.0, chf)

        out_bands.append(chf.cast("uchar"))

    return _bandjoin3(out_bands)

def autolevels_rgb(img: vips.Image, lo_pct: float, hi_pct: float, gamma: float) -> vips.Image:
    if img.format != "uchar":
        img = img.cast("uchar")
    if img.bands < 3:
        raise ValueError("Need RGB (>=3 bands).")

    # luma for stats (keeps color stable)
    # vips "b-w" is a standard grayscale conversion in sRGB space
    luma = img.colourspace("b-w")  # 1 band uchar (or cast-able)

    if luma.format != "uchar":
        luma = luma.cast("uchar")

    hist = luma.hist_find()
    lo = _percentile_from_hist(hist, lo_pct)
    hi = _percentile_from_hist(hist, hi_pct)

    if hi <= lo:
        return img.extract_band(0, n=3)

    # stretch all channels with SAME lo/hi
    scale = 255.0 / float(hi - lo)

    # auto gamma from luma median, applied globally
    mid = _percentile_from_hist(hist, 50.0)
    mid01 = (mid - lo) / float(hi - lo)
    auto_g = gamma_to_preserve_mid(mid01, 0.5)
    g = auto_g * float(gamma)

    out_bands = []
    for b in range(3):
        ch = img[b].cast("float").linear(scale, -scale * float(lo))
        ch = (ch / 255.0) ** g * 255.0

        ch = (ch < 0.0).ifthenelse(0.0, ch)
        ch = (ch > 255.0).ifthenelse(255.0, ch)

        out_bands.append(ch.cast("uchar"))

    return _bandjoin3(out_bands)

def autolevels_lab(img: vips.Image, lo_pct: float, hi_pct: float, gamma: float) -> vips.Image:
    """
    Auto-levels on Lab lightness (L) only, with chroma compensation to avoid “washed out”.

    Changes vs your version:
      - NO auto-gamma-from-median (too brightening / flat)
      - apply user gamma only (default 1.0 = none)
      - compensate saturation by scaling a/b based on how much L was lifted
    """
    if img.format != "uchar":
        img = img.cast("uchar")
    if img.bands < 3:
        raise ValueError("Need RGB (>=3 bands).")

    # sRGB -> Lab (float)
    lab = img.colourspace("lab")
    if lab.format != "float":
        lab = lab.cast("float")

    L = lab[0]  # 0..100
    a = lab[1]
    b = lab[2]

    # Histogram on 8-bit proxy of L
    L_u8 = (L * (255.0 / 100.0)).cast("uchar")
    hist = L_u8.hist_find()

    lo = _percentile_from_hist(hist, lo_pct)
    hi = _percentile_from_hist(hist, hi_pct)

    # anchor near-black/near-white
    if lo <= 2:
        lo = 0
    if hi >= 253:
        hi = 255

    if hi <= lo:
        return img.extract_band(0, n=3)

    # Levels in 0..255 space
    scale = 255.0 / float(hi - lo)
    Lu = L_u8.cast("float").linear(scale, -scale * float(lo))

    # User gamma only (no auto gamma)
    if gamma != 1.0:
        Lu = (Lu / 255.0) ** float(gamma) * 255.0

    # clamp 0..255
    Lu = (Lu < 0.0).ifthenelse(0.0, Lu)
    Lu = (Lu > 255.0).ifthenelse(255.0, Lu)

    # Back to Lab L range 0..100
    L2 = (Lu * (100.0 / 255.0)).cast("float")

    # ---- chroma compensation (key to “not washed out”) ----
    # When L increases, perceived saturation drops. Boost a/b slightly where L rose.
    eps = 1e-3
    ratio = (L2 + eps) / (L + eps)          # >1 where we brightened
    k = 0.35                                 # strength: 0.25–0.5 are sensible
    boost = ratio ** k

    # Cap the boost so highlights don’t go nuclear
    boost = (boost > 1.25).ifthenelse(1.25, boost)

    a2 = a * boost
    b2 = b * boost

    lab2 = L2.bandjoin(a2).bandjoin(b2)
    out = lab2.colourspace("srgb").cast("uchar")
    return out



def print_rgb_stats(img: vips.Image, percentiles=(0.1, 0.5, 1, 5, 50, 95, 99, 99.5, 99.9)) -> None:
    if img.bands < 3:
        raise ValueError("Need RGB (>=3 bands).")

    names = ["R", "G", "B"]
    for b in range(3):
        ch = img[b]
        hist = ch.hist_find()
        h = np.frombuffer(hist.write_to_memory(), dtype=np.uint32)

        total = int(h.sum())
        cdf = np.cumsum(h, dtype=np.uint64)

        # derived stats from histogram
        mean = float(np.sum(np.arange(256, dtype=np.float64) * h) / max(1, total))
        var = float(np.sum(((np.arange(256, dtype=np.float64) - mean) ** 2) * h) / max(1, total))
        std = var ** 0.5

        def pct_to_val(p):
            target = (p / 100.0) * (total - 1)
            return int(np.searchsorted(cdf, target, side="left"))

        vals = {p: pct_to_val(p) for p in percentiles}

        # min/max
        mn = int(np.argmax(h > 0)) if total else 0
        mx = int(255 - np.argmax(h[::-1] > 0)) if total else 255

        print(f"{names[b]}: min={mn} max={mx} mean={mean:.2f} std={std:.2f}  "
              + " ".join([f"p{p:g}={vals[p]}" for p in percentiles]))
        


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Preview-like Auto Levels (percentile clip + stretch).")
    p.add_argument("input", help="Input image (JPG/PNG/etc.)")
    p.add_argument("output", nargs="?", default=None, help="Output image filename (optional)")
    p.add_argument("--lo", type=float, default=0.5, help="Lower percentile clip (default: 0.5)")
    p.add_argument("--hi", type=float, default=99.5, help="Upper percentile clip (default: 99.5)")
    p.add_argument("--gamma", type=float, default=1.0, help="Gamma after stretch (default: 1.0)")
    p.add_argument("--keep-alpha", action="store_true", help="If input has alpha, keep it unchanged in output.")
    p.add_argument("--stats", action="store_true", help="Print RGB channel histogram stats and exit.")

    args = p.parse_args(argv)

    if not (0.0 <= args.lo < args.hi <= 100.0):
        raise SystemExit("--lo/--hi must satisfy 0 <= lo < hi <= 100")
    if args.gamma <= 0:
        raise SystemExit("--gamma must be > 0")

    in_path = Path(args.input)
    if not in_path.is_file():
        raise SystemExit(f"Input not found: {in_path}")

    if args.output is not None:
        out_path = Path(args.output)
    else:
        out_path = in_path.with_name(in_path.stem + "_autolvl" + in_path.suffix)

    # Always open in random mode to avoid libjpeg "out of order read" issues.
    img = vips.Image.new_from_file(str(in_path), access="random")

    alpha = None
    if args.keep_alpha and img.bands == 4:
        rgb = img.extract_band(0, n=3)
        alpha = img[3]
    else:
        if img.bands < 3:
            raise SystemExit(f"Need RGB input (>=3 bands). Got bands={img.bands}")
        rgb = img.extract_band(0, n=3)

    if args.stats:
        print_rgb_stats(rgb)
        return 0

    rgb_out = autolevels_lab(rgb, lo_pct=args.lo, hi_pct=args.hi, gamma=args.gamma)

    if alpha is not None:
        out_img = rgb_out.bandjoin(alpha)
    else:
        out_img = rgb_out

    out_img.write_to_file(str(out_path))
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
