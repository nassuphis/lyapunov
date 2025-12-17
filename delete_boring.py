#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pyvips


def vips_to_gray_np(im: pyvips.Image) -> np.ndarray:
    # Ensure 8-bit, then grayscale
    if im.format != "uchar":
        im = im.cast("uchar")
    if im.bands >= 3:
        # simple luma-ish weights
        r = im[0]
        g = im[1]
        b = im[2]
        im = (0.299 * r + 0.587 * g + 0.114 * b).cast("uchar")
    elif im.bands != 1:
        im = im.extract_band(0)

    # downsample for speed
    w = im.width
    if w > 256:
        im = im.resize(256.0 / w)

    arr = np.ndarray(
        buffer=im.write_to_memory(),
        dtype=np.uint8,
        shape=(im.height, im.width),
    )
    return arr


def is_boring_jpg(path: Path,
                  size_kb_max: int,
                  std_raw_max: float,
                  std_blur_max: float,
                  lap_var_max: float) -> tuple[bool, dict]:
    st = path.stat()
    size_kb = st.st_size / 1024.0

    im = pyvips.Image.new_from_file(str(path), access="sequential")
    g = vips_to_gray_np(im).astype(np.float32)

    std_raw = float(g.std())

    # heavy blur via repeated box blur (fast enough + good proxy)
    # (Gaussian is fine too, but this keeps deps minimal.)
    # Use vips for blur for speed.
    gv = pyvips.Image.new_from_memory(g.astype(np.uint8).tobytes(), g.shape[1], g.shape[0], 1, "uchar")
    gv_blur = gv.gaussblur(6)  # strong blur
    gb = np.ndarray(
        buffer=gv_blur.write_to_memory(),
        dtype=np.uint8,
        shape=(gv_blur.height, gv_blur.width),
    ).astype(np.float32)

    std_blur = float(gb.std())

    # simple edge energy proxy: Laplacian variance (approx)
    # use finite differences on blurred image to avoid “noise looks edgy”
    lap = (
        -4 * gb
        + np.roll(gb, 1, 0) + np.roll(gb, -1, 0)
        + np.roll(gb, 1, 1) + np.roll(gb, -1, 1)
    )
    lap_var = float(lap.var())

    # Decision:
    # - Very small JPEG is almost always garbage in your case
    # - Or: flat (std_raw tiny)
    # - Or: noise-like (std_blur tiny AND lap_var tiny)
    boring = (
        (size_kb <= size_kb_max) or
        (std_raw <= std_raw_max) or
        (std_blur <= std_blur_max and lap_var <= lap_var_max)
    )

    stats = dict(size_kb=size_kb, std_raw=std_raw, std_blur=std_blur, lap_var=lap_var)
    return boring, stats


def fmt_stats(st: dict) -> tuple[str, str, str, str]:
    # rounded, aligned string fields
    return (
        f"{st['size_kb']:8.0f}",
        f"{st['std_raw']:7.2f}",
        f"{st['std_blur']:7.2f}",
        f"{st['lap_var']:9.2f}",
    )

def print_row(tag: str, name: str, st: dict) -> None:
    size_kb, std_raw, std_blur, lap_var = fmt_stats(st)
    print(f"{tag:6}  {name:28}  {size_kb}  {std_raw}  {std_blur}  {lap_var}")



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--glob", default="*.jpg")
    ap.add_argument("--hot", action="store_true")
    ap.add_argument("--verbose", action="store_true")

    # Tunables (start conservative; tighten after one run)
    ap.add_argument("--size-kb-max", type=int, default=120)     # your “64KB vs 3MB” signal
    ap.add_argument("--std-raw-max", type=float, default=2.0)   # flat/black
    ap.add_argument("--std-blur-max", type=float, default=1.5)  # collapses after blur
    ap.add_argument("--lap-var-max", type=float, default=8.0)   # low structure

    args = ap.parse_args()

    if args.verbose:
        print(f"{'TAG':6}  {'FILE':28}  {'KB':>8}  {'STD':>7}  {'BLUR':>7}  {'LAP_VAR':>9}")

    files = sorted(args.dir.glob(args.glob), key=lambda p: p.name)
    if not files:
        raise SystemExit("no files matched")

    n_del = 0
    for p in files:
        boring, st = is_boring_jpg(
            p,
            size_kb_max=args.size_kb_max,
            std_raw_max=args.std_raw_max,
            std_blur_max=args.std_blur_max,
            lap_var_max=args.lap_var_max,
        )

        if boring:
            n_del += 1
            if args.verbose:
                print_row("BORING", p.name, st)
            else:
                print(f"BORING  {p.name}")
            if args.hot:
                p.unlink()
        else:
            if args.verbose:
                print_row("INTRST", p.name, st)


    print(f"{'DRY ' if not args.hot else ''}deleted {n_del} / {len(files)}")


if __name__ == "__main__":
    main()

