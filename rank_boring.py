#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pyvips
import zlib
import random

# --- reuse your existing functions (paste yours here) ---
def slot_from_name(p: Path) -> str:
    return p.stem.split("_")[-1]

def vips_to_gray_np(im: pyvips.Image) -> np.ndarray:
    if im.format != "uchar":
        im = im.cast("uchar")
    if im.bands >= 3:
        r, g, b = im[0], im[1], im[2]
        im = (0.299 * r + 0.587 * g + 0.114 * b).cast("uchar")
    elif im.bands != 1:
        im = im.extract_band(0)
    if im.width > 256:
        im = im.resize(256.0 / im.width)
    return np.frombuffer(im.write_to_memory(), dtype=np.uint8).reshape(im.height, im.width)

def compressibility_stats(g_u8: np.ndarray) -> tuple[float, float]:
    raw = g_u8.tobytes()
    z = zlib.compress(raw, level=9)
    zlib_ratio = len(z) / max(1, len(raw))
    h, w = g_u8.shape
    gv = pyvips.Image.new_from_memory(raw, w, h, 1, "uchar")
    jb = gv.write_to_buffer(".jpg", Q=10, strip=True)
    jpeg_q10_kb = len(jb) / 1024.0
    return float(zlib_ratio), float(jpeg_q10_kb)

def patch_std_stats(g_u8: np.ndarray, n: int = 8) -> tuple[float, float, float]:
    g = g_u8.astype(np.float32)
    h, w = g.shape
    ys = np.linspace(0, h, n + 1, dtype=int)
    xs = np.linspace(0, w, n + 1, dtype=int)
    vals = []
    for i in range(n):
        for j in range(n):
            block = g[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            vals.append(float(block.std()))
    vals = np.asarray(vals, dtype=np.float64)
    return float(np.median(vals)), float(np.percentile(vals, 75) - np.percentile(vals, 25)), float(np.max(vals))

def compute_stats(path: Path) -> dict:
    size_kb = path.stat().st_size / 1024.0
    im = pyvips.Image.new_from_file(str(path), access="sequential")
    g_u8 = vips_to_gray_np(im)
    g = g_u8.astype(np.float32)
    std_raw = float(g.std())

    gv = pyvips.Image.new_from_memory(g_u8.tobytes(), g_u8.shape[1], g_u8.shape[0], 1, "uchar")
    gb_u8 = np.frombuffer(gv.gaussblur(6).write_to_memory(), dtype=np.uint8).reshape(gv.height, gv.width)
    std_blur = float(gb_u8.astype(np.float32).std())

    zlib_ratio, jpeg_q10_kb = compressibility_stats(g_u8)
    pstd_med, pstd_iqr, pstd_max = patch_std_stats(g_u8, n=8)

    return dict(
        size_kb=size_kb,
        std_raw=std_raw,
        std_blur=std_blur,
        jpeg_q10_kb=jpeg_q10_kb,
        zlib_ratio=zlib_ratio,
        patch_std_max=pstd_max,
        patch_std_iqr=pstd_iqr,
    )

def percentile_ranks(vals: np.ndarray) -> np.ndarray:
    # ranks in [0,1], ties handled simply
    order = np.argsort(vals)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, len(vals), endpoint=True)
    return ranks

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--glob", default="*.jpg")
    ap.add_argument("--kill", type=int, default=3)
    ap.add_argument("--hot", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    files = sorted(args.dir.glob(args.glob), key=lambda p: p.name)
    if not files:
        raise SystemExit("no files matched")

    rows = [(p, compute_stats(p)) for p in files]

    std_raw = np.array([st["std_raw"] for _, st in rows], dtype=np.float64)
    pmax    = np.array([st["patch_std_max"] for _, st in rows], dtype=np.float64)
    piqr    = np.array([st["patch_std_iqr"] for _, st in rows], dtype=np.float64)
    jq10    = np.array([st["jpeg_q10_kb"] for _, st in rows], dtype=np.float64)

    # percentile ranks (low=0, high=1). For boring we care about "low", so use (1 - pct).
    pct_std  = percentile_ranks(std_raw)
    pct_pmax = percentile_ranks(pmax)
    pct_piqr = percentile_ranks(piqr)
    pct_jq10 = percentile_ranks(jq10)

    # weights (tune later; these are sane defaults)
    score = (
        1.2 * (1.0 - pct_std)
        + 1.2 * (1.0 - pct_pmax)
        + 1.0 * (1.0 - pct_piqr)
        + 0.8 * (1.0 - pct_jq10)
    )

    idx = np.argsort(-score)  # descending: most boring first
    top = idx[: min(args.kill, len(idx))]
    if args.verbose:
        show=idx
    else:
        show=top
    print("SLOT   SCORE   std   blur  pmax  piqr  jq10")
    for i in show:
        p, st = rows[i]
        print(
            f"{slot_from_name(p):>5}  {score[i]:6.3f}  "
            f"{st['std_raw']:5.1f} {st['std_blur']:5.1f} "
            f"{st['patch_std_max']:5.1f} {st['patch_std_iqr']:5.1f} "
            f"{st['jpeg_q10_kb']:4.1f}"
        )
    print()
   
    if args.hot and len(top) > 0:
        rng = random.Random(args.seed)
        # delete exactly the chosen ones
        for i in top:
            rows[i][0].unlink()
        print(f"deleted {len(top)}")

if __name__ == "__main__":
    main()

