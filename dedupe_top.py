#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pyvips
import random
import zlib

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


def boring_score_simple(g_u8: np.ndarray) -> float:
    # cheap, monotonic "boring-ness"
    # higher = more boring
    zlib_ratio, jpeg_q10_kb = compressibility_stats(g_u8)
    _, pstd_iqr, pstd_max = patch_std_stats(g_u8, n=8)

    return (
        1.2 * (1.0 / (pstd_max + 1e-6)) +
        1.0 * (1.0 / (pstd_iqr + 1e-6)) +
        0.8 * (1.0 / (jpeg_q10_kb + 1e-6)) +
        0.5 * zlib_ratio
    )

def slot_from_name(p: Path) -> str:
    return p.stem.split("_")[-1]


def embed(path: Path, side: int = 32) -> np.ndarray:
    im = pyvips.Image.new_from_file(str(path), access="sequential")

    if im.bands >= 3:
        r, g, b = im[0], im[1], im[2]
        im = (0.299 * r + 0.587 * g + 0.114 * b)
    elif im.bands != 1:
        im = im.extract_band(0)

    im = im.resize(side / im.width)
    im = im.cast("float")

    arr = np.frombuffer(im.write_to_memory(), dtype=np.float32).reshape(im.height, im.width)
    v = arr.reshape(-1)
    v = v - v.mean()
    n = np.linalg.norm(v) + 1e-9
    return v / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--glob", default="*.jpg")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--hot", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    files = sorted(args.dir.glob(args.glob), key=lambda p: p.name)
    if len(files) < 2:
        raise SystemExit("need at least 2 files")
    
    boring_scores = {}
    for p in files:
        im = pyvips.Image.new_from_file(str(p), access="sequential")
        g_u8 = vips_to_gray_np(im)
        boring_scores[p] = boring_score_simple(g_u8)

    X = np.stack([embed(p) for p in files], axis=0)  # (N, D)
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)

    pairs = []
    n = len(files)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((float(S[i, j]), i, j))

    pairs.sort(reverse=True, key=lambda t: t[0])

    if args.verbose:
        show = pairs
    else:
        show = pairs[: args.top]

    print("SIMILARITY   A     B")
    for sim, i, j in show:
        print(
            f"{sim:8.4f}  {slot_from_name(files[i]):>5}  {slot_from_name(files[j]):>5}"
        )
    print()

    if args.hot:
        rng = random.Random(args.seed)
        used = set()
        killed = 0

        for sim, i, j in pairs:
            if killed >= args.top:
                break
            if i in used or j in used:
                continue

            a = files[i]
            b = files[j]

            if boring_scores[a] > boring_scores[b]:
                victim = a
            elif boring_scores[b] > boring_scores[a]:
                victim = b
            else:
                victim = a if rng.random() < 0.5 else b
            victim.unlink()
            used.add(i)
            used.add(j)
            killed += 1
            print(f"deleted {victim.name}")

        if killed:
            print(f"deleted {killed}")
        else:
            print("nothing deleted")


if __name__ == "__main__":
    main()


