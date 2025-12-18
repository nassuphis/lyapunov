#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pyvips
import zlib

def slot_from_name(p: Path) -> str:
    return p.stem.split("_")[-1]

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

def compressibility_stats(g_u8: np.ndarray) -> tuple[float, float]:
    """
    Returns (zlib_ratio, jpeg_q10_kb) on the downsampled grayscale.
    zlib_ratio ~ compressed_size / raw_size (lower = more compressible).
    jpeg_q10_kb = encoded bytes at Q=10 (lower = more compressible).
    """
    raw = g_u8.tobytes()
    z = zlib.compress(raw, level=9)
    zlib_ratio = len(z) / max(1, len(raw))

    h, w = g_u8.shape
    gv = pyvips.Image.new_from_memory(raw, w, h, 1, "uchar")
    jb = gv.write_to_buffer(".jpg", Q=10, strip=True)  # bytes
    jpeg_q10_kb = len(jb) / 1024.0

    return float(zlib_ratio), float(jpeg_q10_kb)


def sobel_stats_from_gray_u8(g_u8: np.ndarray) -> tuple[float, float, float]:
    h, w = g_u8.shape
    gv = pyvips.Image.new_from_memory(g_u8.tobytes(), w, h, 1, "uchar").cast("float")

    kx = pyvips.Image.new_from_array([
        [-1.0, 0.0,  1.0],
        [-2.0, 0.0,  2.0],
        [-1.0, 0.0,  1.0],
    ])
    ky = pyvips.Image.new_from_array([
        [-1.0, -2.0, -1.0],
        [ 0.0,  0.0,  0.0],
        [ 1.0,  2.0,  1.0],
    ])

    gxv = gv.conv(kx)
    gyv = gv.conv(ky)

    gx = np.frombuffer(gxv.write_to_memory(), dtype=np.float32).reshape(gxv.height, gxv.width)
    gy = np.frombuffer(gyv.write_to_memory(), dtype=np.float32).reshape(gyv.height, gyv.width)

    mag = np.sqrt(gx * gx + gy * gy)
    sobel_mean = float(mag.mean())
    sobel_p95  = float(np.percentile(mag, 95.0))

    # coherence in [0,1]
    Jxx = gx * gx
    Jyy = gy * gy
    Jxy = gx * gy
    tr = Jxx + Jyy
    det_term = (Jxx - Jyy) * (Jxx - Jyy) + 4.0 * (Jxy * Jxy)
    s = np.sqrt(np.maximum(det_term, 0.0))
    l1 = 0.5 * (tr + s)
    l2 = 0.5 * (tr - s)
    coherence = float(np.mean((l1 - l2) / (l1 + l2 + 1e-6)))

    return sobel_mean, sobel_p95, coherence

def patch_std_stats(g_u8: np.ndarray, n: int = 8) -> tuple[float, float, float]:
    """
    Returns (patch_std_median, patch_std_iqr, patch_std_max)
    on an n×n grid of patches.
    """
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
    med = float(np.median(vals))
    iqr = float(np.percentile(vals, 75) - np.percentile(vals, 25))
    mx  = float(np.max(vals))
    return med, iqr, mx

def compute_stats(path: Path) -> dict:
    st = path.stat()
    size_kb = st.st_size / 1024.0

    im = pyvips.Image.new_from_file(str(path), access="sequential")
    g = vips_to_gray_np(im).astype(np.float32)

    std_raw = float(g.std())

    gv = pyvips.Image.new_from_memory(g.astype(np.uint8).tobytes(), g.shape[1], g.shape[0], 1, "uchar")
    gv_blur = gv.gaussblur(6)
    gb = np.frombuffer(gv_blur.write_to_memory(), dtype=np.uint8).reshape(gv_blur.height, gv_blur.width).astype(np.float32)
    std_blur = float(gb.std())

    g_u8 = g.astype(np.uint8)
    zlib_ratio, jpeg_q10_kb = compressibility_stats(g_u8)
    sobel_mean, sobel_p95, coh = sobel_stats_from_gray_u8(g_u8)
    pstd_med, pstd_iqr, pstd_max = patch_std_stats(g_u8, n=8)

    lap = (
        -4 * gb
        + np.roll(gb, 1, 0) + np.roll(gb, -1, 0)
        + np.roll(gb, 1, 1) + np.roll(gb, -1, 1)
    )
    lap_var = float(lap.var())

    return dict(
        size_kb=size_kb,
        std_raw=std_raw,
        std_blur=std_blur,
        lap_var=lap_var,
        sobel_mean=sobel_mean,
        sobel_p95=sobel_p95,
        coherence=coh,
        zlib_ratio=zlib_ratio,
        jpeg_q10_kb=jpeg_q10_kb,
        patch_std_med=pstd_med,
        patch_std_iqr=pstd_iqr,
        patch_std_max=pstd_max,
    )


def percentile(xs: list[float], p: float) -> float:
    return float(np.percentile(np.asarray(xs, dtype=np.float64), p))

def median(xs: list[float]) -> float:
    return float(np.median(np.asarray(xs, dtype=np.float64)))

def decide_boring(
    st: dict,
    pop: dict,
    hard_size_kb: float = 300.0,
    hard_zlib: float = 0.10,
) -> tuple[bool, str]:

    # Hard degenerate
    if st["size_kb"] < hard_size_kb:
        return True, "tiny"

    if st["zlib_ratio"] < hard_zlib:
        return True, "ultra-compressible"

    # Uniformly dead (catches 44/45)
    if st["std_raw"] < pop["p10_std"] and st["patch_std_max"] < pop["p10_patch_std_max"]:
        return True, "uniform-dead"

    # Low-energy / smooth relative to the run (catches borderline like 37/39 when they’re bottom-quintile)
    if (
        st["jpeg_q10_kb"] < pop["p10_jq10"]
        and st["std_raw"] < pop["p20_std"]
        and st["patch_std_max"] < pop["p20_patch_std_max"]
    ):
        return True, "smooth-low-entropy"
    
    # Homogeneous junk: compressible + patch stats look the same everywhere
    if (
        st["jpeg_q10_kb"] < pop["p20_jq10"]
        and st["patch_std_iqr"] < pop["p10_patch_std_iqr"]
        and st["patch_std_max"] < pop["p20_patch_std_max"]
    ):
        return True, "homogeneous"

    return False, ""

def print_row(mark: str, slot: str, st: dict) -> None:
    print(
        f"{mark} {slot:>5}  "
        f"{st['std_raw']:6.1f} {st['std_blur']:6.1f} "
        f"{st['patch_std_max']:6.1f} {st['patch_std_iqr']:6.1f} "
        f"{st['jpeg_q10_kb']:5.1f} "
        f"{st['zlib_ratio']:5.3f}"
    )



def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--glob", default="*.jpg")
    ap.add_argument("--hot", action="store_true")
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    if args.verbose:
        print("  SLOT   STD   BLUR  PSTDMX  PSTDIQ   JQ10  ZLIB")

    files = sorted(args.dir.glob(args.glob), key=lambda p: p.name)
    if not files:
        raise SystemExit("no files matched")

    rows = [(p, compute_stats(p)) for p in files]
    stds  = [st["std_raw"] for _, st in rows]
    blurs = [st["std_blur"] for _, st in rows]
    jq10  = [st["jpeg_q10_kb"] for _, st in rows]
    pmax  = [st["patch_std_max"] for _, st in rows]
    piqr = [st["patch_std_iqr"] for _, st in rows]

    pop = dict(
        # central tendency
        med_std=median(stds),
        med_blur=median(blurs),

        # std thresholds
        p10_std=percentile(stds, 10),
        p20_std=percentile(stds, 20),

        # patch max thresholds
        p10_patch_std_max=percentile(pmax, 10),
        p20_patch_std_max=percentile(pmax, 20),

        # patch IQR thresholds
        p10_patch_std_iqr=percentile(piqr, 10),

        # compressibility thresholds
        p10_jq10=percentile(jq10, 10),
        p20_jq10=percentile(jq10, 20),
    )

    n_del = 0
    for p, st in rows:
        boring, reason = decide_boring(st, pop)

        mark = "X" if boring else " "
        if args.verbose:
            print_row(mark, slot_from_name(p), st)
            if boring:
                print(f"        reason={reason}")

        if boring:
            n_del += 1
            if args.hot:
                p.unlink()

    print(f"{'DRY ' if not args.hot else ''}deleted {n_del} / {len(files)}")


if __name__ == "__main__":
    main()

