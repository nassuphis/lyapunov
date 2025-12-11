#!/usr/bin/env python
"""
repaletter.py

Re-map a PNG that was generated with rgb_scheme_palette_eq
(from a tri-palette in colors.COLOR_TRI_STRINGS, gamma=1)
to a different tri-palette, preserving the equalized
coordinate t and sign.

Usage examples:
    python repaletter.py nn14loc51_012450.png --pin redgold --pout swiss_modern
    python repaletter.py nn14loc51_012450.png --pin rg --pout swiss_modern --suffix _swiss
"""

import sys
from pathlib import Path

# ---------------------------------------------------------------------
# Make project root importable and pull in rasterizer.colors
# ---------------------------------------------------------------------
parent = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent))

from rasterizer import colors  # noqa: E402

import argparse
import numpy as np
import pyvips as vips
import colorsys


# ---------------------------------------------------------------------
# Helpers to work with your tri-palettes
# ---------------------------------------------------------------------

def _palette_specs(name: str) -> tuple[str, str, str]:
    """
    Given a tri-palette name (in colors.COLOR_TRI_STRINGS),
    return (neg_spec, zero_spec, pos_spec).
    """
    try:
        palette_spec = colors.COLOR_TRI_STRINGS[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown tri-palette {name!r}. "
            f"Available: {', '.join(sorted(colors.COLOR_TRI_STRINGS.keys()))}"
        ) from exc

    parts = palette_spec.split(":")
    if len(parts) < 3:
        raise ValueError(
            f"Invalid tri-palette {name!r}: {palette_spec!r} "
            "(expected 'NEG:ZERO:POS')"
        )
    return parts[0], parts[1], parts[2]


def _parse_color_triplet(spec: str) -> np.ndarray:
    """
    Parse a single color spec into float64 RGB (shape (3,)).
    Uses your existing colors.parse_color_spec.
    """
    r, g, b = colors.parse_color_spec(spec, (0.0, 0.0, 0.0))
    return np.array([r, g, b], dtype=np.float64)


def _compute_t_and_error(
    rgb: np.ndarray,       # (H, W, 3) float64
    zero_rgb: np.ndarray,  # (3,)
    end_rgb: np.ndarray,   # (3,)
) -> tuple[np.ndarray, np.ndarray]:
    """
    For every pixel, estimate t in [0,1] along the line segment
    zero_rgb -> end_rgb and compute reconstruction error.

    Returns:
        t      : (H, W) float, clipped to [0,1]
        error2 : (H, W) float, squared error between rgb and reconstructed
    """
    den = end_rgb - zero_rgb        # (3,)
    mask_ch = (den != 0.0)          # (3,) informative channels
    n_inf = int(mask_ch.sum())

    # Degenerate (all channels identical): no information about t
    if n_inf == 0:
        H, W, _ = rgb.shape
        t = np.zeros((H, W), dtype=np.float64)
        err2 = np.full((H, W), np.inf, dtype=np.float64)
        return t, err2

    den_safe = np.where(mask_ch, den, 1.0)  # avoid div-by-zero
    diff = rgb - zero_rgb                   # (H,W,3)
    t_ch = diff / den_safe                  # (H,W,3)

    # Average t over informative channels only
    mask_ch_exp = mask_ch.reshape(1, 1, 3)  # (1,1,3)
    num = np.sum(t_ch * mask_ch_exp, axis=2)    # (H,W)
    t = num / float(n_inf)
    t = np.clip(t, 0.0, 1.0)

    # Reconstruction and error
    recon = zero_rgb.reshape(1, 1, 3) + t[..., None] * den.reshape(1, 1, 3)
    err2 = np.sum((rgb - recon) ** 2, axis=2)

    return t, err2

def _rgb255_to_hsv01(rgb: np.ndarray) -> np.ndarray:
    """
    Convert a single RGB triplet in 0-255 to HSV in 0-1.
    rgb: shape (3,)
    """
    r, g, b = rgb / 255.0
    h, s, v = colorsys.rgb_to_hsv(float(r), float(g), float(b))
    return np.array([h, s, v], dtype=np.float64)


def _hsv01_to_rgb255_batch(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Vectorized HSV(0-1) -> RGB(0-255) for arrays of shape (...,).
    Returns array of shape (..., 3) in float64.
    """
    c = v * s
    hp = h * 6.0
    x = c * (1.0 - np.abs((hp % 2.0) - 1.0))
    m = v - c

    r_ = np.zeros_like(h)
    g_ = np.zeros_like(h)
    b_ = np.zeros_like(h)

    cond0 = (0.0 <= hp) & (hp < 1.0)
    cond1 = (1.0 <= hp) & (hp < 2.0)
    cond2 = (2.0 <= hp) & (hp < 3.0)
    cond3 = (3.0 <= hp) & (hp < 4.0)
    cond4 = (4.0 <= hp) & (hp < 5.0)
    cond5 = (5.0 <= hp) & (hp < 6.0)

    r_[cond0], g_[cond0], b_[cond0] = c[cond0], x[cond0], 0.0
    r_[cond1], g_[cond1], b_[cond1] = x[cond1], c[cond1], 0.0
    r_[cond2], g_[cond2], b_[cond2] = 0.0, c[cond2], x[cond2]
    r_[cond3], g_[cond3], b_[cond3] = 0.0, x[cond3], c[cond3]
    r_[cond4], g_[cond4], b_[cond4] = x[cond4], 0.0, c[cond4]
    r_[cond5], g_[cond5], b_[cond5] = c[cond5], 0.0, x[cond5]

    r = (r_ + m) * 255.0
    g = (g_ + m) * 255.0
    b = (b_ + m) * 255.0

    return np.stack([r, g, b], axis=-1)


def _interp_hue_short_arc(h0: float, h1: float, t: np.ndarray) -> np.ndarray:
    """
    Interpolate hue in [0,1] along the shortest arc on the circle.
    h0, h1: scalars in [0,1]
    t: array of weights in [0,1]
    """
    dh = h1 - h0
    # wrap to [-0.5, 0.5]
    if dh > 0.5:
        dh -= 1.0
    elif dh < -0.5:
        dh += 1.0
    h = (h0 + dh * t) % 1.0
    return h


def repalette_array(
    rgb_in: np.ndarray,
    palette_in: str,
    palette_out: str,
    keep_colors: list[np.ndarray] = None,
    use_hsv: bool = False,
) -> np.ndarray:
    """
    Core repaletting logic.

    Args:
        rgb_in     : (H,W,3) uint8
        palette_in : original tri-palette name (colors.COLOR_TRI_STRINGS)
        palette_out: target tri-palette name (colors.COLOR_TRI_STRINGS)

    Returns:
        rgb_out: (H,W,3) uint8
    """
    rgb = rgb_in.astype(np.float64)

    # Original palette colors (NEG, ZERO, POS)
    neg_spec_in, zero_spec_in, pos_spec_in = _palette_specs(palette_in)
    neg_in = _parse_color_triplet(neg_spec_in)
    zero_in = _parse_color_triplet(zero_spec_in)
    pos_in = _parse_color_triplet(pos_spec_in)

    # Target palette colors
    neg_spec_out, zero_spec_out, pos_spec_out = _palette_specs(palette_out)
    neg_out = _parse_color_triplet(neg_spec_out)
    zero_out = _parse_color_triplet(zero_spec_out)
    pos_out = _parse_color_triplet(pos_spec_out)

    H, W, _ = rgb.shape

    # Detect "exact zero" pixels in original palette (center color).
    zero_color_in = zero_in.reshape(1, 1, 3)
    is_zero_pixel = np.all(rgb == zero_color_in, axis=2)  # (H,W) bool

    # Estimate t along NEG and POS branches + their errors
    t_neg, err_neg = _compute_t_and_error(rgb, zero_in, neg_in)
    t_pos, err_pos = _compute_t_and_error(rgb, zero_in, pos_in)

    # Decide which branch a pixel belongs to: pos if err_pos < err_neg
    use_pos = err_pos < err_neg  # (H,W) bool

    # Combined t: pick t_pos where use_pos, else t_neg
    t = np.where(use_pos, t_pos, t_neg)

    # Map to output palette
    rgb_out = np.zeros_like(rgb)

    # Zero pixels → map to new zero color
    rgb_out[is_zero_pixel] = zero_out.reshape(1, 1, 3)

    if not use_hsv:
        # ---------- existing RGB interpolation ----------
        # Negative branch pixels (non-zero)
        neg_pixels = (~is_zero_pixel) & (~use_pos)
        if np.any(neg_pixels):
            rgb_out[neg_pixels] = (
                zero_out.reshape(1, 1, 3)
                + t[neg_pixels][..., None] * (neg_out - zero_out).reshape(1, 1, 3)
            )

        # Positive branch pixels (non-zero)
        pos_pixels = (~is_zero_pixel) & use_pos
        if np.any(pos_pixels):
            rgb_out[pos_pixels] = (
                zero_out.reshape(1, 1, 3)
                + t[pos_pixels][..., None] * (pos_out - zero_out).reshape(1, 1, 3)
            )
    else:
        # ---------- HSV interpolation ----------
        # Convert palette stops to HSV (0-1)
        zero_hsv = _rgb255_to_hsv01(zero_out)
        neg_hsv = _rgb255_to_hsv01(neg_out)
        pos_hsv = _rgb255_to_hsv01(pos_out)

        # Negative branch pixels (non-zero)
        neg_pixels = (~is_zero_pixel) & (~use_pos)
        if np.any(neg_pixels):
            t_neg = t[neg_pixels]

            h = _interp_hue_short_arc(zero_hsv[0], neg_hsv[0], t_neg)
            s = zero_hsv[1] + t_neg * (neg_hsv[1] - zero_hsv[1])
            v = zero_hsv[2] + t_neg * (neg_hsv[2] - zero_hsv[2])

            rgb_neg = _hsv01_to_rgb255_batch(h, s, v)
            rgb_out[neg_pixels] = rgb_neg

        # Positive branch pixels (non-zero)
        pos_pixels = (~is_zero_pixel) & use_pos
        if np.any(pos_pixels):
            t_pos = t[pos_pixels]

            h = _interp_hue_short_arc(zero_hsv[0], pos_hsv[0], t_pos)
            s = zero_hsv[1] + t_pos * (pos_hsv[1] - zero_hsv[1])
            v = zero_hsv[2] + t_pos * (pos_hsv[2] - zero_hsv[2])

            rgb_pos = _hsv01_to_rgb255_batch(h, s, v)
            rgb_out[pos_pixels] = rgb_pos

    rgb_out = np.clip(np.rint(rgb_out), 0, 255).astype(np.uint8)
    if keep_colors:
        for kc in keep_colors:
            mask = np.all(rgb_in == kc.reshape(1,1,3), axis=2)
            rgb_out[mask] = kc  # restore exact original
    return rgb_out


# ---------------------------------------------------------------------
# pyvips <-> numpy helpers
# ---------------------------------------------------------------------

def vips_to_numpy(image: vips.Image) -> np.ndarray:
    """
    Convert a pyvips Image (uchar) to a numpy array (H,W,B) uint8.
    """
    mem = image.write_to_memory()
    H, W, B = image.height, image.width, image.bands
    arr = np.frombuffer(mem, dtype=np.uint8).reshape(H, W, B)
    return arr


def numpy_to_vips(arr: np.ndarray) -> vips.Image:
    """
    Convert a numpy array (H,W,B) uint8 to a pyvips Image.
    """
    H, W, B = arr.shape
    return vips.Image.new_from_memory(
        arr.tobytes(),
        W,
        H,
        B,
        vips.BandFormat.UCHAR,
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Re-palette a PNG created with rgb_scheme_palette_eq "
                    "(tri-palette, gamma=1)."
    )
    parser.add_argument("input", help="Input PNG filename")
    parser.add_argument("--pin", required=True,
                        help="Original tri-palette name (e.g. redgold, rg)")
    parser.add_argument("--pout", required=True,
                        help="Target tri-palette name (e.g. swiss_modern)")
    parser.add_argument(
        "--suffix",
        default=None,
        help="Suffix to append before extension for output file "
             "(default: _<pout>, e.g. _swiss_modern)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Explicit output filename (overrides suffix logic)",
    )

    parser.add_argument(
        "--keep",
        action="append",
        default=[],
        help="Hex RGB colors to leave unmodified (e.g. --keep FFFFFF). "
            "Can be used multiple times."
    )

    parser.add_argument(
        "--hsv",
        action="store_true",
        help="Interpolate output palette in HSV instead of RGB.",
    )

    args = parser.parse_args(argv)

    in_path = Path(args.input)
    if not in_path.is_file():
        raise SystemExit(f"Input file not found: {in_path}")

    # Decide output path
    if args.output:
        out_path = Path(args.output)
    else:
        suffix = args.suffix if args.suffix is not None else f"_{args.pout}"
        # auto-append _hsv only when hsv is active AND user didn't give custom suffix
        if args.hsv and args.suffix is None: suffix += "_hsv"
        out_path = in_path.with_name(in_path.stem + suffix + in_path.suffix)

    # Convert keep hex strings -> list of RGB uint8 triplets
    keep_colors = []
    for spec in args.keep:
        # accept with or without #
        s = spec.strip().lstrip("#")
        if len(s) != 6:
            raise SystemExit(f"--keep expects hex like FFFFFF, got {spec}")
        try:
            r = int(s[0:2], 16)
            g = int(s[2:4], 16)
            b = int(s[4:6], 16)
        except ValueError:
            raise SystemExit(f"--keep: invalid hex {spec}")
        keep_colors.append(np.array([r, g, b], dtype=np.uint8))

    # Load via pyvips
    img = vips.Image.new_from_file(str(in_path), access="sequential")
    arr = vips_to_numpy(img)

    if arr.ndim != 3:
        raise SystemExit(f"Unsupported image shape: {arr.shape}")

    H, W, B = arr.shape
    if B == 4:
        rgb_in = arr[..., :3]
        alpha = arr[..., 3]
        has_alpha = True
    elif B == 3:
        rgb_in = arr
        alpha = None
        has_alpha = False
    else:
        raise SystemExit(f"Unsupported number of bands: {B} (expected 3 or 4)")

    rgb_out = repalette_array(
        rgb_in, 
        args.pin, 
        args.pout, 
        keep_colors=keep_colors,
        use_hsv=args.hsv,
    )

    if has_alpha:
        arr_out = np.dstack([rgb_out, alpha])
    else:
        arr_out = rgb_out

    out_img = numpy_to_vips(arr_out)
    out_img.write_to_file(str(out_path))

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

