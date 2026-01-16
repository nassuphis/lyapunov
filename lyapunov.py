"""
Lyapunov fractal field generation from spec strings.

Provides spec2lyapunov() which converts a spec string into an RGB image array.
"""

import numpy as np

from specparser import chain as specparser

from fields import (
    lyapunov_field_1d,
    lyapunov_field_2d_ab,
    lyapunov_field_2d,
    entropy_field_1d,
    entropy_field_2d_ab,
    entropy_field_2d,
    hist_field_1d,
    hist_field_2d_ab,
    hist_field_2d,
    hist_field_1d_x0,
    hist_field_2d_ab_xy0,
    hist_field_2d_xy0,
)
import field_color
from config import make_cfg


# ---------------------------------------------------------------------------
# dict-based interfaces to numba functions
# ---------------------------------------------------------------------------

def do_lyapunov_field_1d(map_cfg, pix):
    field = lyapunov_field_1d(
        map_cfg["step"],
        map_cfg["deriv"],
        map_cfg["seq_arr"],
        map_cfg["domain_affine"],
        int(pix),
        float(map_cfg["x0"]),
        int(map_cfg["n_tr"]),
        int(map_cfg["n_it"]),
        float(map_cfg["eps"]),
        map_cfg["params"],
    )
    return field


def do_lyapunov_field_2d_ab(map_cfg, pix):
    field = lyapunov_field_2d_ab(
        map_cfg["step2_ab"],
        map_cfg["jac2_ab"],
        map_cfg["seq_arr"],
        map_cfg["domain_affine"],
        int(pix),
        float(map_cfg["x0"]),
        float(map_cfg["y0"]),
        int(map_cfg["n_tr"]),
        int(map_cfg["n_it"]),
        float(map_cfg.get("eps_floor", 1e-16)),
        map_cfg["params"],
    )
    return field


def do_lyapunov_field_2d(map_cfg, pix):
    field = lyapunov_field_2d(
        map_cfg["step2"],
        map_cfg["jac2"],
        map_cfg["domain_affine"],
        int(pix),
        float(map_cfg["x0"]),
        float(map_cfg["y0"]),
        int(map_cfg["n_tr"]),
        int(map_cfg["n_it"]),
        float(map_cfg.get("eps_floor", 1e-16)),
        map_cfg["params"],
    )
    return field


def do_entropy_field_1d(map_cfg, pix):
    raw = entropy_field_1d(
        map_cfg["step"],
        map_cfg["seq_arr"],
        map_cfg["domain_affine"],
        int(pix),
        float(map_cfg["x0"]),
        int(map_cfg["n_tr"]),
        int(map_cfg["n_it"]),
        map_cfg["omegas"],
        map_cfg["params"],
    )
    field = map_cfg["entropy_sign"] * (2.0 * raw - 1.0)
    return field


def do_entropy_field_2d_ab(map_cfg, pix):
    raw = entropy_field_2d_ab(
        map_cfg["step2_ab"],
        map_cfg["seq_arr"],
        map_cfg["domain_affine"],
        int(pix),
        float(map_cfg["x0"]),
        float(map_cfg["y0"]),
        int(map_cfg["n_tr"]),
        int(map_cfg["n_it"]),
        map_cfg["omegas"],
        map_cfg["params"],
    )
    field = map_cfg["entropy_sign"] * (2.0 * raw - 1.0)
    return field


def do_entropy_field_2d(map_cfg, pix):
    raw = entropy_field_2d(
        map_cfg["step2"],
        map_cfg["domain_affine"],
        int(pix),
        float(map_cfg["x0"]),
        float(map_cfg["y0"]),
        int(map_cfg["n_tr"]),
        int(map_cfg["n_it"]),
        map_cfg["omegas"],
        map_cfg["params"],
    )
    field = map_cfg["entropy_sign"] * (2.0 * raw - 1.0)
    return field


def do_hist_field_1d(map_cfg, pix):
    raw = hist_field_1d(
        map_cfg["step"],
        map_cfg["seq_arr"],
        map_cfg["domain_affine"],
        int(pix),
        float(map_cfg["x0"]),
        int(map_cfg["n_tr"]),
        int(map_cfg["n_it"]),
        int(map_cfg["vcalc"]),
        int(map_cfg["hcalc"]),
        int(map_cfg["hbins"]),
        map_cfg["params"],
    )
    field = raw - np.median(raw)
    return field


def do_hist_field_2d_ab(map_cfg, pix):
    raw = hist_field_2d_ab(
        map_cfg["step2_ab"],
        map_cfg["seq_arr"],
        map_cfg["domain_affine"],
        int(pix),
        float(map_cfg["x0"]),
        float(map_cfg["y0"]),
        int(map_cfg["n_tr"]),
        int(map_cfg["n_it"]),
        int(map_cfg["vcalc"]),
        int(map_cfg["hcalc"]),
        int(map_cfg.get("hbins", 32)),
        map_cfg["params"],
    )
    field = raw - np.median(raw)
    return field


def do_hist_field_2d(map_cfg, pix):
    raw = hist_field_2d(
        map_cfg["step2"],
        map_cfg["domain_affine"],
        int(pix),
        float(map_cfg["x0"]),
        float(map_cfg["y0"]),
        int(map_cfg["n_tr"]),
        int(map_cfg["n_it"]),
        int(map_cfg["vcalc"]),
        int(map_cfg["hcalc"]),
        int(map_cfg.get("hbins", 32)),
        map_cfg["params"],
    )
    field = raw - np.median(raw)
    return field


# ---------------------------------------------------------------------------
# x0/xy0 wrappers: initial conditions as 2D arrays
# ---------------------------------------------------------------------------

def do_hist_field_1d_x0(map_cfg):
    """Wrapper for hist_field_1d_x0. x0 must be a 2D array in map_cfg."""
    raw = hist_field_1d_x0(
        map_cfg["step"],
        map_cfg["seq_arr"],
        map_cfg["domain_affine"],
        map_cfg["x0"],  # 2D array
        int(map_cfg["n_tr"]),
        int(map_cfg["n_it"]),
        int(map_cfg["vcalc"]),
        int(map_cfg["hcalc"]),
        int(map_cfg.get("hbins", 32)),
        map_cfg["params"],
    )
    field = raw - np.median(raw)
    return field


def do_hist_field_2d_ab_xy0(map_cfg):
    """Wrapper for hist_field_2d_ab_xy0. x0, y0 must be 2D arrays in map_cfg."""
    raw = hist_field_2d_ab_xy0(
        map_cfg["step2_ab"],
        map_cfg["seq_arr"],
        map_cfg["domain_affine"],
        map_cfg["x0"],  # 2D array
        map_cfg["y0"],  # 2D array
        int(map_cfg["n_tr"]),
        int(map_cfg["n_it"]),
        int(map_cfg["vcalc"]),
        int(map_cfg["hcalc"]),
        int(map_cfg.get("hbins", 32)),
        map_cfg["params"],
    )
    field = raw - np.median(raw)
    return field


def do_hist_field_2d_xy0(map_cfg):
    """Wrapper for hist_field_2d_xy0. x0, y0 must be 2D arrays in map_cfg."""
    raw = hist_field_2d_xy0(
        map_cfg["step2"],
        map_cfg["domain_affine"],
        map_cfg["x0"],  # 2D array
        map_cfg["y0"],  # 2D array
        int(map_cfg["n_tr"]),
        int(map_cfg["n_it"]),
        int(map_cfg["vcalc"]),
        int(map_cfg["hcalc"]),
        int(map_cfg.get("hbins", 32)),
        map_cfg["params"],
    )
    field = raw - np.median(raw)
    return field


def spec2lyapunov(spec: str, pix: int = 5000) -> np.ndarray:
    """
    Generate a Lyapunov fractal RGB image from a spec string.

    Args:
        spec: Spec string like "map:logistic:AB:2:4:2:4,iter:1000,rgb:mh:..."
        pix: Image resolution in pixels (width and height)

    Returns:
        RGB image as numpy array of shape (pix, pix, 3)
    """
    map_cfg = make_cfg(spec, pix)

    if map_cfg["type"] == "step1d":
        print("lyapunov_field_generic_1d")
        field = do_lyapunov_field_1d(map_cfg, pix)

    elif map_cfg["type"] == "step2d_ab":
        print("lyapunov_field_generic_2d_ab")
        field = do_lyapunov_field_2d_ab(map_cfg, pix)

    elif map_cfg["type"] == "step2d":
        print("lyapunov_field_generic_2d")
        field = do_lyapunov_field_2d(map_cfg, pix)

    elif map_cfg["type"] == "step1d_entropy":
        print("entropy_field_generic_1d")
        field = do_entropy_field_1d(map_cfg, pix)

    elif map_cfg["type"] == "step2d_ab_entropy":
        print("entropy_field_generic_2d_ab")
        field = do_entropy_field_2d_ab(map_cfg, pix)

    elif map_cfg["type"] == "step2d_entropy":
        print("entropy_field_generic_2d")
        field = do_entropy_field_2d(map_cfg, pix)

    elif map_cfg["type"] == "step1d_hist":
        print("hist_field_1d")
        field = do_hist_field_1d(map_cfg, pix)

    elif map_cfg["type"] == "step2d_ab_hist":
        print("hist_field_2d_ab")
        field = do_hist_field_2d_ab(map_cfg, pix)

    elif map_cfg["type"] == "step2d_hist":
        print("hist_field_2d")
        field = do_hist_field_2d(map_cfg, pix)

    elif map_cfg["type"] == "step1d_x0_hist":
        print("hist_field_1d_x0")
        field = do_hist_field_1d_x0(map_cfg)

    elif map_cfg["type"] == "step2d_ab_xy0_hist":
        print("hist_field_2d_ab_xy0")
        field = do_hist_field_2d_ab_xy0(map_cfg)

    elif map_cfg["type"] == "step2d_xy0_hist":
        print("hist_field_2d_xy0")
        field = do_hist_field_2d_xy0(map_cfg)

    elif map_cfg["type"] in ("step1d_x0", "step2d_ab_xy0", "step2d_xy0"):
        raise SystemExit(
            f"type={map_cfg['type']} only supported with hist mode. "
            f"Add 'hist:...' to your spec."
        )

    else:
        raise SystemExit(f"Unsupported type={map_cfg['type']} for map '{map_cfg['map_name']}'")

    rgb = field_color.lyapunov_to_rgb(field, specparser.split_chain(spec))

    return rgb
