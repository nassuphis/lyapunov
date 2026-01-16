"""
Lyapunov fractal field generation from spec strings.

Provides spec2lyapunov() which converts a spec string into an RGB image array.
"""

import numpy as np

from specparser import chain as specparser

import fields
import field_color
from config import make_cfg


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
        field = fields.do_lyapunov_field_1d(map_cfg, pix)

    elif map_cfg["type"] == "step2d_ab":
        print("lyapunov_field_generic_2d_ab")
        field = fields.do_lyapunov_field_2d_ab(map_cfg, pix)

    elif map_cfg["type"] == "step2d":
        print("lyapunov_field_generic_2d")
        field = fields.do_lyapunov_field_2d(map_cfg, pix)

    elif map_cfg["type"] == "step1d_entropy":
        print("entropy_field_generic_1d")
        field = fields.do_entropy_field_1d(map_cfg, pix)

    elif map_cfg["type"] == "step2d_ab_entropy":
        print("entropy_field_generic_2d_ab")
        field = fields.do_entropy_field_2d_ab(map_cfg, pix)

    elif map_cfg["type"] == "step2d_entropy":
        print("entropy_field_generic_2d")
        field = fields.do_entropy_field_2d(map_cfg, pix)

    elif map_cfg["type"] == "step1d_hist":
        print("hist_field_1d")
        field = fields.do_hist_field_1d(map_cfg, pix)

    elif map_cfg["type"] == "step2d_ab_hist":
        print("hist_field_2d_ab")
        field = fields.do_hist_field_2d_ab(map_cfg, pix)

    elif map_cfg["type"] == "step2d_hist":
        print("hist_field_2d")
        field = fields.do_hist_field_2d(map_cfg, pix)

    elif map_cfg["type"] == "step1d_x0_hist":
        print("hist_field_1d_x0")
        field = fields.do_hist_field_1d_x0(map_cfg)

    elif map_cfg["type"] == "step2d_ab_xy0_hist":
        print("hist_field_2d_ab_xy0")
        field = fields.do_hist_field_2d_ab_xy0(map_cfg)

    elif map_cfg["type"] == "step2d_xy0_hist":
        print("hist_field_2d_xy0")
        field = fields.do_hist_field_2d_xy0(map_cfg)

    elif map_cfg["type"] in ("step1d_x0", "step2d_ab_xy0", "step2d_xy0"):
        raise SystemExit(
            f"type={map_cfg['type']} only supported with hist mode. "
            f"Add 'hist:...' to your spec."
        )

    else:
        raise SystemExit(f"Unsupported type={map_cfg['type']} for map '{map_cfg['map_name']}'")

    rgb = field_color.lyapunov_to_rgb(field, specparser.split_chain(spec))

    return rgb
