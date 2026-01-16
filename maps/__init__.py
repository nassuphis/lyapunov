"""
Maps package - dynamical system map templates and build functions.

This package organizes map templates by type:
- maps_step1d: 1D maps with A/B forcing
- maps_step2d: 2D maps with two independent parameters
- maps_step2d_ab: 2D maps with A/B forcing
- maps_step2d_xy0: 2D maps with per-pixel initial conditions

And provides build/compile utilities in map_functions.
"""

from .defaults import (
    DEFAULT_MAP_NAME,
    DEFAULT_SEQ,
    DEFAULT_TRANS,
    DEFAULT_ITER,
    DEFAULT_X0,
    DEFAULT_EPS_LYAP,
    DEFAULT_CLIP,
    DEFAULT_GAMMA,
)

from .maps_step1d import MAPS_STEP1D
from .maps_step2d import MAPS_STEP2D
from .maps_step2d_ab import MAPS_STEP2D_AB
from .maps_step2d_xy0 import MAPS_STEP2D_XY0

from .map_functions import (
    # Symbolic helpers
    sympy_deriv,
    sympy_jacobian_2d,
    # Function text generators
    funtext_1d,
    funtext_1d_deriv,
    funtext_2d_ab_step,
    funtext_2d_ab_jac,
    funtext_2d_step,
    funtext_2d_jac,
    # Python function builders
    funpy_1d,
    funpy_1d_deriv,
    funpy_2d_ab_step,
    funpy_2d_ab_jac,
    funpy_2d_step,
    funpy_2d_jac,
    # JIT compilation
    funjit_1d,
    funjit_1d_deriv,
    funjit_2d_ab_step,
    funjit_2d_ab_jag,
    funjit_2d_step,
    funjit_2d_jag,
    # Type signatures
    STEP_SIG,
    DERIV_SIG,
    STEP2_AB_SIG,
    JAC2_AB_SIG,
    STEP2_SIG,
    JAC2_SIG,
    # Build function (internal version that takes templates)
    build_map as _build_map_impl,
    substitute_common,
    # Sequence handling
    SEQ_ALLOWED_RE,
    looks_like_sequence_token,
    decode_sequence_token,
    seq_to_array,
)

# Combine all map templates into a single dict
MAP_TEMPLATES: dict[str, dict] = {}
MAP_TEMPLATES.update(MAPS_STEP1D)
MAP_TEMPLATES.update(MAPS_STEP2D)
MAP_TEMPLATES.update(MAPS_STEP2D_AB)
MAP_TEMPLATES.update(MAPS_STEP2D_XY0)


def build_map(name: str) -> dict:
    """
    Build a map configuration from a template.

    Args:
        name: Map name from MAP_TEMPLATES

    Returns:
        dict with compiled step/deriv functions and configuration
    """
    return _build_map_impl(name, MAP_TEMPLATES)

__all__ = [
    # Defaults
    "DEFAULT_MAP_NAME",
    "DEFAULT_SEQ",
    "DEFAULT_TRANS",
    "DEFAULT_ITER",
    "DEFAULT_X0",
    "DEFAULT_EPS_LYAP",
    "DEFAULT_CLIP",
    "DEFAULT_GAMMA",
    # Template dicts
    "MAPS_STEP1D",
    "MAPS_STEP2D",
    "MAPS_STEP2D_AB",
    "MAPS_STEP2D_XY0",
    "MAP_TEMPLATES",
    # Build functions
    "sympy_deriv",
    "sympy_jacobian_2d",
    "funtext_1d",
    "funtext_1d_deriv",
    "funtext_2d_ab_step",
    "funtext_2d_ab_jac",
    "funtext_2d_step",
    "funtext_2d_jac",
    "funpy_1d",
    "funpy_1d_deriv",
    "funpy_2d_ab_step",
    "funpy_2d_ab_jac",
    "funpy_2d_step",
    "funpy_2d_jac",
    "funjit_1d",
    "funjit_1d_deriv",
    "funjit_2d_ab_step",
    "funjit_2d_ab_jag",
    "funjit_2d_step",
    "funjit_2d_jag",
    "STEP_SIG",
    "DERIV_SIG",
    "STEP2_AB_SIG",
    "JAC2_AB_SIG",
    "STEP2_SIG",
    "JAC2_SIG",
    "build_map",
    "substitute_common",
    "SEQ_ALLOWED_RE",
    "looks_like_sequence_token",
    "decode_sequence_token",
    "seq_to_array",
]
