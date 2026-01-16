"""
Fields module - re-exports all field computation kernels.

This module provides backwards compatibility. All field kernels are now
organized in separate files:

- lyapunov_fields.py: Lyapunov exponent field kernels
- spectral_fields.py: Spectral entropy field kernels
- hist_fields.py: Histogram-based field kernels
- hist_helpers.py: Histogram helper functions
"""

# Re-export coordinate mapping
from .lyapunov_fields import map_logical_to_physical

# Lyapunov exponent fields
from .lyapunov_fields import (
    lyapunov_field_1d,
    lyapunov_field_2d_ab,
    lyapunov_field_2d,
)

# Spectral entropy fields
from .spectral_fields import (
    entropy_from_amplitudes,
    entropy_field_1d,
    entropy_field_2d_ab,
    entropy_field_2d,
)

# Histogram fields
from .hist_fields import (
    hist_field_1d,
    hist_field_2d_ab,
    hist_field_2d,
    hist_field_1d_x0,
    hist_field_2d_ab_xy0,
    hist_field_2d_xy0,
)

# Histogram helpers (for direct use if needed)
from .hist_helpers import (
    hist_fixed_bins_inplace,
    compute_orbit,
    compute_orbit_2d_ab,
    compute_orbit_2d,
    transform_values,
    transform_hist,
)

__all__ = [
    # Coordinate mapping
    "map_logical_to_physical",
    # Lyapunov fields
    "lyapunov_field_1d",
    "lyapunov_field_2d_ab",
    "lyapunov_field_2d",
    # Spectral fields
    "entropy_from_amplitudes",
    "entropy_field_1d",
    "entropy_field_2d_ab",
    "entropy_field_2d",
    # Histogram fields
    "hist_field_1d",
    "hist_field_2d_ab",
    "hist_field_2d",
    "hist_field_1d_x0",
    "hist_field_2d_ab_xy0",
    "hist_field_2d_xy0",
    # Histogram helpers
    "hist_fixed_bins_inplace",
    "compute_orbit",
    "compute_orbit_2d_ab",
    "compute_orbit_2d",
    "transform_values",
    "transform_hist",
]
