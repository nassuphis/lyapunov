"""
Histogram-based field computation kernels.

These compute various statistics of orbit histograms over a 2D parameter grid
for 1D and 2D dynamical systems.
"""

import numpy as np
from numba import njit, prange

from .lyapunov_fields import map_logical_to_physical
from .hist_helpers import (
    hist_fixed_bins_inplace,
    compute_orbit,
    compute_orbit_2d_ab,
    compute_orbit_2d,
    transform_values,
    transform_hist,
)


@njit(cache=False, fastmath=False, parallel=True)
def hist_field_1d(
    step,
    seq,
    domain,     # [llx, lly, ulx, uly, lrx, lry] in (A,B)
    pix,
    x0,
    n_transient,
    n_iter,
    vcalc,
    hcalc,
    hbins,
    params,
):

    out = np.empty((pix, pix), dtype=np.float64)
    denom = 1.0 if pix <= 1 else (pix - 1.0)

    for j in prange(pix):
        xs = np.empty(n_iter, dtype=np.float64)
        vs = np.empty(n_iter, dtype=np.float64)
        hist = np.zeros(hbins, dtype=np.int64)
        for i in range(pix):
            A, B = map_logical_to_physical(domain, i / denom, j / denom)
            compute_orbit(step, x0, A, B, seq, n_transient, n_iter, xs, params)
            transform_values(vcalc, xs, vs)
            vmin = 1e300
            vmax = -1e300
            for n in range(n_iter):
                v = vs[n]
                if v < vmin: vmin = v
                if v > vmax: vmax = v
            for k in range(hist.size): hist[k] = 0  # reset
            hist_fixed_bins_inplace(hist, vs, vmin, vmax)
            e = transform_hist(hcalc, hist, vs, vmin, vmax)
            out[j, i] = -e

    return out


@njit(cache=False, fastmath=False, parallel=True)
def hist_field_2d_ab(
    step2_ab,
    seq,
    domain,     # [llx, lly, ulx, uly, lrx, lry] in (A,B)
    pix,
    x0,
    y0,
    n_transient,
    n_iter,
    vcalc,
    hcalc,
    hbins,
    params,
):
    out = np.empty((pix, pix), dtype=np.float64)
    denom = 1.0 if pix <= 1 else (pix - 1.0)

    for j in prange(pix):
        xs   = np.empty(n_iter, dtype=np.float64)   # will hold y-orbit
        vs   = np.empty(n_iter, dtype=np.float64)   # transformed values
        hist = np.zeros(hbins, dtype=np.int64)

        for i in range(pix):
            A, B = map_logical_to_physical(domain, i / denom, j / denom)

            compute_orbit_2d_ab(step2_ab, x0, y0, A, B, seq, n_transient, n_iter, xs, params)
            transform_values(vcalc, xs, vs)

            vmin = 1e300
            vmax = -1e300
            for n in range(n_iter):
                v = vs[n]
                if v < vmin: vmin = v
                if v > vmax: vmax = v

            for k in range(hist.size):
                hist[k] = 0

            hist_fixed_bins_inplace(hist, vs, vmin, vmax)
            e = transform_hist(hcalc, hist, vs, vmin, vmax)
            out[j, i] = -e

    return out


@njit(cache=False, fastmath=False, parallel=True)
def hist_field_2d(
    step2,
    domain,     # [llx, lly, ulx, uly, lrx, lry] in (first,second)
    pix,
    x0,
    y0,
    n_transient,
    n_iter,
    vcalc,
    hcalc,
    hbins,
    params,
):
    out = np.empty((pix, pix), dtype=np.float64)
    denom = 1.0 if pix <= 1 else (pix - 1.0)

    for j in prange(pix):
        xs   = np.empty(n_iter, dtype=np.float64)   # will hold y-orbit
        vs   = np.empty(n_iter, dtype=np.float64)   # transformed values
        hist = np.zeros(hbins, dtype=np.int64)

        for i in range(pix):
            first_param, second_param = map_logical_to_physical(domain, i / denom, j / denom)

            compute_orbit_2d(step2, x0, y0, first_param, second_param, n_transient, n_iter, xs, params)
            transform_values(vcalc, xs, vs)

            vmin = 1e300
            vmax = -1e300
            for n in range(n_iter):
                v = vs[n]
                if v < vmin: vmin = v
                if v > vmax: vmax = v

            for k in range(hist.size):
                hist[k] = 0

            hist_fixed_bins_inplace(hist, vs, vmin, vmax)
            e = transform_hist(hcalc, hist, vs, vmin, vmax)
            out[j, i] = -e

    return out


# ---------------------------------------------------------------------------
# x0/xy0 variants: initial conditions as 2D arrays
# ---------------------------------------------------------------------------

@njit(cache=False, fastmath=False, parallel=True)
def hist_field_1d_x0(
    step,
    seq,
    domain,     # [llx, lly, ulx, uly, lrx, lry] in (A,B)
    x0,         # (pix, pix) array of initial x values
    n_transient,
    n_iter,
    vcalc,
    hcalc,
    hbins,
    params,
):
    """
    Histogram field for 1D map with per-pixel initial conditions.
    pix is inferred from x0.shape[0].
    """
    pix = x0.shape[0]
    out = np.empty((pix, pix), dtype=np.float64)
    denom = 1.0 if pix <= 1 else (pix - 1.0)

    for j in prange(pix):
        xs = np.empty(n_iter, dtype=np.float64)
        vs = np.empty(n_iter, dtype=np.float64)
        hist = np.zeros(hbins, dtype=np.int64)

        for i in range(pix):
            A, B = map_logical_to_physical(domain, i / denom, j / denom)
            compute_orbit(step, x0[j, i], A, B, seq, n_transient, n_iter, xs, params)
            transform_values(vcalc, xs, vs)

            vmin = 1e300
            vmax = -1e300
            for n in range(n_iter):
                v = vs[n]
                if v < vmin: vmin = v
                if v > vmax: vmax = v

            for k in range(hist.size):
                hist[k] = 0

            hist_fixed_bins_inplace(hist, vs, vmin, vmax)
            e = transform_hist(hcalc, hist, vs, vmin, vmax)
            out[j, i] = -e

    return out


@njit(cache=False, fastmath=False, parallel=True)
def hist_field_2d_ab_xy0(
    step2_ab,
    seq,
    domain,     # [llx, lly, ulx, uly, lrx, lry] in (A,B)
    x0,         # (pix, pix) array of initial x values
    y0,         # (pix, pix) array of initial y values
    n_transient,
    n_iter,
    vcalc,
    hcalc,
    hbins,
    params,
):
    """
    Histogram field for 2D AB-forced map with per-pixel initial conditions.
    pix is inferred from x0.shape[0].
    """
    pix = x0.shape[0]
    out = np.empty((pix, pix), dtype=np.float64)
    denom = 1.0 if pix <= 1 else (pix - 1.0)

    for j in prange(pix):
        xs = np.empty(n_iter, dtype=np.float64)
        vs = np.empty(n_iter, dtype=np.float64)
        hist = np.zeros(hbins, dtype=np.int64)

        for i in range(pix):
            A, B = map_logical_to_physical(domain, i / denom, j / denom)
            compute_orbit_2d_ab(step2_ab, x0[j, i], y0[j, i], A, B, seq, n_transient, n_iter, xs, params)
            transform_values(vcalc, xs, vs)

            vmin = 1e300
            vmax = -1e300
            for n in range(n_iter):
                v = vs[n]
                if v < vmin: vmin = v
                if v > vmax: vmax = v

            for k in range(hist.size):
                hist[k] = 0

            hist_fixed_bins_inplace(hist, vs, vmin, vmax)
            e = transform_hist(hcalc, hist, vs, vmin, vmax)
            out[j, i] = -e

    return out


@njit(cache=False, fastmath=False, parallel=True)
def hist_field_2d_xy0(
    step2,
    domain,     # [llx, lly, ulx, uly, lrx, lry] in (first,second)
    x0,         # (pix, pix) array of initial x values
    y0,         # (pix, pix) array of initial y values
    n_transient,
    n_iter,
    vcalc,
    hcalc,
    hbins,
    params,
):
    """
    Histogram field for 2D non-forced map with per-pixel initial conditions.
    pix is inferred from x0.shape[0].
    """
    pix = x0.shape[0]
    out = np.empty((pix, pix), dtype=np.float64)
    denom = 1.0 if pix <= 1 else (pix - 1.0)

    for j in prange(pix):
        xs = np.empty(n_iter, dtype=np.float64)
        vs = np.empty(n_iter, dtype=np.float64)
        hist = np.zeros(hbins, dtype=np.int64)

        for i in range(pix):
            first_param, second_param = map_logical_to_physical(domain, i / denom, j / denom)
            compute_orbit_2d(step2, x0[j, i], y0[j, i], first_param, second_param, n_transient, n_iter, xs, params)
            transform_values(vcalc, xs, vs)

            vmin = 1e300
            vmax = -1e300
            for n in range(n_iter):
                v = vs[n]
                if v < vmin: vmin = v
                if v > vmax: vmax = v

            for k in range(hist.size):
                hist[k] = 0

            hist_fixed_bins_inplace(hist, vs, vmin, vmax)
            e = transform_hist(hcalc, hist, vs, vmin, vmax)
            out[j, i] = -e

    return out
