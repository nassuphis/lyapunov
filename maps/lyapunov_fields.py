"""
Lyapunov exponent field computation kernels.

These compute the maximal Lyapunov exponent over a 2D parameter grid
for 1D and 2D dynamical systems.
"""

import math
import numpy as np
from numba import njit, prange


@njit(cache=False, fastmath=False)
def map_logical_to_physical(domain, u, v):
    llx, lly, ulx, uly, lrx, lry = domain
    ex = lrx - llx
    ey = lry - lly
    fx = ulx - llx
    fy = uly - lly
    A = llx + u*ex + v*fx
    B = lly + u*ey + v*fy
    return A, B


@njit(cache=False, fastmath=False, parallel=True)
def lyapunov_field_1d(
    step,
    deriv,
    seq,
    domain,     # <- 1D float64 array: [llx, lly, ulx, uly, lrx, lry]
    pix,
    x0,
    n_transient,
    n_iter,
    eps,
    params,
):
    """
    Generic λ-field for a 1‑D map with A/B forcing, over an arbitrary
    parallelogram in (A,B):

        (u,v) in [0,1]^2   (logical)
        (A,B) = LL + u (LR-LL) + v (UL-LL)

    where A,B are the two parameter values used in the A/B sequence.
    """
    seq_len = seq.size
    out = np.empty((pix, pix), dtype=np.float64)
    denom = 1.0 if pix <= 1 else (pix - 1.0)

    for j in prange(pix):
        for i in range(pix):

            A,B = map_logical_to_physical(domain, i / denom, j / denom)
            x = x0
            acc = 0.0

            for n in range(n_transient + n_iter):
                force = seq[n % seq_len] % 2
                forced_param = A if force==0 else B
                d = deriv(x, forced_param)
                x = step(x, forced_param, params)

                if not np.isfinite(x):
                    x = 0.5

                if n >= n_transient:
                    ad = abs(d)
                    if (not np.isfinite(ad)) or ad < eps:
                        ad = eps
                    acc += math.log(ad)

            out[j, i] = acc / float(n_iter)

    return out


# 2d map with single forced parameter
@njit(cache=False, fastmath=False, parallel=True)
def lyapunov_field_2d_ab(
    step2_ab,
    jac2_ab,
    seq,
    domain,        # [llx, lly, ulx, uly, lrx, lry] in (A,B)-plane
    pix,
    x0,
    y0,
    n_transient,
    n_iter,
    eps_floor,
    params,
):
    seq_len = seq.size
    out = np.empty((pix, pix), dtype=np.float64)

    if eps_floor <= 0.0:
        eps_floor = 1e-16

    denom = 1.0 if pix <= 1 else (pix - 1.0)

    for j in prange(pix):
        for i in range(pix):
            A, B = map_logical_to_physical(domain,  i / denom, j / denom)
            x = x0
            y = y0
            vx = 1.0
            vy = 0.0
            acc = 0.0

            for n in range(n_transient + n_iter):
                force = seq[n % seq_len] % 2
                forced_param = A if force==0 else B

                dXdx, dXdy, dYdx, dYdy = jac2_ab(x, y, forced_param)

                vx_new = dXdx * vx + dXdy * vy
                vy_new = dYdx * vx + dYdy * vy
                vx, vy = vx_new, vy_new

                x_next, y_next = step2_ab(x, y, forced_param, params)

                if not np.isfinite(x_next) or not np.isfinite(y_next):
                    x_next = 0.5
                    y_next = 0.5

                norm = math.sqrt(vx * vx + vy * vy)

                if norm < eps_floor:
                    norm = eps_floor

                if n >= n_transient:
                    acc += math.log(norm)

                vx /= norm
                vy /= norm

                x = x_next
                y = y_next

            out[j, i] = acc / float(n_iter)

    return out


@njit(cache=False, fastmath=False, parallel=True)
def lyapunov_field_2d(
    step2,
    jac2,
    domain,        # [llx, lly, ulx, uly, lrx, lry] in (r,s)-plane
    pix,
    x0,
    y0,
    n_transient,
    n_iter,
    eps_floor,
    params,
):
    """
    Generic λ-field for a 2‑D map over an arbitrary parallelogram in the
    (r,s) parameter plane.

        (u,v) in [0,1]^2
        (r,s) = LL + u (LR-LL) + v (UL-LL)
    """
    out = np.empty((pix, pix), dtype=np.float64)

    if eps_floor <= 0.0:
        eps_floor = 1e-16

    denom = 1.0 if pix <= 1 else (pix - 1.0)

    for j in prange(pix):
        for i in range(pix):

            first_param,second_param = map_logical_to_physical(domain, i / denom, j / denom)

            x = x0
            y = y0
            vx = 1.0
            vy = 0.0
            acc = 0.0

            for n in range(n_transient + n_iter):
                x_next, y_next = step2(x, y, first_param,second_param, params)
                if not np.isfinite(x_next) or not np.isfinite(y_next):
                    x_next = 0.5
                    y_next = 0.0

                dXdx, dXdy, dYdx, dYdy = jac2(x, y, first_param,second_param)

                vx_new = dXdx * vx + dXdy * vy
                vy_new = dYdx * vx + dYdy * vy
                vx, vy = vx_new, vy_new

                norm = math.sqrt(vx * vx + vy * vy)
                if norm < 1e-16:
                    norm = 1e-16

                if n >= n_transient:
                    acc += math.log(norm)

                inv_norm = 1.0 / norm
                vx *= inv_norm
                vy *= inv_norm

                x, y = x_next, y_next

            out[j, i] = acc / float(n_iter)

    return out
