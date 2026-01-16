"""
Spectral entropy field computation kernels.

These compute spectral entropy (Shannon entropy of DFT amplitudes)
over a 2D parameter grid for 1D and 2D dynamical systems.
"""

import math
import numpy as np
from numba import njit, prange

from .lyapunov_fields import map_logical_to_physical


@njit(cache=False, fastmath=False)
def entropy_from_amplitudes(A):
    """
    Shannon entropy of non-negative amplitudes A[0..K-1],
    normalized to [0,1].
    """
    K = A.size
    S = 0.0
    for k in range(K):
        S += A[k]
    if S <= 0.0:
        return 0.0

    H = 0.0
    for k in range(K):
        p = A[k] / S
        if p > 0.0:
            H -= p * math.log(p)

    # normalize to [0,1]
    return H / math.log(K)


@njit(cache=False, fastmath=False, parallel=True)
def entropy_field_1d(
    step,
    seq,
    domain,     # [llx, lly, ulx, uly, lrx, lry] in (A,B)
    pix,
    x0,
    n_transient,
    n_iter,
    omegas,     # 1D float64 array of frequencies
    params,
):
    """
    Streaming spectral-entropy field for a 1-D AB-forced map.
    Uses x_n as the observable, but with the mean removed
    (x_n - running_mean) to avoid DC dominance.
    """
    seq_len = seq.size
    out = np.empty((pix, pix), dtype=np.float64)
    denom = 1.0 if pix <= 1 else (pix - 1.0)
    K = omegas.size

    # global per-call frequency increments
    cw = np.empty(K)
    sw = np.empty(K)
    for k in range(K):
        w = omegas[k]
        cw[k] = math.cos(w)
        sw[k] = math.sin(w)

    for j in prange(pix):
        for i in range(pix):
            A, B = map_logical_to_physical(domain, i / denom, j / denom)

            # burn-in
            x = x0
            for n in range(n_transient):
                force = seq[n % seq_len] % 2
                forced_param = A if force == 0 else B
                x = step(x, forced_param, params)
                if not np.isfinite(x):
                    x = 0.5

            # streaming "DFT" accumulators
            C = np.zeros(K)
            S = np.zeros(K)

            # phase state for each frequency, start at angle = 0
            cos_n = np.ones(K)
            sin_n = np.zeros(K)

            # running mean of x (for DC removal)
            mean = 0.0

            for n in range(n_iter):
                force = seq[n % seq_len] % 2
                forced_param = A if force == 0 else B
                x = step(x, forced_param, params)
                if not np.isfinite(x):
                    x = 0.5

                # online mean update (n goes 0..n_iter-1)
                mean += (x - mean) / (n + 1.0)
                obs = x - mean  # de-meaned observable

                for k in range(K):
                    c = cos_n[k]
                    s = sin_n[k]

                    # accumulate with current phase
                    C[k] += obs * c
                    S[k] += obs * s

                    # rotate phase: θ -> θ + ω_k
                    cwk = cw[k]
                    swk = sw[k]
                    c_new = c * cwk - s * swk
                    s_new = c * swk + s * cwk

                    cos_n[k] = c_new
                    sin_n[k] = s_new

            Avals = np.empty(K)
            for k in range(K):
                Avals[k] = math.sqrt(C[k] * C[k] + S[k] * S[k])

            out[j, i] = entropy_from_amplitudes(Avals)

    return out


@njit(cache=False, fastmath=False, parallel=True)
def entropy_field_2d_ab(
    step2_ab,
    seq,
    domain,        # [llx, lly, ulx, uly, lrx, lry] in (A,B)
    pix,
    x0,
    y0,
    n_transient,
    n_iter,
    omegas,
    params,
):
    """
    Streaming spectral-entropy field for a 2-D AB-forced map.
    Observable = x-component, demeaned (x - mean(x)) to avoid DC dominance.
    """
    seq_len = seq.size
    out = np.empty((pix, pix), dtype=np.float64)
    denom = 1.0 if pix <= 1 else (pix - 1.0)
    K = omegas.size

    # global per-call frequency increments
    cw = np.empty(K)
    sw = np.empty(K)
    for k in range(K):
        w = omegas[k]
        cw[k] = math.cos(w)
        sw[k] = math.sin(w)

    for j in prange(pix):
        for i in range(pix):
            A, B = map_logical_to_physical(domain, i / denom, j / denom)

            x = x0
            y = y0

            # burn-in
            for n in range(n_transient):
                force = seq[n % seq_len] % 2
                forced_param = A if force == 0 else B
                x, y = step2_ab(x, y, forced_param, params)
                if not np.isfinite(x) or not np.isfinite(y):
                    x = 0.5
                    y = 0.5

            # streaming accumulators
            C = np.zeros(K)
            S = np.zeros(K)

            # phase state for each frequency, start at angle = 0
            cos_n = np.ones(K)
            sin_n = np.zeros(K)

            # running mean of x for DC removal
            mean = 0.0

            for n in range(n_iter):
                force = seq[n % seq_len] % 2
                forced_param = A if force == 0 else B
                x, y = step2_ab(x, y, forced_param, params)
                if not np.isfinite(x) or not np.isfinite(y):
                    x = 0.5
                    y = 0.5

                # update running mean of x
                mean += (x - mean) / (n + 1.0)
                obs = x - mean  # de-meaned observable

                for k in range(K):
                    c = cos_n[k]
                    s = sin_n[k]

                    # accumulate with current phase
                    C[k] += obs * c
                    S[k] += obs * s

                    # rotate phase: θ -> θ + ω_k
                    cwk = cw[k]
                    swk = sw[k]
                    c_new = c * cwk - s * swk
                    s_new = c * swk + s * cwk

                    cos_n[k] = c_new
                    sin_n[k] = s_new

            Avals = np.empty(K)
            for k in range(K):
                Avals[k] = math.sqrt(C[k] * C[k] + S[k] * S[k])

            out[j, i] = entropy_from_amplitudes(Avals)

    return out


@njit(cache=False, fastmath=False, parallel=True)
def entropy_field_2d(
    step2,
    domain,        # [llx, lly, ulx, uly, lrx, lry] in (first,second)
    pix,
    x0,
    y0,
    n_transient,
    n_iter,
    omegas,
    params,
):
    """
    Streaming spectral-entropy field for a non-forced 2-D map.
    Parameters are (first, second) from the domain; observable = x (demeaned).
    """
    out = np.empty((pix, pix), dtype=np.float64)
    denom = 1.0 if pix <= 1 else (pix - 1.0)
    K = omegas.size

    # global per-call frequency increments
    cw = np.empty(K)
    sw = np.empty(K)
    for k in range(K):
        w = omegas[k]
        cw[k] = math.cos(w)
        sw[k] = math.sin(w)

    for j in prange(pix):
        for i in range(pix):
            first_param, second_param = map_logical_to_physical(
                domain, i / denom, j / denom
            )

            x = x0
            y = y0

            # burn-in
            for n in range(n_transient):
                x, y = step2(x, y, first_param, second_param, params)
                if not np.isfinite(x) or not np.isfinite(y):
                    x = 0.5
                    y = 0.0

            # streaming accumulators
            C = np.zeros(K)
            S = np.zeros(K)

            # phase state for each frequency, start at angle = 0
            cos_n = np.ones(K)
            sin_n = np.zeros(K)

            # running mean of x for DC removal
            mean = 0.0

            for n in range(n_iter):
                x, y = step2(x, y, first_param, second_param, params)
                if not np.isfinite(x) or not np.isfinite(y):
                    x = 0.5
                    y = 0.0

                # update running mean and de-meaned observable
                mean += (x - mean) / (n + 1.0)
                obs = x - mean   # or math.hypot(x, y) - mean if you prefer radius

                for k in range(K):
                    c = cos_n[k]
                    s = sin_n[k]

                    # accumulate with current phase
                    C[k] += obs * c
                    S[k] += obs * s

                    # rotate phase: θ -> θ + ω_k
                    cwk = cw[k]
                    swk = sw[k]
                    c_new = c * cwk - s * swk
                    s_new = c * swk + s * cwk

                    cos_n[k] = c_new
                    sin_n[k] = s_new

            Avals = np.empty(K)
            for k in range(K):
                Avals[k] = math.sqrt(C[k] * C[k] + S[k] * S[k])

            out[j, i] = entropy_from_amplitudes(Avals)

    return out
