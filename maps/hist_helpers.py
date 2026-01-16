"""
Histogram helper functions for field computation.

Contains:
- Orbit computation (1D, 2D AB-forced, 2D non-forced)
- Value transforms (slope, convexity, curvature, etc.)
- Histogram statistics (entropy, zerocross, skewness, etc.)
"""

import math
import numpy as np
from numba import njit


# ---------------------------------------------------------------------------
# Histogram binning
# ---------------------------------------------------------------------------

@njit
def hist_fixed_bins_inplace(bins, x, xmin, xmax):
    nbins = bins.size
    for k in range(nbins):
        bins[k] = 0
    if xmax <= xmin:
        xmax = xmin + 1e-12
    scale = nbins / (xmax - xmin)
    for val in x:
        j = int((val - xmin) * scale)
        if 0 <= j < nbins:
            bins[j] += 1


# ---------------------------------------------------------------------------
# Parameter array helpers
# ---------------------------------------------------------------------------

@njit
def make_params_with_x0(x0, params):
    """Prepend x0 to params array: [x0, params[0], params[1], ...]"""
    n = params.size
    out = np.empty(n + 1, dtype=np.float64)
    out[0] = x0
    for i in range(n):
        out[i + 1] = params[i]
    return out


@njit
def make_params_with_xy0(x0, y0, params):
    """Prepend x0, y0 to params array: [x0, y0, params[0], params[1], ...]"""
    n = params.size
    out = np.empty(n + 2, dtype=np.float64)
    out[0] = x0
    out[1] = y0
    for i in range(n):
        out[i + 2] = params[i]
    return out


# ---------------------------------------------------------------------------
# Orbit computation
# ---------------------------------------------------------------------------

@njit
def compute_orbit(step, x0, A, B, seq, n_transient, n_iter, xs, params):
    seq_len = seq.size
    x = x0
    # Prepend x0 to params: params[0] = x0, then original params
    p = make_params_with_x0(x0, params)
    for n in range(n_transient):
        force = seq[n % seq_len] & 1
        forced_param = A if force == 0 else B
        x = step(x, forced_param, p)
        if not math.isfinite(x):
            x = 0.5
    for n in range(n_iter):
        force = seq[n % seq_len] & 1
        forced_param = A if force == 0 else B
        x = step(x, forced_param, p)
        if not math.isfinite(x):
            x = 0.5
        xs[n] = x
    return


@njit
def compute_orbit_2d_ab(step2_ab, x0, y0, A, B, seq, n_transient, n_iter, xs, params):
    """
    Collect an orbit for a 2-D AB-forced map, storing *y* into xs.
    (If you want x instead, store x.)
    """
    seq_len = seq.size
    x = x0
    y = y0
    # Prepend x0, y0 to params: params[0] = x0, params[1] = y0, then original params
    p = make_params_with_xy0(x0, y0, params)

    for n in range(n_transient):
        force = seq[n % seq_len] & 1
        forced_param = A if force == 0 else B
        x, y = step2_ab(x, y, forced_param, p)
        if (not math.isfinite(x)) or (not math.isfinite(y)):
            x = 0.5
            y = 0.5

    for n in range(n_iter):
        force = seq[n % seq_len] & 1
        forced_param = A if force == 0 else B
        x, y = step2_ab(x, y, forced_param, p)
        if (not math.isfinite(x)) or (not math.isfinite(y)):
            x = 0.5
            y = 0.5
        xs[n] = y   # <-- histogram variable (you suggested y)

    return


@njit
def compute_orbit_2d(step2, x0, y0, first_param, second_param, n_transient, n_iter, xs, params):
    """
    Collect an orbit for a 2-D non-forced map, storing *y* into xs.
    (If you want x instead, store x.)
    """
    x = x0
    y = y0
    # Prepend x0, y0 to params: params[0] = x0, params[1] = y0, then original params
    p = make_params_with_xy0(x0, y0, params)

    for n in range(n_transient):
        x, y = step2(x, y, first_param, second_param, p)
        if (not math.isfinite(x)) or (not math.isfinite(y)):
            x = 0.5
            y = 0.0

    for n in range(n_iter):
        x, y = step2(x, y, first_param, second_param, p)
        if (not math.isfinite(x)) or (not math.isfinite(y)):
            x = 0.5
            y = 0.0
        xs[n] = y   # <-- histogram variable (you suggested y)

    return


# ---------------------------------------------------------------------------
# Value transforms (vcalc)
# ---------------------------------------------------------------------------

@njit
def copy(xs, vs):
    for n in range(xs.size):
        vs[n] = xs[n]
    return


@njit
def negabs(xs, vs):
    for n in range(xs.size):
        vs[n] = -math.fabs(xs[n])
    return


@njit
def modulo1(xs, vs):
    for n in range(xs.size):
        vs[n] = xs[n] % 1
    return


@njit
def slope(xs, vs):
    px = xs[0]
    for n in range(1, xs.size):
        x = xs[n]
        vs[n] = x - px
        px = x
    vs[0] = vs[1]  # preserve range
    return


@njit
def convexity(xs, vs):
    N = xs.size
    if N < 3:
        for n in range(N):
            vs[n] = 0.0
        return
    ppx = xs[0]
    px = xs[1]
    for n in range(2, N):
        x = xs[n]
        vs[n] = x - 2.0 * px + ppx
        ppx = px
        px = x
    vs[0] = vs[2]  # preserve range
    vs[1] = vs[2]
    return


@njit
def curvature(xs, vs):
    N = xs.size
    if N < 3:
        for n in range(N):
            vs[n] = 0.0
        return
    ppx = xs[0]
    px = xs[1]
    for n in range(2, N):
        x = xs[n]
        num = math.fabs(x - 2.0 * px + ppx)
        denom = math.pow(1.0 + (x - px) ** 2, 3 / 2)
        if denom > 0.0:
            v = num / denom
        else:
            v = 0.0
        vs[n] = v
        ppx = px
        px = x
    vs[0] = vs[2]  # preserve range
    vs[1] = vs[2]
    return


@njit
def product(xs, vs):
    px = xs[0]
    for n in range(1, xs.size):
        x = xs[n]
        vs[n] = x * px
        px = x
    vs[0] = vs[1]  # preserve range
    return


@njit
def ema(xs, vs):
    v = xs[0]
    for n in range(1, xs.size):
        x = xs[n]
        v = v * 0.95 + x * 0.05
        vs[n] = v
    vs[0] = vs[1]  # preserve range
    return


@njit
def rsi(xs, vs):
    px = xs[0]
    u = 0.0
    w = 0.0
    v = 0.0
    for n in range(1, xs.size):
        x = xs[n]
        dx = x - px
        u = 0.9 * u + 0.1 * dx
        w = 0.9 * w + 0.1 * abs(dx)
        if w > 0.0:
            v = u / w
        vs[n] = v
        px = x
    vs[0] = vs[1]  # preserve range
    return


@njit
def transform_values(vcalc, xs, vs):
    N = xs.size
    if N == 0:
        return
    if vcalc == 0:
        copy(xs, vs)
    elif vcalc == 1:
        slope(xs, vs)
    elif vcalc == 2:
        convexity(xs, vs)
    elif vcalc == 3:
        curvature(xs, vs)
    elif vcalc == 4:
        product(xs, vs)
    elif vcalc == 5:
        ema(xs, vs)
    elif vcalc == 6:
        rsi(xs, vs)
    elif vcalc == 7:
        negabs(xs, vs)
    elif vcalc == 8:
        modulo1(xs, vs)
    else:
        copy(xs, vs)
    return


# ---------------------------------------------------------------------------
# Histogram statistics (hcalc)
# ---------------------------------------------------------------------------

@njit
def entropy(hist):
    total = float(np.sum(hist))
    e = 0.0
    if total > 0.0:
        for b in hist:
            if b > 0:
                p = b / total
                e += p * math.log(p)
        e = e / math.log(hist.size)
    return e


@njit
def zerocross(hist):
    m = np.mean(hist)
    s = np.sign(hist - m)
    c = s[1:] * s[:-1]
    e = np.sum(c > 0) / hist.size
    return e


@njit
def slopehist(hist):
    for k in range(hist.size - 1):
        hist[k] = hist[k + 1] - hist[k]
    e = np.std(hist[:-1])
    return e


@njit
def convhist(hist):
    for k in range(hist.size - 2):
        hist[k] = hist[k + 2] - 2 * hist[k + 1] + hist[k]
    e = np.std(hist[:-2])
    return e


@njit
def skewhist(hist):
    a = hist - np.mean(hist)
    m2 = np.mean(a * a)
    m3 = np.mean(a * a * a)
    if m2 > 0:
        e = m3 / (m2 ** 1.5)
    else:
        e = 0.0
    return e


@njit
def sumabschange(hist):
    e = 0.0
    for k in range(hist.size - 1):
        e += abs(hist[k + 1] - hist[k])
    return e


@njit
def maxratio(hist):
    hmax = np.max(hist)
    if hmax == 0:
        return 0.0
    hmean = np.mean(hist)
    return float(hmax / hmean)


@njit
def lrratio(hist):
    leftsum = np.sum(hist[int(hist.size / 2):])
    if leftsum == 0:
        return 0.0
    rightsum = np.sum(hist[0:int(hist.size / 2)])
    return float(rightsum / leftsum)


@njit
def tailratio(hist):
    tail = int(hist.size / 4)
    tailsum = np.sum(hist[:tail]) + np.sum(hist[hist.size - tail:])
    midsum = np.sum(hist[tail:hist.size - tail])
    if midsum == 0:
        return 0.0
    return float(tailsum / midsum)


@njit
def transform_hist(hcalc, hist, vs, vmin, vmax):
    """
    Transform histogram/values into a scalar.
    hcalc 0-9: histogram-based statistics
    hcalc 10+: value-based statistics (use vs, vmin, vmax)
    """
    if hcalc == 0:
        return np.std(hist)
    elif hcalc == 1:
        return entropy(hist)
    elif hcalc == 2:
        return zerocross(hist)
    elif hcalc == 3:
        return slopehist(hist)
    elif hcalc == 4:
        return convhist(hist)
    elif hcalc == 5:
        return skewhist(hist)
    elif hcalc == 6:
        return sumabschange(hist)
    elif hcalc == 7:
        return maxratio(hist)
    elif hcalc == 8:
        return lrratio(hist)
    elif hcalc == 9:
        return tailratio(hist)
    # value-based statistics
    elif hcalc == 10:
        return np.median(vs)
    elif hcalc == 11:
        return vmax - vmin  # range
    elif hcalc == 12:
        return np.mean(vs)
    elif hcalc == 13:
        return np.std(vs)
    elif hcalc == 14:
        v = vs[vs.size - 1]
        if not math.isfinite(v):
            return 0.0
        return v
    return 0.0
