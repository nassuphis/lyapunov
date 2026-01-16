
import math
import numpy as np
from numba import njit, types, prange

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

# ---------------------------------------------------------------------------
# Lyapunov field 
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Entropy field 
# ---------------------------------------------------------------------------

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

# ---------------------------------------------------------------------------
# histogram field 
# ---------------------------------------------------------------------------

@njit
def hist_fixed_bins_inplace(bins, x, xmin, xmax):
    nbins = bins.size
    for k in range(nbins): bins[k] = 0
    if xmax <= xmin: xmax = xmin + 1e-12
    scale = nbins / (xmax - xmin)
    for val in x:
        j = int((val - xmin) * scale)
        if 0 <= j < nbins:
            bins[j] += 1


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
        if not math.isfinite(x): x = 0.5
    for n in range(n_iter):
        force = seq[n % seq_len] & 1
        forced_param = A if force == 0 else B
        x = step(x, forced_param, p)
        if not math.isfinite(x): x = 0.5
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


@njit
def copy(xs, vs):
    for n in range(xs.size): vs[n] = xs[n]
    return 
    
@njit
def negabs(xs, vs):
    for n in range(xs.size): vs[n] = -math.fabs(xs[n])
    return 
    
@njit
def modulo1(xs, vs):
    for n in range(xs.size): vs[n] = xs[n] % 1
    return 

@njit
def slope(xs, vs):
    px = xs[0]
    for n in range(1, xs.size):
        x = xs[n]
        vs[n] = x - px
        px = x
    vs[0] = vs[1] # preserve range
    return

@njit
def convexity(xs, vs):
    N = xs.size
    if N < 3:
        for n in range(N): vs[n] = 0.0
        return
    ppx = xs[0]
    px  = xs[1]
    for n in range(2, N):
        x = xs[n]
        vs[n] = x - 2.0 * px + ppx
        ppx = px
        px = x
    vs[0] = vs[2] # preserve range
    vs[1] = vs[2]
    return 

@njit
def curvature(xs, vs):
    N = xs.size
    if N < 3:
        for n in range(N): vs[n] = 0.0
        return
    ppx = xs[0]
    px  = xs[1]
    for n in range(2, N):
        x = xs[n]
        num = math.fabs(x - 2.0 * px + ppx)
        denom = math.pow(1.0 + (x - px)**2, 3/2)
        if denom > 0.0: v = num / denom
        else: v = 0.0
        vs[n] = v
        ppx = px
        px = x
    vs[0] = vs[2] # preserve range
    vs[1] = vs[2]
    return

@njit 
def product(xs, vs):
    px = xs[0]
    for n in range(1, xs.size):
        x = xs[n]
        vs[n] = x * px
        px = x
    vs[0]=vs[1] # preserve range
    return 

@njit 
def ema(xs, vs):
    v = xs[0]
    for n in range(1, xs.size):
        x = xs[n]
        v = v * 0.95 + x * 0.05
        vs[n] = v
    vs[0]=vs[1] # preserve range
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
        if w > 0.0: v = u / w
        vs[n] = v
        px = x
    vs[0]=vs[1] # preserve range
    return

@njit
def transform_values(vcalc, xs, vs):
    N = xs.size
    if N == 0: return
    if vcalc == 0: copy(xs,vs)
    elif vcalc == 1: slope(xs,vs)
    elif vcalc == 2: convexity(xs,vs)
    elif vcalc == 3: curvature(xs,vs)
    elif vcalc == 4: product(xs,vs)
    elif vcalc == 5: ema(xs,vs)
    elif vcalc == 6: rsi(xs,vs)
    elif vcalc == 7: negabs(xs,vs)
    elif vcalc == 8: modulo1(xs,vs)
    else: copy(xs,vs)
    return

@njit
def entropy(hist):
    total = float(np.sum(hist))
    e=0.0
    if total > 0.0:
        for b in hist:
            if b > 0:
                p = b / total
                e += p * math.log(p)
        e = e/math.log(hist.size)
    return e

@njit
def zerocross(hist):
    m=np.mean(hist)
    s=np.sign(hist-m)
    c=s[1:]*s[:-1]
    e = np.sum(c>0)/hist.size
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
        hist[k] = hist[k + 2] - 2*hist[k+1] + hist[k]
    e = np.std(hist[:-2])
    return e

@njit
def skewhist(hist):
    a = hist-np.mean(hist)
    m2 = np.mean(a*a)
    m3 = np.mean(a*a*a)
    if m2 > 0:
        e  = m3 / (m2 ** 1.5)
    else:
        e  = 0.0
    return e


@njit
def sumabschange(hist):
    e = 0.0
    for k in range(hist.size-1): 
        e += abs(hist[k+1] - hist[k])
    return e

@njit
def maxratio(hist):
    hmax = np.max(hist)
    if hmax==0: return 0.0
    hmean = np.mean(hist)
    return float(hmax/hmean)

@njit
def lrratio(hist):
    leftsum = np.sum(hist[int(hist.size/2):])
    if leftsum==0: return 0.0
    rightsum = np.sum(hist[0:int(hist.size/2)])
    return float(rightsum/leftsum)

@njit
def tailratio(hist):
    tail = int(hist.size/4)
    tailsum = np.sum(hist[:tail])+np.sum(hist[hist.size-tail:])
    midsum = np.sum(hist[tail:hist.size-tail])
    if midsum==0: return 0.0
    return float(tailsum/midsum)


@njit
def transform_hist(hcalc, hist, vs, vmin, vmax):
    """
    Transform histogram/values into a scalar.
    hcalc 0-9: histogram-based statistics
    hcalc 10+: value-based statistics (use vs, vmin, vmax)
    """
    if   hcalc==0: return np.std(hist)
    elif hcalc==1: return entropy(hist)
    elif hcalc==2: return zerocross(hist)
    elif hcalc==3: return slopehist(hist)
    elif hcalc==4: return convhist(hist)
    elif hcalc==5: return skewhist(hist)
    elif hcalc==6: return sumabschange(hist)
    elif hcalc==7: return maxratio(hist)
    elif hcalc==8: return lrratio(hist)
    elif hcalc==9: return tailratio(hist)
    # value-based statistics
    elif hcalc==10: return np.median(vs)
    elif hcalc==11: return vmax - vmin  # range
    elif hcalc==12: return np.mean(vs)
    elif hcalc==13: return np.std(vs)
    elif hcalc==14:
        v = vs[vs.size - 1]
        if not math.isfinite(v):
            return 0.0
        return v
    return 0.0



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


# ---------------------------------------------------------------------------
# dict-based interfaces to numba functions
# ---------------------------------------------------------------------------

def do_lyapunov_field_1d(map_cfg,pix):
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

def do_lyapunov_field_2d_ab(map_cfg,pix):
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

def do_lyapunov_field_2d(map_cfg,pix):
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

def do_entropy_field_1d(map_cfg,pix):
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

def do_entropy_field_2d_ab(map_cfg,pix):
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

def do_entropy_field_2d(map_cfg,pix):
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

def do_hist_field_1d(map_cfg,pix):
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
    field = raw-np.median(raw)
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



