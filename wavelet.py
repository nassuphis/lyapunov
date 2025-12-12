import math
from numba import njit, types

# ========================================
# Helpers
# ========================================

@njit(types.float64(types.float64), fastmath=True, cache=True)
def _sinc_pi(z):
    """
    Normalized sinc: sinc(z) = sin(pi z) / (pi z), with sinc(0)=1.
    """
    if z == 0.0:
        return 1.0
    return math.sin(math.pi * z) / (math.pi * z)

@njit(types.float64(types.int64, types.float64), fastmath=True, cache=True)
def _hermite_prob(n, x):
    """
    Probabilists' Hermite polynomial He_n(x), via recurrence:

        He_0(x) = 1
        He_1(x) = x
        He_{n+1}(x) = x He_n(x) - n He_{n-1}(x)

    Used for Gaussian derivative wavelets.
    """
    if n == 0:
        return 1.0
    if n == 1:
        return x

    Hnm1 = 1.0   # He_0
    Hn   = x     # He_1

    for k in range(1, n):
        kf = float(k)
        Hnp1 = x * Hn - kf * Hnm1
        Hnm1 = Hn
        Hn   = Hnp1

    return Hn

# ========================================
# 1) Mexican Hat (Ricker) Wavelet
#    ψ(x; σ) ∝ (1 - (x/σ)^2) exp(-x^2 / (2σ^2))
# ========================================

@njit(types.float64(types.float64, types.float64), fastmath=True, cache=True)
def mexican_hat(sigma, x):
    """
    Mexican hat / Ricker wavelet (unnormalized).

    Parameters
    ----------
    sigma : float64
        Width parameter (>0). Controls localization.
    x : float64
        Evaluation point.
    """
    s = sigma
    t = x / s
    tt = t * t
    return (1.0 - tt) * math.exp(-0.5 * tt)

# ========================================
# 2) Morlet (Gabor) Wavelet
#    ψ(x; ω0, σ) = exp(-x^2/(2σ^2)) cos(ω0 x / σ)
# ========================================

@njit(types.float64(types.float64, types.float64, types.float64),
      fastmath=True, cache=True)
def morlet(omega0, sigma, x):
    """
    Morlet / Gabor-like wavelet (unnormalized).

    Parameters
    ----------
    omega0 : float64
        Central frequency of the carrier (e.g. ~5.0).
    sigma : float64
        Gaussian width. Larger = slower decay.
    x : float64
        Evaluation point.
    """
    t = x / sigma
    return math.exp(-0.5 * t * t) * math.cos(omega0 * t)

# ========================================
# 3) Shannon-type Wavelet
#    ψ(x; a) = sinc(a x / 2) - sinc(a x)
# ========================================

@njit(types.float64(types.float64, types.float64), fastmath=True, cache=True)
def shannon(a, x):
    """
    Shannon-style wavelet using band-limited sinc differences.

    ψ(x; a) = sinc(a x / 2) - sinc(a x), with sinc(z) = sin(pi z)/(pi z).

    Parameters
    ----------
    a : float64
        Frequency / scale factor. a>0.
    x : float64
        Evaluation point.
    """
    return _sinc_pi(0.5 * a * x) - _sinc_pi(a * x)

# ========================================
# 4) Gaussian Derivative Wavelet (order n)
#    ψ_n(x) = d^n/dx^n exp(-x^2/2)
#           = (-1)^n He_n(x) exp(-x^2/2)
# ========================================

@njit(types.float64(types.int64, types.float64), fastmath=True, cache=True)
def gauss_deriv(n, x):
    """
    n-th derivative of a Gaussian exp(-x^2/2), up to a sign:

        ψ_n(x) = d^n/dx^n exp(-x^2/2)
               = (-1)^n He_n(x) exp(-x^2/2),

    where He_n is the probabilists' Hermite polynomial.

    Parameters
    ----------
    n : int64
        Non-negative integer order (e.g. 1..6). n=2 gives Mexican-hat-like
        shape, n=3 is what you had as (x^3 - 3x) * e^{-x^2/2}, etc.
    x : float64
        Evaluation point.
    """
    if n < 0:
        return math.nan

    He = _hermite_prob(n, x)

    # (-1)^n without pow
    if (n & 1) == 0:
        sign = 1.0
    else:
        sign = -1.0

    return sign * He * math.exp(-0.5 * x * x)

# ========================================
# 5) Scaled Gaussian Derivative Wavelet (order n, width σ)
#    ψ_n(x; n, σ) = σ^{-n-1} ψ_n(x/σ)
# (simple σ-scaling; exact normalization not critical for maps)
# ========================================

@njit(types.float64(types.int64, types.float64, types.float64),
      fastmath=True, cache=True)
def gauss_deriv_scaled(n, sigma, x):
    """
    Scaled n-th Gaussian derivative:

        ψ_n(x; n,σ) ≈ σ^{-n-1} * ψ_n(x/σ)

    This lets you control localization via σ.

    Parameters
    ----------
    n : int64
        Derivative order.
    sigma : float64
        Width parameter (>0).
    x : float64
        Evaluation point.
    """
    t = x / sigma
    base = gauss_deriv(n, t)
    # simple scale factor; exact orthonormality not needed for Lyapunov maps
    return base / (sigma ** (n + 1))

