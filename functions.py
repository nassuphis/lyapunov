import math
import cmath
import numpy as np
from numba import njit, types

import wavelet

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PI = math.pi
TWO_PI = 2.0 * PI
# Ai(0), Ai'(0), Bi(0), Bi'(0)
AI0  = 0.3550280538878172
AI0P = -0.2588194037928068
BI0  = 0.6149266274460007
BI0P = 0.4482883573538264
# Tolerance & max iterations for the alternating series
ZETA_TOL = 1e-15
ZETA_MAX_N = 20
#
EULER_GAMMA = 0.5772156649015328606  # Euler–Mascheroni constant
PI = math.pi
#
PI = math.pi
PI_OVER_2 = 0.5 * PI


# ---------------------------------------------------------------------------
# Tiny Numba helpers used inside map expressions
# ---------------------------------------------------------------------------

@njit(types.float64(types.float64), cache=True, fastmath=False)
def i(x):
    return x

@njit(types.float64(types.float64), cache=True, fastmath=False)
def DiracDelta(x):
    # We ignore distributional spikes; enough for Lyapunov purposes.
    return 0.0


@njit(types.float64(types.float64), cache=True, fastmath=False)
def Heaviside(x):
    return 1.0 if x > 0.0 else 0.0


@njit(types.float64(types.float64), cache=True, fastmath=False)
def step(x):
    return 1.0 if x > 0.0 else 0.0


@njit(types.float64(types.float64), cache=True, fastmath=False)
def sign(x):
    return 1.0 if x > 0.0 else -1.0


@njit(types.float64(types.float64), cache=True, fastmath=False)
def Abs(x):
    return np.abs(x)


@njit(types.float64(types.float64), cache=True, fastmath=False)
def re(x):
    return x


@njit(types.float64(types.float64), cache=True, fastmath=False)
def im(x):
    return 0.0


@njit(types.float64(types.float64), cache=True, fastmath=False)
def sec(x):
    return 1.0 / np.cos(x)


@njit(types.float64(types.float64), cache=True, fastmath=False)
def mod1(x):
    return x % 1.0


@njit(types.float64(types.float64, types.float64), cache=True, fastmath=False)
def Mod(x, v):
    return x % v


@njit(types.float64(types.float64, types.float64), cache=True, fastmath=False)
def Derivative(x, v):
    # never actually used; placeholder to keep SymPy happy if it
    # sneaks in.
    return 1.0


@njit(types.float64(types.float64, types.float64), cache=True, fastmath=False)
def apow(x, a):
    return np.sign(x)*np.pow(np.abs(x),a)

@njit(types.float64(types.float64), cache=True, fastmath=False)
def floor(x):
    return math.floor(x)


@njit(types.float64(types.float64), cache=True, fastmath=False)
def ceil(x):
    return math.ceil(x)

@njit(types.float64(types.float64,types.float64), cache=True, fastmath=False)
def abs_cap(x,cap):
    return min(abs(x),cap)*sign(x)

@njit(types.complex128(types.complex128), cache=True, fastmath=False)
def norm(x):
    n = np.abs(x)
    if n<1e-12: return 0+0j
    return x/n

@njit(types.float64(types.float64), cache=True, fastmath=False)
def to01(x):
    xmod1 = x % 1
    return abs(math.cos(2*np.pi*xmod1))

@njit(types.float64(types.float64), cache=True, fastmath=False)
def tri01(x):
    # x mod 1 in [0,1)
    u = x - math.floor(x)
    if u <= 0.5:
        return 2.0 * u        # 0 → 0, 0.5 → 1
    else:
        return 2.0 * (1.0 - u) # 0.5 → 1, 1.0 → 0

@njit(types.float64(types.float64, types.float64), cache=True, fastmath=False)
def tri01p(p, x):
    t = tri01(x)
    if t<=0: return 0.0
    return t ** p

@njit(types.float64(types.float64,types.float64,types.float64,types.float64), cache=True, fastmath=False)
def wavg(x1,x2,v,frac):
    p=frac*to01(v)
    return x1*p+x2*(1-p)

@njit(types.complex128(types.complex128,types.complex128,types.float64,types.float64), cache=True, fastmath=False)
def cwavg(x1,x2,v,frac):
    p=frac*to01(v)
    return x1*p+x2*(1-p)

@njit(types.complex128(types.complex128,types.complex128,types.float64,types.float64), cache=True, fastmath=False)
def cwavgt(x1,x2,v,frac):
    p=frac*tri01(v)
    return x1*p+x2*(1-p)

@njit(types.complex128(types.complex128,types.complex128,types.float64,types.float64,types.float64), cache=True, fastmath=False)
def cwavgtp(x1,x2,v,pow,frac):
    p=frac*tri01p(v,pow)
    return x1*p+x2*(1-p)

@njit(types.int64(types.float64,types.int64,types.int64), cache=True, fastmath=False)
def f2i(x,imin,imax):
    return min(max(int(x),imin),imax)

@njit
def j0s(x):
    ax = abs(x)
    if ax < 8.0:
        y = x*x
        return (1.0 - y*(0.25 - y*(0.046875 - y*(0.003255208333))))
    else:
        z = 8.0/ax
        y = z*z
        xx = ax - 0.7853981633974483096
        return np.sqrt(0.636619772/ax) * (
            np.cos(xx)*(1 - y*(0.001098628627 - y*0.000002073)) -
            np.sin(xx)*(0.01562499997 - y*(0.000143048876 - y*0.000000243))
        )
    
@njit("float64(float64)", fastmath=True, cache=True)
def j0(x):
    ax = x if x >= 0.0 else -x

    # Near zero: J0(x) ≈ 1 - x²/4
    if ax < 1e-8:
        return 1.0 - 0.25 * x * x

    # Power series: J0(x) = Σ (-1)^k (x²/4)^k / (k!)²
    if ax < 20.0:
        y = (x * x) * 0.25
        term = 1.0
        s = 1.0
        # 20 terms is plenty for double precision on this range
        for k in range(1, 20):
            term *= -y / (k * k)
            s += term
        return s

    # Asymptotic for large |x|: J0(x) ~ sqrt(2/(πx)) cos(x - π/4)
    t = ax
    return math.sqrt(2.0 / (math.pi * t)) * math.cos(t - 0.25 * math.pi)


@njit("float64(float64)", fastmath=True, cache=True)
def j1(x):
    ax = x if x >= 0.0 else -x

    # Near zero: J1(x) ≈ x/2
    if ax < 1e-8:
        return 0.5 * x

    # Power series: J1(x) = Σ (-1)^k (x/2)^{2k+1} / (k!(k+1)!)
    if ax < 20.0:
        # k = 0 term
        term = 0.5 * x
        s = term
        y = (x * x) * 0.25
        for k in range(1, 20):
            term *= -y / (k * (k + 1))
            s += term
        return s

    # Asymptotic: J1(x) ~ sqrt(2/(πx)) cos(x - 3π/4)
    t = ax
    val = math.sqrt(2.0 / (math.pi * t)) * math.cos(t - 0.75 * math.pi)
    # J1(-x) = -J1(x)
    return -val if x < 0.0 else val


@njit("float64(float64)", fastmath=True, cache=True)
def i0(x):
    ax = x if x >= 0.0 else -x

    # Near zero: I0(x) ≈ 1 + x²/4
    if ax < 1e-8:
        return 1.0 + 0.25 * x * x

    # Power series: I0(x) = Σ (x²/4)^k / (k!)²
    if ax < 15.0:
        y = 0.25 * x * x
        term = 1.0
        s = 1.0
        for k in range(1, 50):
            term *= y / (k * k)
            s += term
        return s

    # Asymptotic: I0(x) ~ exp(x)/sqrt(2πx)
    t = ax
    val = math.exp(t) / math.sqrt(2.0 * math.pi * t)
    return val


@njit("float64(float64)", fastmath=True, cache=True)
def i1(x):
    ax = x if x >= 0.0 else -x

    # Near zero: I1(x) ≈ x/2
    if ax < 1e-8:
        return 0.5 * x

    # Power series: I1(x) = Σ (x/2)^{2k+1} / (k!(k+1)!)
    if ax < 15.0:
        y = 0.25 * x * x
        term = 0.5 * x  # k=0
        s = term
        for k in range(1, 50):
            term *= y / (k * (k + 1))
            s += term
        return s

    # Asymptotic: I1(x) ~ exp(x)/sqrt(2πx)
    t = ax
    val = math.exp(t) / math.sqrt(2.0 * math.pi * t)
    # I1(-x) = -I1(x)
    return -val if x < 0.0 else val




@njit(types.float64(types.float64, types.float64, types.float64), fastmath=True, cache=True)
def _airy_series(x, c0, c1):
    """
    Generic power series for a solution of y'' - x y = 0
    with y(0) = c0, y'(0) = c1.

    Uses the recurrence from the ODE:
        c_{n+3} = c_n / ((n+3)(n+2))
    split into two nonzero branches n ≡ 0,1 (mod 3).
    """
    x3 = x * x * x
    y = 0.0
    kmax = 50
    tol = 1e-16

    # branch n0 = 0: n = 0, 3, 6, ...
    n = 0.0
    term = c0
    y += term
    for _ in range(kmax):
        denom = (n + 3.0) * (n + 2.0)
        term *= x3 / denom
        y += term
        n += 3.0
        if math.fabs(term) < tol:
            break

    # branch n0 = 1: n = 1, 4, 7, ...
    n = 1.0
    term = c1 * x
    y += term
    for _ in range(kmax):
        denom = (n + 3.0) * (n + 2.0)
        term *= x3 / denom
        y += term
        n += 3.0
        if math.fabs(term) < tol:
            break

    return y


@njit(types.float64(types.float64), fastmath=True, cache=True)
def airy_ai(x):
    """
    Numba-friendly Airy Ai(x).

    - |x| <= 5: power series around 0
    - x  >  5: decaying asymptotic
    - x  < -5: oscillatory asymptotic
    """
    if x > 5.0:
        # Ai(x) ~ (1 / (2√π)) x^{-1/4} exp(-2/3 x^{3/2})
        t = (2.0 / 3.0) * (x ** 1.5)
        amp = 1.0 / (2.0 * math.sqrt(math.pi) * (x ** 0.25))
        return amp * math.exp(-t)
    elif x < -5.0:
        # Ai(x) ~ (1 / (√π |x|^{1/4})) * sin(2/3 |x|^{3/2} + π/4)
        z = -x
        t = (2.0 / 3.0) * (z ** 1.5)
        amp = 1.0 / (math.sqrt(math.pi) * (z ** 0.25))
        return amp * math.sin(t + 0.25 * math.pi)
    else:
        return _airy_series(x, AI0, AI0P)


@njit(types.float64(types.float64), fastmath=True, cache=True)
def airy_bi(x):
    """
    Numba-friendly Airy Bi(x).

    - |x| <= 5: power series around 0
    - x  >  5: growing asymptotic
    - x  < -5: oscillatory asymptotic
    """
    if x > 5.0:
        # Bi(x) ~ (1 / √π) x^{-1/4} exp(+2/3 x^{3/2})
        t = (2.0 / 3.0) * (x ** 1.5)
        amp = 1.0 / (math.sqrt(math.pi) * (x ** 0.25))
        return amp * math.exp(t)
    elif x < -5.0:
        # Bi(x) ~ (1 / (√π |x|^{1/4})) * cos(2/3 |x|^{3/2} + π/4)
        z = -x
        t = (2.0 / 3.0) * (z ** 1.5)
        amp = 1.0 / (math.sqrt(math.pi) * (z ** 0.25))
        return amp * math.cos(t + 0.25 * math.pi)
    else:
        return _airy_series(x, BI0, BI0P)
    

@njit(types.float64(types.float64), fastmath=True, cache=True)
def fresnel_c(x):
    """
    Numba-friendly Fresnel C(x) = ∫_0^x cos(π t^2 / 2) dt

    - |x| <= 2: power series around 0
    - |x|  > 2: simple asymptotic (good qualitatively)
    """
    # C(x) is odd: C(-x) = -C(x)
    sign = 1.0
    if x < 0.0:
        sign = -1.0
        x = -x

    if x <= 2.0:
        # Power series:
        # C(x) = Σ_{k=0}^∞ (-1)^k ( (π/2)^{2k} x^{4k+1} ) / ( (2k)! (4k+1) )
        max_k = 10
        result = 0.0

        # Precompute some powers iteratively
        x2 = x * x
        x4 = x2 * x2
        p = 1.0                    # (π/2)^(2k)
        xpow = x                   # x^(4k+1), start at k=0 -> x^1
        sign_k = 1.0               # (-1)^k
        fact2k = 1.0               # (2k)! (start with 0! = 1)

        for k in range(max_k):
            term = sign_k * p * xpow / (fact2k * (4.0 * k + 1.0))
            result += term

            # Prepare next k
            # sign
            sign_k = -sign_k

            # (π/2)^{2(k+1)} = (π/2)^{2k} * (π/2)^2
            p *= (PI_OVER_2 * PI_OVER_2)

            # x^{4(k+1)+1} = x^{4k+1} * x^4
            xpow *= x4

            # (2(k+1))! from (2k)!:
            # multiply by (2k+1)*(2k+2)
            n1 = 2 * k + 1
            n2 = 2 * k + 2
            fact2k *= n1 * n2

        return sign * result

    # Asymptotic region: x > 2
    # Use a simple two-term asymptotic:
    # C(x) ≈ 1/2 + f(x)*sin(π x^2/2) - g(x)*cos(π x^2/2)
    # with f(x) ≈ 1/(π x), g(x) ≈ 1/(π^2 x^3)
    t = PI_OVER_2 * x * x
    sin_t = math.sin(t)
    cos_t = math.cos(t)
    f = 1.0 / (PI * x)
    g = 1.0 / (PI * PI * x * x * x)
    result = 0.5 + f * sin_t - g * cos_t

    return sign * result


@njit(types.float64(types.float64), fastmath=True, cache=True)
def fresnel_s(x):
    """
    Numba-friendly Fresnel S(x) = ∫_0^x sin(π t^2 / 2) dt

    - |x| <= 2: power series around 0
    - |x|  > 2: simple asymptotic (good qualitatively)
    """
    # S(x) is odd: S(-x) = -S(x)
    sign = 1.0
    if x < 0.0:
        sign = -1.0
        x = -x

    if x <= 2.0:
        # Power series:
        # S(x) = Σ_{k=0}^∞ (-1)^k ( (π/2)^{2k+1} x^{4k+3} ) / ( (2k+1)! (4k+3) )
        max_k = 10
        result = 0.0

        x2 = x * x
        x4 = x2 * x2
        p = PI_OVER_2              # (π/2)^(2k+1); start k=0 -> (π/2)^1
        xpow = x * x2              # x^(4k+3), start at k=0 -> x^3
        sign_k = 1.0               # (-1)^k
        fact2k1 = 1.0              # (2k+1)!; start at k=0 -> 1! = 1

        for k in range(max_k):
            term = sign_k * p * xpow / (fact2k1 * (4.0 * k + 3.0))
            result += term

            # Prepare next k
            sign_k = -sign_k
            p *= (PI_OVER_2 * PI_OVER_2)
            xpow *= x4

            # (2(k+1)+1)! from (2k+1)!:
            # multiply by (2k+2)*(2k+3)
            n1 = 2 * k + 2
            n2 = 2 * k + 3
            fact2k1 *= n1 * n2

        return sign * result

    # Asymptotic region: x > 2
    # S(x) ≈ 1/2 - f(x)*cos(π x^2/2) - g(x)*sin(π x^2/2)
    t = PI_OVER_2 * x * x
    sin_t = math.sin(t)
    cos_t = math.cos(t)
    f = 1.0 / (PI * x)
    g = 1.0 / (PI * PI * x * x * x)
    result = 0.5 - f * cos_t - g * sin_t

    return sign * result




@njit(types.float64(types.float64), fastmath=True, cache=True)
def zeta(s):
    """
    Numba-safe Riemann zeta ζ(s) for real s with s > 0, s ≠ 1.

    Uses the alternating Dirichlet eta series:
        η(s) = Σ_{n>=1} (-1)^{n-1} / n^s
        ζ(s) = η(s) / (1 - 2^{1-s})
    """
    # outside domain: just return NaN
    if s <= 0.0:
        return math.nan

    # zeta has a pole at s=1
    if abs(s - 1.0) < 1e-10:
        # large placeholder instead of +∞
        return 1e6

    # compute eta(s)
    eta = 0.0
    sign = 1.0
    for n in range(1, ZETA_MAX_N + 1):
        term = sign / (n ** s)
        eta += term
        if math.fabs(term) < ZETA_TOL:
            break
        sign = -sign

    # denominator 1 - 2^{1-s}
    denom = 1.0 - (2.0 ** (1.0 - s))
    return eta / denom


@njit(types.float64(types.float64), fastmath=True, cache=True)
def lambertw(x):
    """
    Real principal branch of Lambert W, W0(x), for x >= -1/e.

    Uses Halley's method with a simple initial guess.
    Suitable for maps, not for hardcore special-function work.
    """
    # domain check: real W0 exists for x >= -1/e
    x_min = -1.0 / math.e
    if x < x_min: return math.nan
    if x == 0.0: return 0.0
    if x < 1.0:
        w = x        # near zero, W(x) ~ x
    else:
        w = math.log(x) - math.log(math.log(x + 1.0))
    for _ in range(40):
        e = math.exp(w)
        we = w * e
        f = we - x            # f(w) = w*e^w - x
        if math.fabs(f) < 1e-14:
            break
        wp1 = w + 1.0
        if math.fabs(wp1) < 1e-7:
            wp1 = 1e-7 if wp1 >= 0.0 else -1e-7
        denom = e * wp1 - (wp1 + 1.0) * f / (2.0 * wp1)
        w = w - f / denom
    return w

@njit(types.float64(types.float64, types.float64), fastmath=True, cache=True)
def gammainc(a, x):
    """
    Regularized lower incomplete gamma P(a, x) for a > 0, x >= 0.

        P(a,x) = γ(a,x) / Γ(a)

    Uses:
    - series for x < a + 1
    - continued fraction for x >= a + 1
    """
    if a <= 0.0 or x < 0.0: return math.nan
    if x == 0.0: return 0.0
    gln = math.lgamma(a)
    if x < a + 1.0:
        ap = a
        summ = 1.0 / a
        delta = summ
        for _ in range(1000):
            ap += 1.0
            delta *= x / ap
            summ += delta
            if math.fabs(delta) < math.fabs(summ) * 1e-15:
                break
        return summ * math.exp(-x + a * math.log(x) - gln)

    # Continued fraction for Q(a,x) = Γ(a,x)/Γ(a); then P = 1 - Q
    b = x + 1.0 - a
    c = 1.0 / 1e-30
    d = 1.0 / b
    h = d
    for i in range(1, 2000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if math.fabs(d) < 1e-30:
            d = 1e-30
        c = b + an / c
        if math.fabs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if math.fabs(delta - 1.0) < 1e-15:
            break

    Q = math.exp(-x + a * math.log(x) - gln) * h
    P = 1.0 - Q
    return P

SQRT_PI = math.sqrt(math.pi)

@njit(types.float64(types.float64), fastmath=True, cache=True)
def dawson(x):
    ax = x if x >= 0.0 else -x
    if ax < 0.5:
        x2 = x * x
        x4 = x2 * x2
        x6 = x4 * x2
        x8 = x4 * x4
        return (x
                - (2.0 / 3.0) * x * x2
                + (4.0 / 15.0) * x * x4
                - (8.0 / 105.0) * x * x6
                + (16.0 / 945.0) * x * x8)
    ax2 = ax * ax
    ax3 = ax * ax2
    ax5 = ax3 * ax2
    ax7 = ax5 * ax2
    f = (1.0 / (2.0 * ax)
         + 1.0 / (4.0 * ax3)
         + 3.0 / (8.0 * ax5)
         + 15.0 / (16.0 * ax7))
    return f if x >= 0.0 else -f

@njit(types.float64(types.float64), fastmath=True, cache=True)
def erfi(x):
    return (2.0 / SQRT_PI) * math.exp(x * x) * dawson(x)

@njit(types.float64(types.int64, types.float64), fastmath=True, cache=True)
def legendre(n, x):
    if n < 0:  return math.nan
    if n == 0: return 1.0
    if n == 1: return x
    Pnm1 = 1.0   # P_0
    Pn = x       # P_1
    for k in range(1, n):
        kf = float(k)
        Pnp1 = ((2.0 * kf + 1.0) * x * Pn - kf * Pnm1) / (kf + 1.0)
        Pnm1 = Pn
        Pn = Pnp1
    return Pn

@njit(types.float64(types.float64), fastmath=True, cache=True)
def le1(x):
    return legendre(1,x)

@njit(types.float64(types.float64), fastmath=True, cache=True)
def le2(x):
    return legendre(2,x)

@njit(types.float64(types.float64), fastmath=True, cache=True)
def le3(x):
    return legendre(3,x)

@njit(types.float64(types.float64), fastmath=True, cache=True)
def le4(x):
    return legendre(4,x)


# ============================================================
# 1. Exponential integral Ei(x)
# ============================================================

@njit(types.float64(types.float64), fastmath=True, cache=True)
def ei(x):
    """
    Numba-friendly exponential integral Ei(x), real-valued.

    - Uses the series:
        Ei(x) = γ + ln|x| + Σ_{k=1..∞} x^k / (k * k!)
      for |x| <= 4, with a truncation tolerance.

    - For |x| > 4, uses a simple asymptotic:
        Ei(x) ≈ e^x / x * (1 + 1/x + 2/x^2 + 6/x^3 + 24/x^4)

    This is intended for dynamical systems / maps
    (nice shapes), not high-precision analysis.
    """
    if x == 0.0:
        # Logarithmic singularity at 0; use a big negative value
        return -1.0e308

    ax = x if x >= 0.0 else -x

    # Series region
    if ax <= 4.0:
        # Ei(x) = γ + ln|x| + Σ_{k=1}^∞ x^k / (k * k!)
        max_k = 100
        tol = 1e-16

        term_power = x  # x^1
        fact = 1.0      # 1! after first loop step
        series_sum = 0.0

        for k in range(1, max_k + 1):
            fact *= k            # k!
            term = term_power / (k * fact)
            series_sum += term
            if math.fabs(term) < tol:
                break
            term_power *= x      # x^{k+1}

        return EULER_GAMMA + math.log(ax) + series_sum

    # Asymptotic region |x| > 4
    invx = 1.0 / x
    invx2 = invx * invx

    # 1 + 1/x + 2/x^2 + 6/x^3 + 24/x^4
    poly = 1.0 \
           + invx \
           + 2.0 * invx2 \
           + 6.0 * invx2 * invx \
           + 24.0 * invx2 * invx2

    return math.exp(x) * invx * poly


# ============================================================
# 2. Sine integral Si(x) and Cosine integral Ci(x)
# ============================================================

@njit(types.float64(types.float64), fastmath=True, cache=True)
def si(x):
    """
    Numba-friendly Sine integral Si(x) = ∫_0^x (sin t / t) dt (principal value).

    - |x| <= 4: power series
        Si(x) = Σ_{k=0}^∞ (-1)^k x^{2k+1} / ((2k+1)(2k+1)!)
    - |x|  > 4: asymptotic expansion
        Si(x) ≈ π/2
                 - cos x * (1/x - 2!/x^3 + 4!/x^5)
                 - sin x * (1/x^2 - 3!/x^4 + 5!/x^6)

    Odd function: Si(-x) = -Si(x).
    """
    if x == 0.0:
        return 0.0

    # Use symmetry
    sign = 1.0
    if x < 0.0:
        sign = -1.0
        x = -x

    # Series region
    if x <= 4.0:
        max_k = 40
        tol = 1e-16

        # term_k = (-1)^k x^{2k+1} / ((2k+1)(2k+1)!)
        result = 0.0
        pow_x = x          # x^(2k+1), start with k=0 -> x^1
        fact = 1.0         # (2k+1)!; for k=0: 1! = 1
        sign_k = 1.0

        for k in range(max_k):
            denom = (2.0 * k + 1.0) * fact
            term = sign_k * pow_x / denom
            result += term

            if math.fabs(term) < tol:
                break

            # next k
            sign_k = -sign_k
            # increase power by x^2
            pow_x *= x * x
            # update factorial: (2k+3)! from (2k+1)!:
            # multiply by (2k+2)(2k+3)
            n1 = 2.0 * k + 2.0
            n2 = 2.0 * k + 3.0
            fact *= n1 * n2

        return sign * result

    # Asymptotic region
    # Si(x) ≈ π/2 - cos x * A(x) - sin x * B(x)
    # where:
    #   A(x) = 1/x - 2!/x^3 + 4!/x^5
    #   B(x) = 1/x^2 - 3!/x^4 + 5!/x^6
    invx = 1.0 / x
    invx2 = invx * invx
    invx3 = invx2 * invx
    invx4 = invx2 * invx2
    invx5 = invx3 * invx2
    invx6 = invx3 * invx3

    # factorials: 2! = 2, 3! = 6, 4! = 24, 5! = 120
    A = invx - 2.0 * invx3 + 24.0 * invx5
    B = invx2 - 6.0 * invx4 + 120.0 * invx6

    sx = math.sin(x)
    cx = math.cos(x)

    result = 0.5 * PI - cx * A - sx * B
    return sign * result


@njit(types.float64(types.float64), fastmath=True, cache=True)
def ci(x):
    """
    Numba-friendly Cosine integral Ci(x).

    For x > 0:
        Ci(x) = γ + ln x + ∫_0^x (cos t - 1) / t dt

    Implementation:
    - |x| <= 4: series
        Ci(x) = γ + ln|x| + Σ_{k=1}^∞ (-1)^k x^{2k} / (2k (2k)!)
    - |x|  > 4: asymptotic
        Ci(x) ≈   sin x * (1/x - 2!/x^3 + 4!/x^5)
                + cos x * (1/x^2 - 3!/x^4 + 5!/x^6)

    We use Ci(|x|) for negative x to keep it real-valued.
    """
    if x == 0.0:
        # logarithmic singularity
        return -1.0e308

    if x < 0.0:
        x = -x

    # Series region
    if x <= 4.0:
        max_k = 40
        tol = 1e-16

        result = 0.0
        pow_x2 = x * x  # x^(2k), start with k=1 -> x^2
        fact = 2.0      # (2k)!; for k=1 -> 2! = 2
        sign_k = -1.0   # (-1)^1

        for k in range(1, max_k + 1):
            denom = (2.0 * k) * fact
            term = sign_k * pow_x2 / denom
            result += term

            if math.fabs(term) < tol:
                break

            # next k
            sign_k = -sign_k
            # update power: x^{2(k+1)} = x^{2k} * x^2
            pow_x2 *= x * x
            # update factorial: (2(k+1))! from (2k)!:
            # multiply by (2k+1)(2k+2)
            n1 = 2.0 * k + 1.0
            n2 = 2.0 * k + 2.0
            fact *= n1 * n2

        return EULER_GAMMA + math.log(x) + result

    # Asymptotic region
    # Ci(x) ≈ sin x * A(x) + cos x * B(x)
    # where A, B as above
    invx = 1.0 / x
    invx2 = invx * invx
    invx3 = invx2 * invx
    invx4 = invx2 * invx2
    invx5 = invx3 * invx2
    invx6 = invx3 * invx3

    A = invx - 2.0 * invx3 + 24.0 * invx5
    B = invx2 - 6.0 * invx4 + 120.0 * invx6

    sx = math.sin(x)
    cx = math.cos(x)

    return sx * A + cx * B


# ============================================================
# 3. Gudermannian gd(x)
# ============================================================

@njit(types.float64(types.float64), fastmath=True, cache=True)
def gd(x):
    return 2.0 * math.atan(math.tanh(0.5 * x))



# ============================================================
# 1. Spherical Bessel j_n(x)
# ============================================================

@njit(types.float64(types.int64, types.float64), fastmath=True, cache=True)
def sbessel(n, x):
    """
    Spherical Bessel function j_n(x) for integer n >= 0 (real x).

    Uses:
        j0(x) = sin x / x
        j1(x) = sin x / x^2 - cos x / x
        j_{n+1}(x) = (2n+1)/x * j_n(x) - j_{n-1}(x)

    For |x| very small, uses simple small-x limits:
        j0(0) = 1, j1(0) ~ x/3, j_n(0) ~ 0 for n>=2
    """
    if n < 0:
        return math.nan

    ax = x if x >= 0.0 else -x

    # small-x handling
    if ax < 1e-8:
        if n == 0:
            return 1.0
        elif n == 1:
            return x / 3.0
        else:
            return 0.0

    # j0
    if n == 0:
        return math.sin(x) / x

    # j1
    if n == 1:
        return math.sin(x) / (x * x) - math.cos(x) / x

    # upward recurrence
    jnm1 = math.sin(x) / x                                      # j0
    jn = math.sin(x) / (x * x) - math.cos(x) / x                # j1

    for k in range(1, n):
        kf = float(k)
        jnp1 = ((2.0 * kf + 1.0) / x) * jn - jnm1
        jnm1 = jn
        jn = jnp1

    return jn


@njit(types.float64(types.float64), fastmath=True, cache=True)
def sbs1(x):
    return sbessel(1,x)

@njit(types.float64(types.float64), fastmath=True, cache=True)
def sbs2(x):
    return sbessel(2,x)

@njit(types.float64(types.float64), fastmath=True, cache=True)
def sbs3(x):
    return sbessel(3,x)

@njit(types.float64(types.float64), fastmath=True, cache=True)
def sbs4(x):
    return sbessel(4,x)

# ============================================================
# 2. Chebyshev polynomials T_n(x), U_n(x)
# ============================================================

@njit(types.float64(types.int64, types.float64), fastmath=True, cache=True)
def chebt(n, x):
    """
    Chebyshev polynomial of the first kind T_n(x).

        T_0(x) = 1
        T_1(x) = x
        T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x)
    """
    if n < 0:
        return math.nan
    if n == 0:
        return 1.0
    if n == 1:
        return x

    Tnm1 = 1.0  # T_0
    Tn = x      # T_1

    for _ in range(1, n):
        Tnp1 = 2.0 * x * Tn - Tnm1
        Tnm1 = Tn
        Tn = Tnp1

    return Tn

@njit(types.float64(types.float64), fastmath=True, cache=True)
def cht1(x):
    return chebt(1,x)

@njit(types.float64(types.float64), fastmath=True, cache=True)
def cht2(x):
    return chebt(2,x)

@njit(types.float64(types.float64), fastmath=True, cache=True)
def cht3(x):
    return chebt(3,x)

@njit(types.float64(types.float64), fastmath=True, cache=True)
def cht4(x):
    return chebt(4,x)


@njit(types.float64(types.int64, types.float64), fastmath=True, cache=True)
def chebu(n, x):
    """
    Chebyshev polynomial of the second kind U_n(x).

        U_0(x) = 1
        U_1(x) = 2x
        U_{n+1}(x) = 2x U_n(x) - U_{n-1}(x)
    """
    if n < 0:
        return math.nan
    if n == 0:
        return 1.0
    if n == 1:
        return 2.0 * x

    Unm1 = 1.0        # U_0
    Un = 2.0 * x      # U_1

    for _ in range(1, n):
        Unp1 = 2.0 * x * Un - Unm1
        Unm1 = Un
        Un = Unp1

    return Un

@njit(types.float64(types.float64), fastmath=True, cache=True)
def chu1(x):
    return chebu(1,x)

@njit(types.float64(types.float64), fastmath=True, cache=True)
def chu2(x):
    return chebu(2,x)

@njit(types.float64(types.float64), fastmath=True, cache=True)
def chu3(x):
    return chebu(3,x)

@njit(types.float64(types.float64), fastmath=True, cache=True)
def chu4(x):
    return chebu(4,x)


# ============================================================
# 3. Clausen function Cl2(theta) = sum_{k>=1} sin(kθ)/k^2
# ============================================================

@njit(types.float64(types.float64), fastmath=True, cache=True)
def clausen(theta):
    """
    Clausen function Cl2(theta) ≈ sum_{k>=1} sin(kθ)/k^2
    (Fourier series truncation, good for dynamics)
    """
    # reduce angle to [-π, π] for better convergence
    # use % instead of math.fmod (Numba-safe)
    t = theta % TWO_PI
    if t > PI:
        t -= TWO_PI
    elif t < -PI:
        t += TWO_PI

    max_k =10
    tol = 1e-14

    s = 0.0
    for k in range(1, max_k + 1):
        kf = float(k)
        term = math.sin(kf * t) / (kf * kf)
        s += term
        if math.fabs(term) < tol:
            break

    return s


# ============================================================
# 4. Dilogarithm Li2(x) for real x in [-1, 1]
# ============================================================

@njit(types.float64(types.float64), fastmath=True, cache=True)
def _li2_series(x):
    """
    Power series for Li2(x) valid for |x| <= 0.5:

        Li2(x) = Σ_{k>=1} x^k / k^2
    """
    max_k = 200
    tol = 1e-16

    s = 0.0
    term = x
    for k in range(1, max_k + 1):
        if k > 1:
            term *= x
        add = term / (float(k) * float(k))
        s += add
        if math.fabs(add) < tol:
            break
    return s


@njit(types.float64(types.float64), fastmath=True, cache=True)
def li2(x):
    """
    Dilogarithm Li2(x) for real x, approx domain [-1, 1].

    Uses:
    - series Li2(x) = Σ x^k / k^2   for x <= 0.5
    - reflection for x in (0.5, 1):

        Li2(x) = π^2/6 - ln(x) ln(1-x) - Li2(1 - x)

    For x < -1 or x > 1 we just return NaN.
    """
    if x == 1.0:
        # Li2(1) = π^2 / 6
        return (PI * PI) / 6.0

    if x < -1.0 or x > 1.0:
        return math.nan

    # x <= 0.5 (including negatives): series converges decently
    if x <= 0.5:
        return _li2_series(x)

    # 0.5 < x < 1: use reflection
    one_minus_x = 1.0 - x
    # one_minus_x in (0, 0.5]
    base = (PI * PI) / 6.0 - math.log(x) * math.log(one_minus_x)
    corr = _li2_series(one_minus_x)
    return base - corr


# ============================================================
# 5. Logistic integral L(x) = ∫_0^x dt / (1 + e^{-t})
# ============================================================

@njit(types.float64(types.float64), fastmath=True, cache=True)
def lint(x):
    """
    Logistic integral:

        L(x) = ∫_0^x dt / (1 + e^{-t})

    Closed form:
        1 / (1 + e^{-t}) = e^t / (1 + e^t)
        => integral = ln(1 + e^t) + C

    So:
        L(x) = ln(1 + e^x) - ln(1 + e^0)
             = ln(1 + e^x) - ln 2
    """
    # use log1p for better stability
    return math.log1p(math.exp(x)) - math.log(2.0)

# ============================================================
# Hermite polynomials H_n(x) (physicists' convention)
# ============================================================

@njit(types.float64(types.int64, types.float64), fastmath=True, cache=True)
def hermite(n, x):
    """
    Physicists' Hermite polynomials H_n(x):

        H_0(x) = 1
        H_1(x) = 2x
        H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x)

    n >= 0
    """
    if n < 0:
        return math.nan
    if n == 0:
        return 1.0
    if n == 1:
        return 2.0 * x

    Hnm1 = 1.0        # H_0
    Hn   = 2.0 * x    # H_1

    for k in range(1, n):
        kf = float(k)
        Hnp1 = 2.0 * x * Hn - 2.0 * kf * Hnm1
        Hnm1 = Hn
        Hn   = Hnp1

    return Hn

@njit(types.float64(types.float64), fastmath=True, cache=True)
def he1(x):
    return hermite(1,x)

@njit(types.float64(types.float64), fastmath=True, cache=True)
def he2(x):
    return hermite(2,x)

@njit(types.float64(types.float64), fastmath=True, cache=True)
def he3(x):
    return hermite(3,x)

@njit(types.float64(types.float64), fastmath=True, cache=True)
def he4(x):
    return hermite(4,x)

# ============================================================
# Laguerre polynomials L_n(x) (α = 0)
# ============================================================

@njit(types.float64(types.int64, types.float64), fastmath=True, cache=True)
def laguerre(n, x):
    """
    Laguerre polynomials L_n(x) with α = 0:

        L_0(x) = 1
        L_1(x) = 1 - x
        (n+1)L_{n+1}(x) = (2n+1 - x)L_n(x) - n L_{n-1}(x)

    n >= 0, x >= 0 typically.
    """
    if n < 0:
        return math.nan
    if n == 0:
        return 1.0
    if n == 1:
        return 1.0 - x

    Lnm1 = 1.0        # L_0
    Ln   = 1.0 - x    # L_1

    for k in range(1, n):
        kf = float(k)
        Lnp1 = ((2.0 * kf + 1.0 - x) * Ln - kf * Lnm1) / (kf + 1.0)
        Lnm1 = Ln
        Ln   = Lnp1

    return Ln

@njit(types.float64(types.float64), fastmath=True, cache=True)
def la1(x):
    return laguerre(1,x)

@njit(types.float64(types.float64), fastmath=True, cache=True)
def la2(x):
    return laguerre(2,x)

@njit(types.float64(types.float64), fastmath=True, cache=True)
def la3(x):
    return laguerre(3,x)

@njit(types.float64(types.float64), fastmath=True, cache=True)
def la4(x):
    return laguerre(4,x)

@njit(types.float64(types.int64, types.float64, types.float64, types.float64), fastmath=True, cache=True)
def jacobi(n, alpha, beta, x):
    """
    Jacobi polynomial P_n^{(alpha, beta)}(x), n >= 0, real alpha,beta,x.

    Recurrence (Szegő / Abramowitz-Stegun):

        P_0^{(α,β)}(x) = 1
        P_1^{(α,β)}(x) = 0.5 * [ (2+α+β)x + (α-β) ]

    For n >= 1:

        2(n+1)(n+α+β+1)(2n+α+β) P_{n+1}(x) =
            (2n+α+β+1) [ (2n+α+β+2)(2n+α+β)x + α^2 - β^2 ] P_n(x)
          - 2(n+α)(n+β)(2n+α+β+2) P_{n-1}(x)

    This implementation is intended for small n (e.g. n <= 10–20) as
    shape functions in iterated maps, not for extreme parameter regimes.

    Legendre Jacobi(N,0,0,x)
    ChebT Jacobi(N,-1/2,-1/2,x)
    Gegenbauer Jacobi(N,lambda-1/2,1/2,x)

    """
    if n < 0:
        return math.nan

    # P_0
    if n == 0:
        return 1.0

    # P_1
    if n == 1:
        return 0.5 * ((2.0 + alpha + beta) * x + (alpha - beta))

    # P_0 and P_1 as starting values
    Pnm1 = 1.0
    Pn = 0.5 * ((2.0 + alpha + beta) * x + (alpha - beta))

    for k in range(1, n):
        kf = float(k)
        two_k_ab = 2.0 * kf + alpha + beta

        # coefficients in the recurrence
        A = 2.0 * (kf + 1.0) * (kf + alpha + beta + 1.0) * two_k_ab
        B = (two_k_ab + 1.0) * ((two_k_ab + 2.0) * two_k_ab * x +
                                (alpha * alpha - beta * beta))
        C = 2.0 * (kf + alpha) * (kf + beta) * (two_k_ab + 2.0)

        # P_{k+1}
        Pnp1 = (B * Pn - C * Pnm1) / A

        Pnm1 = Pn
        Pn = Pnp1

    return Pn

@njit(types.complex128(
    types.int64, 
    types.complex128, 
    types.complex128, 
    types.complex128
), fastmath=True, cache=True)
def meixpol(n, lam, phi, x):
    if n < 0: return math.nan
    if n == 0: return 1.0
    sinphi = np.sin(phi)
    cosphi = np.cos(phi)
    if n == 1: return 2.0 * (lam * cosphi + x * sinphi)
    Pnm1 = 1.0
    Pn   = 2.0 * (lam * cosphi + x * sinphi)
    for k in range(1, n):
        kf = float(k)
        numerator = 2.0 * (x * sinphi + (kf + lam) * cosphi) * Pn - (kf + 2.0 * lam - 1.0) * Pnm1
        Pnp1 = numerator / (kf + 1.0)
        Pnm1, Pn = Pn, Pnp1
    return Pn

@njit(types.complex128(
    types.int64,
    types.complex128,
    types.complex128,
    types.complex128,
    types.complex128,
    types.complex128
),fastmath=True, cache=True)
def hahn(n, a, b, c, d, x):
    if n < 0: return np.nan + 0j
    if n == 0: return 1.0 + 0j
    s = a + b + c + d
    A_prev = - (a + c) * (a + d) / s
    C0 = 0.0 + 0j
    pnm1 = 1.0 + 0j
    pn   = x - 1j * (A_prev + C0)
    if n == 1: return pn
    for k in range(1, n):
        kf = float(k)
        numA = (kf + s - 1.0) * (kf + a + c) * (kf + a + d)
        denA = (2.0*kf + s - 1.0) * (2.0*kf + s)
        Ak = -numA / denA
        numC = kf * (kf + b + c - 1.0) * (kf + b + d - 1.0)
        denC = (2.0*kf + s - 2.0) * (2.0*kf + s - 1.0)
        Ck = numC / denC
        pnp1 = (x - 1j*(Ak + Ck)) * pn + A_prev * Ck * pnm1
        pnm1 = pn
        pn = pnp1
        A_prev = Ak
    return pn

# ============================================================
# sinc, sinhc, sech — cheap but very nice shapes
# ============================================================

@njit(types.float64(types.float64), fastmath=True, cache=True)
def sinc(x):
    """
    Unnormalized sinc: sinc(x) = sin(x)/x, with a safe limit at x=0.

    You can always call sinc(pi * x) if you want the normalized sinc.
    """
    ax = x if x >= 0.0 else -x
    if ax < 1e-8:
        # Taylor: sin x / x ≈ 1 - x^2/6
        return 1.0 - (x * x) / 6.0
    return math.sin(x) / x


@njit(types.float64(types.float64), fastmath=True, cache=True)
def sinhc(x):
    """
    Hyperbolic sinc: sinhc(x) = sinh(x)/x, with limit 1 at x=0.
    """
    ax = x if x >= 0.0 else -x
    if ax < 1e-8:
        # Taylor: sinh x / x ≈ 1 + x^2/6
        return 1.0 + (x * x) / 6.0
    return math.sinh(x) / x


@njit(types.float64(types.float64), fastmath=True, cache=True)
def sech(x):
    """
    Hyperbolic secant: sech(x) = 1 / cosh(x).
    """
    return 1.0 / math.cosh(x)


NS = {
    "i": i,
    "step": step,
    "Heaviside": Heaviside,
    "DiracDelta": DiracDelta,
    "sign": sign,
    "Abs": Abs,
    "abs": Abs,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "sec": sec,
    "cosh": np.cosh,
    "sinh": np.sinh,
    "exp": np.exp,
    "pow": np.power,
    "apow": apow,
    "log": np.log,
    "mod1": mod1,
    "Mod": Mod,
    "Derivative": Derivative,
    "re": re,
    "im": im,
    "pi": np.pi,
    "max": max,
    "min": min,
    "floor": floor,
    "ceil": ceil,
    "round": round,
    "to01": to01,
    "wavg": wavg,
    "cwavg": cwavg,
    "cwavgt": cwavgt, # complex, weighted average, tri
    "cwavgtp": cwavgtp, # complex, weighted average, power tri
    "f2i": f2i,
    "abs_cap": abs_cap,
    "norm": norm,
    "j0s": j0s,
    "j0": j0,
    "j1": j1,
    "i0": i0,
    "i1": i1,
    "lgamma": math.lgamma,
    "aira": airy_ai,
    "airb": airy_bi,
    "frec": fresnel_c,
    "fres": fresnel_s,
    "zeta": zeta,
    "lambertw": lambertw,
    "gammainc": gammainc,
    "dawson": dawson,
    "ei": ei,
    "si": si,
    "ci": ci,
    "gd": gd,
    "clausen": clausen,
    "li2": li2,
    "lint": lint,
    "legendre": legendre,
    "le1": le1,
    "le2": le2,
    "le3": le3,
    "le4": le4,
    "hermite": hermite,
    "he1": he1,
    "he2": he2,
    "he3": he3,
    "he4": he4,
    "laguerre": laguerre,
    "la1": la1,
    "la2": la2,
    "la3": la3,
    "la4": la4,
    "sbessel": sbessel,
    "sbs1": sbs1,
    "sbs2": sbs2,
    "sbs3": sbs3,
    "sbs4": sbs4,
    "chebt": chebt,
    "cht1": cht1,
    "cht2": cht2,
    "cht3": cht3,
    "cht4": cht4,
    "chebu": chebu,
    "chu1": chu1,
    "chu2": chu2,
    "chu3": chu3,
    "chu4": chu4,
    "jacobi": jacobi,
    "meixpol": meixpol,
    "hahn": hahn,
    "sech": sech,
    "sinc": sinc,
    "sinhc": sinhc,
    "mexhat": wavelet.mexican_hat,
    "morlet": wavelet.morlet,
    "shannon": wavelet.shannon,
    "gdev": wavelet.gauss_deriv,
    "gdevs": wavelet.gauss_deriv_scaled,
    "np": np,
    "math": math,
    "cmath": cmath,
}

