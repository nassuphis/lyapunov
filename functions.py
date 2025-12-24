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

_C = types.complex128

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

@njit(cache=True, fastmath=True)
def polevl(x, coef):
    # Evaluate polynomial with coef[0]..coef[n-1] (degree n-1), Cephes order.
    # y = coef[0]*x^(n-1) + ... + coef[n-2]*x + coef[n-1]
    y = 0.0
    for c in coef:
        y = y * x + c
    return y

@njit(cache=True, fastmath=True)
def p1evl(x, coef):
    # Evaluate (x + coef[0]) * x^(n-1) + ...? Cephes p1evl:
    # y = (x + coef[0]) * x^(n-1) + coef[1]*x^(n-2) + ... + coef[n-1]
    y = x + coef[0]
    for i in range(1, len(coef)):
        y = y * x + coef[i]
    return y

@njit(cache=True, fastmath=True)
def chbevl(x, coef):
    # Chebyshev evaluation on [-1,1] (Cephes chbevl, Clenshaw)
    b0 = 0.0
    b1 = 0.0
    b2 = 0.0
    for c in coef:
        b2 = b1
        b1 = b0
        b0 = x * b1 - b2 + c
    return 0.5 * (b0 - b2)

# Bessel, order 0     
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
    
# Bessel, order 1
@njit(cache=True, fastmath=True)
def j1s(x):
    ax = abs(x)
    # --- small |x|: J1(x) = x/2 - x^3/16 + x^5/384 - x^7/18432 + x^9/1474560 - ...
    # This is cheap and quite decent up to ~3-4; beyond that it degrades.
    if ax < 4.0:
        y = x * x
        # Horner in y for: x*(1/2 + y*(-1/16 + y*(1/384 + y*(-1/18432 + y*(1/1474560)))))
        return x * (0.5 + y * (-0.0625 + y * (0.0026041666666666665
                      + y * (-5.425347222222222e-05
                      + y * (6.781684027777778e-07)))))

    # --- large |x|: Hankel-type asymptotic with 1/x and 1/x^2 corrections
    # J1(x) ~ sqrt(2/(pi*ax)) * [ cos(ax-3pi/4)*P(1/ax^2) - sin(ax-3pi/4)*Q(1/ax) ]
    # Using first couple terms from standard asymptotic expansion.
    t = ax
    inv = 1.0 / t
    inv2 = inv * inv

    chi = t - 0.75 * np.pi  # ax - 3π/4
    c = np.cos(chi)
    s = np.sin(chi)

    # For ν=1: μ=4ν^2=4
    # P ≈ 1 - 3/(8x)^2 = 1 - 3/(64 x^2)
    # Q ≈ 3/(8x) - 15/( (8x)^3 )  (keep 1/x and 1/x^3 terms; here we include 1/x only for speed)
    P = 1.0 - (3.0 / 64.0) * inv2
    Q = (3.0 / 8.0) * inv  # (drop next term for speed)

    val = np.sqrt(2.0 / (np.pi * t)) * (c * P - s * Q)
    return -val if x < 0.0 else val  # J1 is odd

# Modified Bessel, order 0  
@njit(cache=True, fastmath=True)
def i0s(x):
    ax = abs(x)

    # I0(x) = 1 + x^2/4 + x^4/64 + x^6/2304 + x^8/147456 + ...
    if ax < 6.0:
        y = x * x
        return (1.0
                + y * (0.25
                + y * (0.015625
                + y * (0.00043402777777777775
                + y * (6.781684027777778e-06)))))

    # I0(x) ~ exp(ax)/sqrt(2*pi*ax) * (1 + 1/(8ax) + 9/(128 ax^2) + ...)
    t = ax
    inv = 1.0 / t
    inv2 = inv * inv
    poly = 1.0 + (1.0/8.0)*inv + (9.0/128.0)*inv2
    return np.exp(t) * poly / np.sqrt(2.0 * np.pi * t)

# Modified Bessel, order 1
@njit(cache=True, fastmath=True)
def i1s(x):
    ax = abs(x)

    # I1(x) = x/2 + x^3/16 + x^5/384 + x^7/18432 + x^9/1474560 + ...
    if ax < 6.0:
        y = x * x
        return x * (0.5 + y * (0.0625 + y * (0.0026041666666666665
                     + y * (5.425347222222222e-05
                     + y * (6.781684027777778e-07)))))

    # I1(x) ~ exp(ax)/sqrt(2*pi*ax) * (1 - 3/(8ax) + 15/(128 ax^2) + ...)
    t = ax
    inv = 1.0 / t
    inv2 = inv * inv
    poly = 1.0 - (3.0/8.0)*inv + (15.0/128.0)*inv2
    val = np.exp(t) * poly / np.sqrt(2.0 * np.pi * t)
    return -val if x < 0.0 else val  # I1 is odd

DEFAULT_BESSEL_N = 20

@njit(_C(_C), fastmath=True, cache=True)
def j0(z):
    # J0(z) = sum_{k>=0} (-1)^k (z^2/4)^k / (k!)^2
    zz = (z * z) * 0.25
    term = 1.0 + 0.0j
    s = term
    # ratio update: term_{k+1} = term_k * (-(z^2/4))/((k+1)^2)
    for k in range(1, DEFAULT_BESSEL_N):
        kk = float(k)
        term *= (-zz) / (kk * kk)
        s += term
        if np.abs(term) < 1e-16 * np.abs(s):
            break
    return s


@njit(_C(_C), fastmath=True, cache=True)
def j1(z):
    # J1(z) = sum_{k>=0} (-1)^k (z/2)^{2k+1} / (k!(k+1)!)
    # Start with k=0 term: z/2
    halfz = 0.5 * z
    term = halfz
    s = term
    # ratio update: term_{k+1} = term_k * (-(z^2/4))/((k+1)(k+2))
    zz = (z * z) * 0.25
    for k in range(0, DEFAULT_BESSEL_N):
        k1 = float(k + 1)
        term *= (-zz) / (k1 * (k1 + 1.0))
        s += term
        if np.abs(term) < 1e-16 * np.abs(s):
            break
    return s


@njit(_C(_C), fastmath=True, cache=True)
def i0(z):
    # I0(z) = sum_{k>=0} (z^2/4)^k / (k!)^2
    zz = (z * z) * 0.25
    term = 1.0 + 0.0j
    s = term
    for k in range(1, DEFAULT_BESSEL_N):
        kk = float(k)
        term *= (zz) / (kk * kk)
        s += term
        if np.abs(term) < 1e-16 * np.abs(s):
            break
    return s


@njit(_C(_C), fastmath=True, cache=True)
def i1(z):
    # I1(z) = sum_{k>=0} (z/2)^{2k+1} / (k!(k+1)!)
    halfz = 0.5 * z
    term = halfz
    s = term
    zz = (z * z) * 0.25
    for k in range(0, DEFAULT_BESSEL_N):
        k1 = float(k + 1)
        term *= (zz) / (k1 * (k1 + 1.0))
        s += term
        if np.abs(term) < 1e-16 * np.abs(s):
            break
    return s

DEFAULT_AIRY_N = 20

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
    kmax = DEFAULT_AIRY_N
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

@njit(types.complex128(types.int64, types.complex128), fastmath=True, cache=True)
def legendre(n, z):
    if n < 0:  return complex(np.nan, 0.0)
    if n == 0: return 1.0 + 0.0j
    if n == 1: return z
    Pnm1 = 1.0 + 0.0j
    Pn   = z
    for k in range(1, n):
        kf = float(k)
        Pnp1 = ((2.0*kf + 1.0)*z*Pn - kf*Pnm1) / (kf + 1.0)
        Pnm1 = Pn
        Pn   = Pnp1
    return Pn

@njit(types.complex128(types.complex128), fastmath=True, cache=True)
def le1(x):
    return legendre(1,x)

@njit(types.complex128(types.complex128), fastmath=True, cache=True)
def le2(x):
    return legendre(2,x)

@njit(types.complex128(types.complex128), fastmath=True, cache=True)
def le3(x):
    return legendre(3,x)

@njit(types.complex128(types.complex128), fastmath=True, cache=True)
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

@njit(types.float64(types.int64), cache=True, fastmath=True)
def _double_fact_odd(m):
    # m is expected odd and >= 1; returns (m)!! as float64
    d = 1.0
    k = 1
    while k <= m:
        d *= k
        k += 2
    return d

@njit(types.complex128(types.int64, types.complex128), cache=True, fastmath=True)
def sbessel(n, z):
    """
    Spherical Bessel j_n(z), complex z, integer n >= 0.
    """
    if n < 0:
        return np.nan + 0.0j

    az = np.abs(z)
    if az < 1e-8:
        # small-z series leading term:
        # j_n(z) ~ z^n / (2n+1)!!
        if n == 0:
            return 1.0 + 0.0j
        # compute z^n by repeated multiply (numba-safe)
        zp = 1.0 + 0.0j
        for _ in range(n):
            zp *= z
        denom = _double_fact_odd(2 * n + 1)
        return zp / denom

    if n == 0:
        return np.sin(z) / z

    if n == 1:
        return (np.sin(z) / (z * z)) - (np.cos(z) / z)

    jnm1 = np.sin(z) / z
    jn   = (np.sin(z) / (z * z)) - (np.cos(z) / z)

    for k in range(1, n):
        kf = float(k)
        jnp1 = ((2.0 * kf + 1.0) / z) * jn - jnm1
        jnm1 = jn
        jn   = jnp1

    return jn


@njit(types.complex128(types.complex128), cache=True, fastmath=True)
def sbs1(z): return sbessel(1, z)
@njit(types.complex128(types.complex128), cache=True, fastmath=True)
def sbs2(z): return sbessel(2, z)
@njit(types.complex128(types.complex128), cache=True, fastmath=True)
def sbs3(z): return sbessel(3, z)
@njit(types.complex128(types.complex128), cache=True, fastmath=True)
def sbs4(z): return sbessel(4, z)

# ============================================================
# 2. Chebyshev polynomials T_n(x), U_n(x)
# ============================================================

@njit(types.complex128(types.int64, types.complex128), fastmath=True, cache=True)
def chebt(n, z):
    """
    Chebyshev T_n(z) (first kind), complex z.

      T_0 = 1
      T_1 = z
      T_{k+1} = 2 z T_k - T_{k-1}
    """
    if n < 0:
        return complex(float('nan'), 0.0)

    if n == 0:
        return 1.0 + 0.0j
    if n == 1:
        return z

    Tnm1 = 1.0 + 0.0j
    Tn   = z
    for _ in range(1, n):
        Tnp1 = 2.0 * z * Tn - Tnm1
        Tnm1 = Tn
        Tn   = Tnp1
    return Tn

@njit(types.complex128(types.complex128), fastmath=True, cache=True)
def cht1(z): return chebt(1, z)
@njit(types.complex128(types.complex128), fastmath=True, cache=True)
def cht2(z): return chebt(2, z)
@njit(types.complex128(types.complex128), fastmath=True, cache=True)
def cht3(z): return chebt(3, z)
@njit(types.complex128(types.complex128), fastmath=True, cache=True)
def cht4(z): return chebt(4, z)


@njit(types.complex128(types.int64, types.complex128), fastmath=True, cache=True)
def chebu(n, z):
    """
    Chebyshev U_n(z) (second kind), complex z.

      U_0 = 1
      U_1 = 2 z
      U_{k+1} = 2 z U_k - U_{k-1}
    """
    if n < 0:
        return complex(float('nan'), 0.0)

    if n == 0:
        return 1.0 + 0.0j
    if n == 1:
        return 2.0 * z

    Unm1 = 1.0 + 0.0j
    Un   = 2.0 * z
    for _ in range(1, n):
        Unp1 = 2.0 * z * Un - Unm1
        Unm1 = Un
        Un   = Unp1
    return Un

@njit(types.complex128(types.complex128), fastmath=True, cache=True)
def chu1(z): return chebu(1, z)
@njit(types.complex128(types.complex128), fastmath=True, cache=True)
def chu2(z): return chebu(2, z)
@njit(types.complex128(types.complex128), fastmath=True, cache=True)
def chu3(z): return chebu(3, z)
@njit(types.complex128(types.complex128), fastmath=True, cache=True)
def chu4(z): return chebu(4, z)


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

@njit(types.complex128(types.int64, types.complex128), fastmath=True, cache=True)
def hermite(n, z):
    """
    Physicists' Hermite polynomial H_n(z), complex z.

        H_0(z) = 1
        H_1(z) = 2 z
        H_{n+1}(z) = 2 z H_n(z) - 2 n H_{n-1}(z)

    Valid for all complex z.
    """
    if n < 0:
        return np.nan + 0.0j

    if n == 0:
        return 1.0 + 0.0j

    if n == 1:
        return 2.0 * z

    Hnm1 = 1.0 + 0.0j   # H_0
    Hn   = 2.0 * z      # H_1

    for k in range(1, n):
        Hnp1 = 2.0 * z * Hn - 2.0 * float(k) * Hnm1
        Hnm1 = Hn
        Hn   = Hnp1

    return Hn

@njit(types.complex128(types.complex128), fastmath=True, cache=True)
def he1(x):
    return hermite(1,x)

@njit(types.complex128(types.complex128), fastmath=True, cache=True)
def he2(x):
    return hermite(2,x)

@njit(types.complex128(types.complex128), fastmath=True, cache=True)
def he3(x):
    return hermite(3,x)

@njit(types.complex128(types.complex128), fastmath=True, cache=True)
def he4(x):
    return hermite(4,x)

# ============================================================
# Laguerre polynomials L_n(x) (α = 0)
# ============================================================

@njit(types.complex128(types.int64, types.complex128), fastmath=True, cache=True)
def laguerre(n, z):
    """
    Laguerre polynomial L_n(z) with alpha=0, complex z.

      L_0(z) = 1
      L_1(z) = 1 - z
      (k+1) L_{k+1}(z) = (2k + 1 - z) L_k(z) - k L_{k-1}(z)
    """
    if n < 0:
        return complex(float('nan'), 0.0)

    if n == 0:
        return 1.0 + 0.0j

    if n == 1:
        return 1.0 - z

    Lnm1 = 1.0 + 0.0j   # L_0
    Ln   = 1.0 - z      # L_1

    for k in range(1, n):
        kf = float(k)
        Lnp1 = ((2.0 * kf + 1.0 - z) * Ln - kf * Lnm1) / (kf + 1.0)
        Lnm1 = Ln
        Ln   = Lnp1

    return Ln

@njit(types.complex128(types.complex128), fastmath=True, cache=True)
def la1(z): return laguerre(1, z)
@njit(types.complex128(types.complex128), fastmath=True, cache=True)
def la2(z): return laguerre(2, z)
@njit(types.complex128(types.complex128), fastmath=True, cache=True)
def la3(z): return laguerre(3, z)
@njit(types.complex128(types.complex128), fastmath=True, cache=True)
def la4(z): return laguerre(4, z)

@njit(types.complex128(
    types.int64,
    types.complex128,
    types.complex128,
    types.complex128
), fastmath=True, cache=True)
def jacobi(n, alpha, beta, z):
    """
    Jacobi polynomial P_n^{(alpha, beta)}(z), complex parameters.

    Valid analytic continuation in z for fixed alpha, beta.
    Recurrence (Szegő / Abramowitz–Stegun).
    """
    if n < 0:
        return np.nan + 0.0j

    if n == 0:
        return 1.0 + 0.0j

    if n == 1:
        return 0.5 * ((2.0 + alpha + beta) * z + (alpha - beta))

    Pnm1 = 1.0 + 0.0j
    Pn   = 0.5 * ((2.0 + alpha + beta) * z + (alpha - beta))

    for k in range(1, n):
        kf = float(k)
        two_k_ab = 2.0 * kf + alpha + beta

        A = 2.0 * (kf + 1.0) * (kf + alpha + beta + 1.0) * two_k_ab
        B = (two_k_ab + 1.0) * (
              (two_k_ab + 2.0) * two_k_ab * z
              + (alpha * alpha - beta * beta)
            )
        C = 2.0 * (kf + alpha) * (kf + beta) * (two_k_ab + 2.0)

        Pnp1 = (B * Pn - C * Pnm1) / A

        Pnm1 = Pn
        Pn   = Pnp1

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

@njit(types.complex128(types.complex128), fastmath=True, cache=True)
def sinc(z):
    """
    sinc(z) = sin(z)/z with analytic continuation at z=0.
    """
    if np.abs(z) < 1e-8:
        # sin(z)/z ≈ 1 - z^2/6
        return 1.0 - (z * z) / 6.0
    return np.sin(z) / z


@njit(types.complex128(types.complex128), fastmath=True, cache=True)
def sinhc(z):
    """
    sinhc(z) = sinh(z)/z with analytic continuation at z=0.
    """
    if np.abs(z) < 1e-8:
        # sinh(z)/z ≈ 1 + z^2/6
        return 1.0 + (z * z) / 6.0
    return np.sinh(z) / z


@njit(types.complex128(types.complex128), fastmath=True, cache=True)
def sech(z):
    return 1.0 / np.cosh(z)


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
    "j1s": j0s,
    "i0s": i0s,
    "i1s": i1s,
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

