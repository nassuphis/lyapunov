#!/usr/bin/env python
"""
lyapounov.py

Lyapunov fractal renderer for 1‑D maps with A/B forcing, driven by
specparser/expandspec.

Key idea: every map is defined by a *single* symbolic expression

    f(x, r; alpha, beta, delta, epsilon)

where:
    - x is the state
    - r is the driven parameter (A/B alternating via a sequence)
    - alpha, beta, delta, epsilon are optional extra parameters

For each expression we automatically:
    - build a Numba‑compatible stepping function
    - build its symbolic derivative df/dx using SymPy
    - JIT both with the same (x, r, params) signature

The Lyapunov code then treats all maps generically; adding a new map is
just:
    1) add an entry in MAP_TEMPLATES with an expression string
    2) (optionally) set default (A,B) window and parameter defaults
"""

import sys
from pathlib import Path
parent = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent))
import os
from pathlib import Path
import time
import math
import cmath
import argparse
import re as regex
import numpy as np
import sympy as sp
from numba import njit, types, prange

from specparser import specparser, expandspec
from rasterizer import raster
from rasterizer import colors

import functions
import maps

# ---------------------------------------------------------------------------
# Global defaults
# ---------------------------------------------------------------------------

DEFAULT_MAP_NAME = "logistic"
DEFAULT_SEQ      = "AB"
DEFAULT_TRANS    = 200
DEFAULT_ITER     = 1000
DEFAULT_X0       = 0.5
DEFAULT_EPS_LYAP = 1e-12
DEFAULT_CLIP     = None     # auto from data
DEFAULT_GAMMA    = 1.0      # colormap gamma


# ---------------------------------------------------------------------------
# Symbolic derivative helper (x derivative of map expression)
# ---------------------------------------------------------------------------

def _sympy_deriv(expr_str: str) -> str:
    """
    Return d/dx of the given expression string in SymPy's sstr format.

    The expression can use:
        x, r,  x, r, a, b, c, d, epsilon, eps, zeta, eta
        sin, cos, tan, sec, cosh, exp, sign, abs/Abs, max
        step (Heaviside), DiracDelta, pow, mod1, Mod, pi
    """
    x, r = sp.symbols("x r")
    a, b, c, d = sp.symbols("a b c d")
    eps, zeta, eta, epsilon = sp.symbols("eps zeta eta epsilon")

    locs = {
        "x": x,
        "r": r,
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "epsilon": epsilon,
        "eps": eps,
        "zeta": zeta,
        "eta": eta,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "sec": sp.sec,
        "cosh": sp.cosh,
        "exp": sp.exp,
        "sign": sp.sign,
        "abs": sp.Abs,
        "Abs": sp.Abs,
        "max": sp.Max,
        "min": sp.Min,
        "step": sp.Heaviside,
        "Heaviside": sp.Heaviside,
        "DiracDelta": sp.DiracDelta,
        "pow": sp.Pow,
        "apow": lambda x, a: sp.sign(x) * sp.Pow(sp.Abs(x),a),
        "mod1": lambda v: sp.Mod(v, 1),
        "Mod": sp.Mod,
        "pi": sp.pi,
        "floor": sp.floor,
        "ceil": sp.ceiling,
    }

    expr = sp.sympify(expr_str, locals=locs)
    expr_der = sp.diff(expr, x)
    return sp.sstr(expr_der)


def _sympy_jacobian_2d(expr_x: str, expr_y: str):
    """
    Return 4 SymPy sstr expressions for the Jacobian of a 2‑D map:
        expr_x = f(x, y, r, s, ...)
        expr_y = g(x, y, r, s, ...)
    """
    x, y, r, s = sp.symbols("x y r s")
    a, b, c, d = sp.symbols("a b c d")
    eps, zeta, eta, epsilon = sp.symbols("eps zeta eta epsilon")

    locs = {
        "x": x,
        "y": y,
        "r": r,
        "s": s,
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "epsilon": epsilon,
        "eps": eps,
        "zeta": zeta,
        "eta": eta,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "sec": sp.sec,
        "cosh": sp.cosh,
        "exp": sp.exp,
        "sign": sp.sign,
        "abs": sp.Abs,
        "Abs": sp.Abs,
        "max": sp.Max,
        "min": sp.Min,
        "step": sp.Heaviside,
        "Heaviside": sp.Heaviside,
        "DiracDelta": sp.DiracDelta,
        "pow": sp.Pow,
        "apow": lambda x, a: sp.sign(x) * sp.Pow(sp.Abs(x), a),
        "mod1": lambda v: sp.Mod(v, 1),
        "Mod": sp.Mod,
        "pi": sp.pi,
        "floor": sp.floor,
        "ceil": sp.ceiling,
    }

    fx = sp.sympify(expr_x, locals=locs)
    fy = sp.sympify(expr_y, locals=locs)

    dfx_dx = sp.diff(fx, x)
    dfx_dy = sp.diff(fx, y)
    dfy_dx = sp.diff(fy, x)
    dfy_dy = sp.diff(fy, y)

    return tuple(sp.sstr(e) for e in (dfx_dx, dfx_dy, dfy_dx, dfy_dy))


# ---------------------------------------------------------------------------
# allowed functions
# ---------------------------------------------------------------------------


NS = {
    "i": functions.i,
    "step": functions.step,
    "Heaviside": functions.Heaviside,
    "DiracDelta": functions.DiracDelta,
    "sign": functions.sign,
    "Abs": functions.Abs,
    "abs": functions.Abs,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "sec": functions.sec,
    "cosh": np.cosh,
    "sinh": np.sinh,
    "exp": np.exp,
    "pow": np.power,
    "apow": functions.apow,
    "log": np.log,
    "mod1": functions.mod1,
    "Mod": functions.Mod,
    "Derivative": functions.Derivative,
    "re": functions.re,
    "im": functions.im,
    "pi": np.pi,
    "max": max,
    "min": min,
    "floor": functions.floor,
    "ceil": functions.ceil,
    "round": round,
    "to01": functions.to01,
    "wavg": functions.wavg,
    "cwavg": functions.cwavg,
    "f2i": functions.f2i,
    "abs_cap": functions.abs_cap,
    "norm": functions.norm,
    "j0s": functions.j0s,
    "j0": functions.j0,
    "j1": functions.j1,
    "i0": functions.i0,
    "i1": functions.i1,
    "lgamma": math.lgamma,
    "aira": functions.airy_ai,
    "airb": functions.airy_bi,
    "frec": functions.fresnel_c,
    "fres": functions.fresnel_s,
    "zeta": functions.zeta,
    "lambertw": functions.lambertw,
    "gammainc": functions.gammainc,
    "dawson": functions.dawson,
    "ei": functions.ei,
    "si": functions.si,
    "ci": functions.ci,
    "gd": functions.gd,
    "clausen": functions.clausen,
    "li2": functions.li2,
    "lint": functions.lint,
    "legendre": functions.legendre,
    "le1": functions.le1,
    "le2": functions.le2,
    "le3": functions.le3,
    "le4": functions.le4,
    "hermite": functions.hermite,
    "he1": functions.he1,
    "he2": functions.he2,
    "he3": functions.he3,
    "he4": functions.he4,
    "laguerre": functions.laguerre,
    "la1": functions.la1,
    "la2": functions.la2,
    "la3": functions.la3,
    "la4": functions.la4,
    "sbessel": functions.sbessel,
    "chebt": functions.chebt,
    "cht1": functions.cht1,
    "cht2": functions.cht2,
    "cht3": functions.cht3,
    "cht4": functions.cht4,
    "chebu": functions.chebu,
    "chu1": functions.chu1,
    "chu2": functions.chu2,
    "chu3": functions.chu3,
    "chu4": functions.chu4,
    "jacobi": functions.jacobi,
    "meixpol": functions.meixpol,
    "hahn": functions.hahn,
    "sech": functions.sech,
    "sinc": functions.sinc,
    "sinhc": functions.sinhc,
    "np": np,
    "math": math,
    "cmath": cmath,
}

# ---------------------------------------------------------------------------
# build python function text
# ---------------------------------------------------------------------------

def _funtext_1d(name:str ,expr:str, dict) -> str:
    lines = [
        f"def {name}(x, forced):",
    ]
    for i, (key, value) in enumerate(dict.items()):
        if not isinstance(key, str):
            raise TypeError(f"Only str keys supported, got {type(key)!r}")
        if not key.isidentifier():
            raise ValueError(f"Key {key!r} is not a valid Python identifier")
        lines.extend([
        f"    {key} = {value}"
        ])
    lines.extend([
        f"    x_next = {expr}",
        f"    return x_next"
    ])
    source = "\n".join(lines)
    return source

def _funtext_2d_ab_step(name: str, expr_x: str, expr_y: str, dict) -> str:
    lines = [
        f"def {name}(x, y, forced):",
    ]
    for i, (key, value) in enumerate(dict.items()):
        if not isinstance(key, str):
            raise TypeError(f"Only str keys supported, got {type(key)!r}")
        if not key.isidentifier():
            raise ValueError(f"Key {key!r} is not a valid Python identifier")
        lines.extend([
        f"    {key} = {value}"
        ])
    lines.extend([
        f"    x_next = {expr_x}",
        f"    y_next = {expr_y}",
        f"    return x_next, y_next",
    ])
    source = "\n".join(lines)
    return source

def _funtext_2d_ab_jac(
     name: str, dXdx: str, dXdy: str, dYdx: str, dYdy: str, dict
) -> str:
    lines = [
        f"def {name}(x, y, forced):",
    ]
    for i, (key, value) in enumerate(dict.items()):
        if not isinstance(key, str):
            raise TypeError(f"Only str keys supported, got {type(key)!r}")
        if not key.isidentifier():
            raise ValueError(f"Key {key!r} is not a valid Python identifier")
        lines.extend([
        f"    {key} = {value}"
        ])
    lines.extend([
        f"    dxdx = {dXdx}",
        f"    dxdy = {dXdy}",
        f"    dydx = {dYdx}",
        f"    dydy = {dYdy}",
        f"    return dxdx, dxdy, dydx, dydy",
    ])
    source = "\n".join(lines)
    return source

def _funtext_2d_step(name: str, expr_x: str, expr_y: str, dict) -> str:
    lines = [
        f"def {name}(x, y, first, second):",
    ]
    for i, (key, value) in enumerate(dict.items()):
        if not isinstance(key, str):
            raise TypeError(f"Only str keys supported, got {type(key)!r}")
        if not key.isidentifier():
            raise ValueError(f"Key {key!r} is not a valid Python identifier")
        lines.extend([
        f"    {key} = {value}"
        ])
    lines.extend([
        f"    x_next = {expr_x}",
        f"    y_next = {expr_y}",
        f"    return x_next, y_next",
    ])
    source = "\n".join(lines)
    return source

def _funtext_2d_jac(
     name: str, dXdx: str, dXdy: str, dYdx: str, dYdy: str, dict
) -> str:
    lines = [
        f"def {name}(x, y, first, second):",
    ]
    for i, (key, value) in enumerate(dict.items()):
        if not isinstance(key, str):
            raise TypeError(f"Only str keys supported, got {type(key)!r}")
        if not key.isidentifier():
            raise ValueError(f"Key {key!r} is not a valid Python identifier")
        lines.extend([
        f"    {key} = {value}"
        ])
    lines.extend([
        f"    dxdx = {dXdx}",
        f"    dxdy = {dXdy}",
        f"    dydx = {dYdx}",
        f"    dydy = {dYdy}",
        f"    return dxdx, dxdy, dydx, dydy",
    ])
    source = "\n".join(lines)
    return source

# ---------------------------------------------------------------------------
# build python functions from text
# ---------------------------------------------------------------------------



# 1D forced step + deriv
def _funpy_1d(expr: str, dict):
    ns = NS.copy()
    src = _funtext_1d("impl",expr, dict)
    exec(src, ns, ns)
    return ns["impl"]

# 2D forced step 
def _funpy_2d_ab_step(expr_x: str, expr_y: str, dict):
    ns = NS.copy()
    src = _funtext_2d_ab_step("impl2_step", expr_x, expr_y, dict)
    exec(src, ns, ns)
    return ns["impl2_step"]


# 2D forced jacobian
def _funpy_2d_ab_jac(dXdx, dXdy, dYdx, dYdy,dict):
    ns = NS.copy()
    src = _funtext_2d_ab_jac("impl2_jac", dXdx, dXdy, dYdx, dYdy, dict)
    exec(src, ns, ns)
    return ns["impl2_jac"]

# 2D 
def _funpy_2d_step(expr_x: str, expr_y: str, dict):
    ns = NS.copy()
    src = _funtext_2d_step("impl2_step", expr_x, expr_y, dict)
    exec(src, ns, ns)
    return ns["impl2_step"]

# 2D jacobian
def _funpy_2d_jac(dXdx, dXdy, dYdx, dYdy,dict):
    ns = NS.copy()
    src = _funtext_2d_jac("impl2_jac", dXdx, dXdy, dYdx, dYdy, dict)
    exec(src, ns, ns)
    return ns["impl2_jac"]

# ---------------------------------------------------------------------------
# jit function
# ---------------------------------------------------------------------------

# All jitted step/deriv functions will share these signatures
#

STEP_SIG = types.float64(
    types.float64,   # x, the mapped variable
    types.float64,   # forced
)

STEP2_AB_SIG = types.UniTuple(types.float64, 2)(
    types.float64,  # x, mapped variable
    types.float64,  # y, mapped variable
    types.float64,  # forced
)

JAC2_AB_SIG = types.UniTuple(types.float64, 4)(
    types.float64,  # x, mapped variable
    types.float64,  # y, mapped variable
    types.float64,  # forced
)

STEP2_SIG = types.UniTuple(types.float64, 2)(
    types.float64,  # x, mapped variable
    types.float64,  # y, mapped variable
    types.float64,  # first
    types.float64,  # second
)

JAC2_SIG = types.UniTuple(types.float64, 4)(
    types.float64,  # x, mapped variable
    types.float64,  # y, mapped variable
    types.float64,  # first
    types.float64,  # second
)


def _funjit_1d(expr:str, dict):
    fun = _funpy_1d(expr,dict)
    jit = njit(STEP_SIG, cache=False, fastmath=False)(fun)
    return jit

def _funjit_2d_ab_step(xexpr:str, yexpr:str, dict):
    fun = _funpy_2d_ab_step(xexpr, yexpr, dict)
    jit = njit(STEP2_AB_SIG, cache=False, fastmath=False)(fun)
    return jit

def _funjit_2d_ab_jag(dxdx:str, dxdy:str, dydx:str, dydy:str, dict):
    fun = _funpy_2d_ab_jac( dxdx, dxdy, dydx, dydy, dict )
    jit = njit(JAC2_AB_SIG, cache=False, fastmath=False)(fun)
    return jit

def _funjit_2d_step(xexpr:str, yexpr:str, dict):
    fun = _funpy_2d_step(xexpr, yexpr, dict)
    jit = njit(STEP2_SIG, cache=False, fastmath=False)(fun)
    return jit

def _funjit_2d_jag(dxdx:str, dxdy:str, dydx:str, dydy:str, dict):
    fun = _funpy_2d_jac( dxdx, dxdy, dydx, dydy, dict )
    jit = njit(JAC2_SIG, cache=False, fastmath=False)(fun)
    return jit


# Lazy map cache: compiled maps live here, built on first use.
MAPS: dict[str, dict] = {}

def substitute_common(x,d):
    if d is None:
        return x 
    x=x.format(**d)
    return x


def _build_map(name: str) -> dict:
    """
    Build (compile) a single map configuration from MAP_TEMPLATES[name].
    This does the same work the old _build_maps() did, but for one name.
    """
    if name not in maps.MAP_TEMPLATES:
        raise KeyError(f"Unknown map '{name}'")

    cfg = maps.MAP_TEMPLATES[name]
    new_cfg = dict(cfg)
    type = cfg.get("type", "step1d")
    pardict = cfg.get("pardict",dict())
    new_cfg["pardict"] = pardict
    new_cfg["domain"]  = np.asarray(cfg.get("domain", [0.0, 0.0, 1.0, 1.0]),dtype=np.float64)
    new_cfg["type"] = type

    if type == "step1d":
        expr = substitute_common(cfg["expr"],cfg.get("expr_common"))
        new_cfg["step"]  =  _funjit_1d(expr,pardict)
        if "deriv_expr" in cfg:
            deriv_expr = substitute_common(cfg["deriv_expr"],cfg.get("expr_common"))
        else:
            deriv_expr =  _sympy_deriv(substitute_common(cfg.get("expr"),cfg.get("expr_common"))) 
        new_cfg["deriv"] =  _funjit_1d(deriv_expr,pardict)
        new_cfg["eps_floor"] = cfg.get("eps_floor", 1e-16)
        return new_cfg
    
    if type == "step2d":
        if "step2_func" in cfg and "jac2_func" in cfg:
            new_cfg["step2"] = njit(STEP2_SIG, cache=False, fastmath=False)(cfg["step2_func"])
            new_cfg["jac2"]  = njit(JAC2_SIG, cache=False, fastmath=False)(cfg["jac2_func"])
        else:
            expr_x = substitute_common( cfg["expr_x"], cfg.get("expr_common") )
            expr_y = substitute_common( cfg["expr_y"], cfg.get("expr_common") )
            new_cfg["step2"] = _funjit_2d_step(expr_x,expr_y,pardict)
            if "jac_exprs" in cfg:
                dXdx, dXdy, dYdx, dYdy = cfg["jac_exprs"]
                dXdx = substitute_common( dXdx, cfg.get("expr_common") )
                dXdy = substitute_common( dXdy, cfg.get("expr_common") )
                dYdx = substitute_common( dYdx, cfg.get("expr_common") )
                dYdy = substitute_common( dYdy, cfg.get("expr_common") )
            else:
                dXdx, dXdy, dYdx, dYdy = _sympy_jacobian_2d(expr_x, expr_y)
            new_cfg["jac2"] = _funjit_2d_jag(dXdx,dXdy,dYdx,dYdy,pardict)
        new_cfg["eps_floor"] = cfg.get("eps_floor", 1e-16)
        return new_cfg
    
    if type == "step2d_ab":
        if "step2_func" in cfg and "jac2_func" in cfg:
            new_cfg["step2_ab"] = njit(STEP2_AB_SIG, cache=False, fastmath=False)(cfg["step2_func"])
            new_cfg["jac2_ab"]  = njit(JAC2_AB_SIG, cache=False, fastmath=False)(cfg["jac2_func"])
        else:
            expr_x = substitute_common( cfg["expr_x"], cfg.get("expr_common") )
            expr_y = substitute_common( cfg["expr_y"], cfg.get("expr_common") )
            new_cfg["step2_ab"] = _funjit_2d_ab_step(expr_x,expr_y,pardict)
            if "jac_exprs" in cfg:
                dXdx, dXdy, dYdx, dYdy = cfg["jac_exprs"]
                dXdx = substitute_common( dXdx, cfg.get("expr_common") )
                dXdy = substitute_common( dXdy, cfg.get("expr_common") )
                dYdx = substitute_common( dYdx, cfg.get("expr_common") )
                dYdy = substitute_common( dYdy, cfg.get("expr_common") )
            else:
                dXdx, dXdy, dYdx, dYdy = _sympy_jacobian_2d(expr_x, expr_y)
            new_cfg["jac2_ab"] = _funjit_2d_ab_jag(dXdx,dXdy,dYdx,dYdy,pardict)
        new_cfg["eps_floor"] = cfg.get("eps_floor", 1e-16)
        return new_cfg
    
   
    raise ValueError(f"Unsupported type={type} for map '{name}'")


# ---------------------------------------------------------------------------
# Sequence handling (A/B patterns)
# ---------------------------------------------------------------------------

SEQ_ALLOWED_RE = regex.compile(r"^[AaBb0-9()]+$")


def _looks_like_sequence_token(tok: str) -> bool:
    s = tok.strip()
    if not s:
        return False
    if not SEQ_ALLOWED_RE.match(s):
        return False
    # must contain at least one A/B or '(' so "123" isn't treated as seq
    return any(ch in "AaBb(" for ch in s)


def _decode_sequence_token(tok: str, default_seq: str = DEFAULT_SEQ) -> str:
    """
    Decode a sequence token into a string of 'A' and 'B'.

    Supported syntax:

        ABBA        -> ABBA
        A5B5        -> AAAAA BBBBB
        AB3A2       -> A B B B A A
        (AB)40      -> AB repeated 40 times
        A2(BA)3B    -> AA BABABA B
    """
    s = tok.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        s = s[1:-1]
    if not s:
        return default_seq

    out_parts = []
    i = 0
    n = len(s)

    while i < n:
        ch = s[i]

        # Single letter A/B with optional count
        if ch in "AaBb":
            letter = ch.upper()
            i += 1
            j = i
            while j < n and s[j].isdigit():
                j += 1
            if j == i:
                count = 1
            else:
                try:
                    count = int(s[i:j])
                except Exception:
                    return default_seq
                if count < 0:
                    return default_seq
            out_parts.append(letter * count)
            i = j
            continue

        # Parenthesised group (AB...) with optional count: (AB)40
        if ch == "(":
            j = s.find(")", i + 1)
            if j == -1:
                return default_seq
            group_str = s[i + 1 : j]
            if not group_str:
                return default_seq
            if any(c not in "AaBb" for c in group_str):
                return default_seq
            group = "".join(c.upper() for c in group_str)

            k = j + 1
            while k < n and s[k].isdigit():
                k += 1
            if k == j + 1:
                count = 1
            else:
                try:
                    count = int(s[j + 1 : k])
                except Exception:
                    return default_seq
                if count < 0:
                    return default_seq

            out_parts.append(group * count)
            i = k
            continue

        # Anything else -> fall back to default
        return default_seq

    seq = "".join(out_parts)
    return seq or default_seq


def _seq_to_array(seq_str: str) -> np.ndarray:
    s = (seq_str or "").strip().upper()
    if not s:
        raise ValueError("Sequence must be non-empty (e.g. 'AB')")
    data = []
    for ch in s:
        if ch == "A":
            data.append(0)
        elif ch == "B":
            data.append(1)
        else:
            raise ValueError(f"Invalid symbol '{ch}' in sequence; use only A/B.")
    return np.asarray(data, dtype=np.int32)


# ---------------------------------------------------------------------------
# Lyapunov field 
# ---------------------------------------------------------------------------

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
def _lyapunov_field_1d(
    step,
    deriv,
    seq,        
    domain,     # <- 1D float64 array: [llx, lly, ulx, uly, lrx, lry]
    pix,
    x0,
    n_transient,
    n_iter,
    eps,
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
                x = step(x, forced_param)

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
def _lyapunov_field_2d_ab(
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

                x_next, y_next = step2_ab(x, y, forced_param)

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
def _lyapunov_field_2d(
    step2,
    jac2,
    domain,        # [llx, lly, ulx, uly, lrx, lry] in (r,s)-plane
    pix,
    x0,
    y0,
    n_transient,
    n_iter,
    eps_floor,
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
                x_next, y_next = step2(x, y, first_param,second_param)
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
def _entropy_from_amplitudes(A):
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
def _entropy_field_1d(
    step,
    seq,
    domain,     # [llx, lly, ulx, uly, lrx, lry] in (A,B)
    pix,
    x0,
    n_transient,
    n_iter,
    omegas,     # 1D float64 array of frequencies
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
                x = step(x, forced_param)
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
                x = step(x, forced_param)
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

            out[j, i] = _entropy_from_amplitudes(Avals)

    return out


@njit(cache=False, fastmath=False, parallel=True)
def _entropy_field_2d_ab(
    step2_ab,
    seq,
    domain,        # [llx, lly, ulx, uly, lrx, lry] in (A,B)
    pix,
    x0,
    y0,
    n_transient,
    n_iter,
    omegas,
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
                x, y = step2_ab(x, y, forced_param)
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
                x, y = step2_ab(x, y, forced_param)
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

            out[j, i] = _entropy_from_amplitudes(Avals)

    return out


@njit(cache=False, fastmath=False, parallel=True)
def _entropy_field_2d(
    step2,
    domain,        # [llx, lly, ulx, uly, lrx, lry] in (first,second)
    pix,
    x0,
    y0,
    n_transient,
    n_iter,
    omegas,
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
                x, y = step2(x, y, first_param, second_param)
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
                x, y = step2(x, y, first_param, second_param)
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

            out[j, i] = _entropy_from_amplitudes(Avals)

    return out

# ---------------------------------------------------------------------------
# stat field 
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
def compute_orbit(step, x0, A, B, seq, n_transient, n_iter, xs):
    seq_len = seq.size
    x = x0
    for n in range(n_transient):
        force = seq[n % seq_len] & 1
        forced_param = A if force == 0 else B
        x = step(x, forced_param)
        if not math.isfinite(x): x = 0.5
    for n in range(n_iter):
        force = seq[n % seq_len] & 1
        forced_param = A if force == 0 else B
        x = step(x, forced_param)
        if not math.isfinite(x): x = 0.5
        xs[n] = x
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
def transform_hist(hcalc,hist):
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
    return 0.0



@njit(cache=False, fastmath=False, parallel=True)
def _hist_field_1d(
    step,
    seq,
    domain,     # [llx, lly, ulx, uly, lrx, lry] in (A,B)
    pix,
    x0,
    n_transient,
    n_iter,
    vcalc=0,
    hcalc=0,
    hbins=32,
):

    out = np.empty((pix, pix), dtype=np.float64)
    denom = 1.0 if pix <= 1 else (pix - 1.0)

    for j in prange(pix):
        xs = np.empty(n_iter, dtype=np.float64)
        vs = np.empty(n_iter, dtype=np.float64)
        hist = np.zeros(hbins, dtype=np.int64)
        for i in range(pix):
            A, B = map_logical_to_physical(domain, i / denom, j / denom)
            compute_orbit(step, x0, A, B, seq, n_transient, n_iter, xs)
            transform_values(vcalc, xs, vs)
            vmin = 1e300
            vmax = -1e300
            for n in range(n_iter):
                v = vs[n]
                if v < vmin: vmin = v
                if v > vmax: vmax = v
            for k in range(hist.size): hist[k] = 0  # reset
            hist_fixed_bins_inplace(hist, vs, vmin, vmax)
            e=transform_hist(hcalc,hist)
            out[j, i] = -e

    return out

@njit(cache=False, fastmath=False, parallel=True)
def _hist_field_2d_ab(
    step2_ab,
    seq,
    domain,     # [llx, lly, ulx, uly, lrx, lry] in (A,B)
    pix,
    x0,
    y0,
    n_transient,
    n_iter,
    vcalc=0,
    hcalc=0
):
    """
    Histogram-based texture field for a 2-D AB-forced map.
    Observable is derived from x via vcalc:
        0: value      -> v = x
        1: slope      -> v = x - px
        2: convexity  -> v = x - 2*px + ppx
        3: curvature  -> v = |x-2px+ppx| / (1 + (x-px)^2)^(3/2)
    hcalc selects the histogram functional (same as 1-D version):
        0: std(bins)
        1: entropy(bins)
        2: "zero-crossings" of bins about their mean
        3: std(diff(bins))
    """
    seq_len = seq.size
    out = np.empty((pix, pix), dtype=np.float64)
    denom = 1.0 if pix <= 1 else (pix - 1.0)

    for j in prange(pix):
        vs   = np.zeros(n_iter, dtype=np.float64)
        bins = np.empty(n_iter, dtype=np.int64)

        for i in range(pix):
            A, B = map_logical_to_physical(domain, i / denom, j / denom)

            # burn-in
            x = x0
            y = y0
            px = x0
            ppx = x0
            for n in range(n_transient):
                force = seq[n % seq_len] & 1
                forced_param = A if force == 0 else B
                x, y = step2_ab(x, y, forced_param)
                if (not np.isfinite(x)) or (not np.isfinite(y)):
                    x = 0.5
                    y = 0.5
                ppx = px
                px  = x

            vmin = 1e6
            vmax = -1e6

            for n in range(n_iter):
                force = seq[n % seq_len] & 1
                forced_param = A if force == 0 else B
                x, y = step2_ab(x, y, forced_param)
                if (not np.isfinite(x)) or (not np.isfinite(y)):
                    x = 0.5
                    y = 0.5

                if vcalc == 0:        # value
                    v = x
                elif vcalc == 1:      # slope
                    v = x - px
                elif vcalc == 2:      # convexity
                    v = x - 2.0*px + ppx
                elif vcalc == 3:      # curvature
                    num = math.fabs(x - 2.0*px + ppx)
                    den = math.pow(1.0 + (x - px)*(x - px), 1.5)
                    if den > 0.0:
                        v = num / den
                    else:
                        v = 0.0
                else:
                    v = x

                if v < vmin:
                    vmin = v
                if v > vmax:
                    vmax = v

                vs[n] = v
                ppx = px
                px  = x

            hist_fixed_bins_inplace(bins, vs, vmin, vmax)

            e = 0.0
            if hcalc == 0:      # stdev of bin counts
                e = np.std(bins)
            elif hcalc == 1:    # entropy
                total = float(np.sum(bins))
                if total > 0.0:
                    H = 0.0
                    for b in bins:
                        if b > 0:
                            p = b / total
                            H += p * math.log(p)
                    e = H / math.log(bins.size)
            elif hcalc == 2:    # "zero-crossing" of bins around mean
                m = float(np.mean(bins))
                s = np.sign(bins - m)
                c = s[1:] * s[:-1]
                e = np.sum(c > 0) / bins.size
            elif hcalc == 3:    # std of changes
                for k in range(bins.size - 1):
                    bins[k] = bins[k + 1] - bins[k]
                e = np.std(bins[:-1])
            else:
                e = 0.0

            out[j, i] = -e

    return out

@njit(cache=False, fastmath=False, parallel=True)
def _hist_field_2d(
    step2,
    domain,     # [llx, lly, ulx, uly, lrx, lry] in (first,second)
    pix,
    x0,
    y0,
    n_transient,
    n_iter,
    vcalc=0,
    hcalc=0
):
    """
    Histogram-based texture field for a 2-D non-forced map.
    Parameters are (first, second) from the domain; observable derived from x.
    vcalc and hcalc as in _hist_field_2d_ab.
    """
    out = np.empty((pix, pix), dtype=np.float64)
    denom = 1.0 if pix <= 1 else (pix - 1.0)

    for j in prange(pix):
        vs   = np.zeros(n_iter, dtype=np.float64)
        bins = np.empty(n_iter, dtype=np.int64)

        for i in range(pix):
            first_param, second_param = map_logical_to_physical(
                domain, i / denom, j / denom
            )

            x = x0
            y = y0
            px = x0
            ppx = x0

            # burn-in
            for n in range(n_transient):
                x, y = step2(x, y, first_param, second_param)
                if (not np.isfinite(x)) or (not np.isfinite(y)):
                    x = 0.5
                    y = 0.0
                ppx = px
                px  = x

            vmin = 1e6
            vmax = -1e6

            for n in range(n_iter):
                x, y = step2(x, y, first_param, second_param)
                if (not np.isfinite(x)) or (not np.isfinite(y)):
                    x = 0.5
                    y = 0.0

                if vcalc == 0:        # value
                    v = x
                elif vcalc == 1:      # slope
                    v = x - px
                elif vcalc == 2:      # convexity
                    v = x - 2.0*px + ppx
                elif vcalc == 3:      # curvature
                    num = math.fabs(x - 2.0*px + ppx)
                    den = math.pow(1.0 + (x - px)*(x - px), 1.5)
                    if den > 0.0:
                        v = num / den
                    else:
                        v = 0.0
                else:
                    v = x

                if v < vmin:
                    vmin = v
                if v > vmax:
                    vmax = v

                vs[n] = v
                ppx = px
                px  = x

            hist_fixed_bins_inplace(bins, vs, vmin, vmax)

            e = 0.0
            if hcalc == 0:
                e = np.std(bins)
            elif hcalc == 1:
                total = float(np.sum(bins))
                if total > 0.0:
                    H = 0.0
                    for b in bins:
                        if b > 0:
                            p = b / total
                            H += p * math.log(p)
                    e = H / math.log(bins.size)
            elif hcalc == 2:
                m = float(np.mean(bins))
                s = np.sign(bins - m)
                c = s[1:] * s[:-1]
                e = np.sum(c > 0) / bins.size
            elif hcalc == 3:
                for k in range(bins.size - 1):
                    bins[k] = bins[k + 1] - bins[k]
                e = np.std(bins[:-1])
            else:
                e = 0.0

            out[j, i] = -e

    return out


# ---------------------------------------------------------------------------
# Color mapping: Lyapunov exponent or Entropy or Custom -> RGB (schemes)
# ---------------------------------------------------------------------------

# Scheme registry: ADD NEW SCHEMES HERE ONLY
RGB_SCHEMES: dict[str, dict] = {
    "mh": dict(
        func=colors.rgb_scheme_mh,
        defaults=dict(
            gamma=0.25,
            pos_color="FF0000",
            zero_color="000000",
            neg_color="FFFF00",
        ),
    ),

    "mh_eq": dict(
        func=colors.rgb_scheme_mh_eq,
        defaults=dict(
            gamma=1,
            pos_color="FF0000",
            zero_color="000000",
            neg_color="FFFF00",
            nbins=2048,
        ),
    ),

    "palette": dict(
        func=colors.rgb_scheme_palette_eq,
        defaults=dict(
            palette="bauhaus_primaries",
            gamma=1,
            nbins=2048,
        ),
    ),

    "multi": dict(
        func=colors.rgb_scheme_multipoint,
        defaults=dict(
            palette="bauhaus_primaries",
            gamma=1,
            nbins=2048,
        ),
    ),
}

DEFAULT_RGB_SCHEME = "mh"


def lyapunov_to_rgb(lyap: np.ndarray, specdict: dict) -> np.ndarray:
    """
    Apply a colorization scheme to the λ-field based on the 'rgb' spec.

    Syntax:
        # Markus–Hess style:
        rgb:mh                          -> use mh defaults
        rgb:mh:0.25                     -> override gamma
        rgb:mh:*:#FF0000:#FFFF00        -> keep gamma, set pos/neg colors

        # Equalized variant (γ, pos_color, neg_color, nbins):
        rgb:mh_eq                       -> defaults
        rgb:mh_eq:0.3                   -> gamma=0.3
        rgb:mh_eq:*:#00FF00:#0000FF:512 -> custom colors, nbins=512

    """
    # --- 1) choose scheme ---
    rgb_vals = specdict.get("rgb")
    if rgb_vals:
        scheme_name = str(rgb_vals[0]).strip().lower()
    else:
        scheme_name = DEFAULT_RGB_SCHEME

    scheme_cfg = RGB_SCHEMES.get(scheme_name, RGB_SCHEMES[DEFAULT_RGB_SCHEME])

    # --- 2) start from scheme defaults ---
    params = dict(scheme_cfg["defaults"])  # shallow copy

    # --- 3) optional global gamma: override if present and scheme uses gamma ---
    gamma_vals = specdict.get("gamma")
    if gamma_vals and "gamma" in params:
        try:
            params["gamma"] = float(_eval_number(gamma_vals[0]).real)
        except Exception:
            pass

     # --- 4) parse positional args from rgb:scheme:arg1:arg2:... ---
    if rgb_vals and len(rgb_vals) > 1:
        arg_tokens = rgb_vals[1:]

        # order is exactly the insertion order of defaults
        defaults = scheme_cfg["defaults"]
        order = list(defaults.keys())

        for idx, tok in enumerate(arg_tokens):
            if idx >= len(order):
                break
            name = order[idx]
            if name not in params:
                continue

            default_val = params[name]
            tok_str = str(tok).strip()
            if tok_str == "*":
                # '*' -> keep default
                continue

            # parse based on type of default
            if isinstance(default_val, (float, int)):
                try:
                    params[name] = float(_eval_number(tok_str).real)
                except Exception:
                    pass
            elif isinstance(default_val, str):
                params[name] = tok_str
            else:
                # unsupported type, leave default
                pass

    func = scheme_cfg["func"]
    return func(lyap, params)


# ---------------------------------------------------------------------------
# Spec helpers using specparser.split_chain
# ---------------------------------------------------------------------------

def _eval_number(tok: str) -> complex:
    return specparser.simple_eval_number(tok)


def _get_float(d: dict, key: str, default: float) -> float:
    vals = d.get(key)
    if not vals:
        return float(default)
    try:
        return float(_eval_number(vals[0]).real)
    except Exception:
        return float(default)


def _get_int(d: dict, key: str, default: int) -> int:
    vals = d.get(key)
    if not vals:
        return int(default)
    try:
        return int(round(float(_eval_number(vals[0]).real)))
    except Exception:
        return int(default)

# ---------------------------------------------------------------------------
# Domain / affine mapping helpers
# ---------------------------------------------------------------------------

def _get_corner(d: dict, key: str, default_x: float, default_y: float):
    """
    Parse a corner operator like:

        ll:x:y
        ul:*:y
        lr:x:*
        ll:x        (x only, y from default)
        ll          (no args, all defaults)

    '*' means "keep default". Missing args also keep defaults.
    """
    vals = d.get(key)
    if not vals:
        return float(default_x), float(default_y)

    x = default_x
    y = default_y

    try:
        # First argument: x
        if len(vals) >= 1:
            v0 = vals[0].strip()
            if v0 != "*":
                x = float(_eval_number(v0).real)

        # Second argument: y
        if len(vals) >= 2:
            v1 = vals[1].strip()
            if v1 != "*":
                y = float(_eval_number(v1).real)

    except Exception:
        # On parse error, fall back to defaults
        x, y = default_x, default_y

    return float(x), float(y)


def _build_affine_domain(
    specdict: dict,
    a0: float,
    b0: float,
    a1: float,
    b1: float,
) -> np.ndarray:
    """
    Build a 2‑D affine domain mapping from logical (u,v) in [0,1]^2
    to physical (A,B) coordinates.

    We use three corners:

        LL = lower-left   (u=0, v=0)
        UL = upper-left   (u=0, v=1)
        LR = lower-right  (u=1, v=0)

    The user can override them via:

        ll:x:y   ul:x:y   lr:x:y

    with '*' as "keep default" and optional 1-arg forms ll:x, etc.

    Additionally, 'ur:x:y' can be used to complete a rectangle when
    ul/lr are not given explicitly:

        ll:x:y, ur:ux:uy

    means "axis-aligned rectangle" from (x,y) to (ux,uy).
    """

    # 0) defaults: axis-aligned rectangle from [a0,a1] x [b0,b1]
    llx, lly = a0, b0
    ulx, uly = a0, b1
    lrx, lry = a1, b0

    # 1) apply ll/ul/lr with '*' semantics
    llx, lly = _get_corner(specdict, "ll", llx, lly)
    ulx, uly = _get_corner(specdict, "ul", ulx, uly)
    lrx, lry = _get_corner(specdict, "lr", lrx, lry)

    # 2) ur, if present and ul/lr not explicitly given, completes rectangle
    if "ur" in specdict:
        urx, ury = _get_corner(specdict, "ur", a1, b1)

        # Only fill UL/LR from UR if user *didn't* specify them directly
        if "ul" not in specdict:
            ulx, uly = llx, ury
        if "lr" not in specdict:
            lrx, lry = urx, lly

    # 3) fine-grained llx/lly/ulx/... overrides (power user layer)
    llx = _get_float(specdict, "llx", llx)
    lly = _get_float(specdict, "lly", lly)
    ulx = _get_float(specdict, "ulx", ulx)
    uly = _get_float(specdict, "uly", uly)
    lrx = _get_float(specdict, "lrx", lrx)
    lry = _get_float(specdict, "lry", lry)

    domain_affine = np.asarray(
        [llx, lly, ulx, uly, lrx, lry],
        dtype=np.float64,
    )

    # Optional sanity check: are the three points colinear?
    vx0 = lrx - llx
    vy0 = lry - lly
    vx1 = ulx - llx
    vy1 = uly - lly
    area = abs(vx0 * vy1 - vx1 * vy0)
    if area == 0.0:
        print("WARNING: affine domain is degenerate (LL, UL, LR colinear)")

    return domain_affine

def debug_affine_for_spec(spec: str) -> None:
    """
    Print the resolved affine domain and a few logical->physical
    sample points for the given spec string.
    """
    specdict = specparser.split_chain(spec)

    map_name = None
    for op in specdict.keys():
        if op in maps.MAP_TEMPLATES:
            map_name = op
            break
    if map_name is None:
        print(f"No map name found in spec {spec}")
        return

    map_cfg = maps.MAP_TEMPLATES[map_name]
    type = map_cfg.get("type", "step1d")
    domain = map_cfg["domain"].copy()

    use_seq = (type=="step1d") or (type=="step2d_ab")
    domain_idx = 0
    for i, v in enumerate(specdict[map_name]):
        if use_seq and i == 0 and _looks_like_sequence_token(v):
            continue
        try:
            domain_component = float(specparser.simple_eval_number(v).real)
        except Exception:
            continue
        if domain_idx < domain.size:
            domain[domain_idx] = domain_component
            domain_idx += 1

    a0 = _get_float(specdict, "a0", domain[0])
    b0 = _get_float(specdict, "b0", domain[1])
    a1 = _get_float(specdict, "a1", domain[2])
    b1 = _get_float(specdict, "b1", domain[3])

    domain_affine = _build_affine_domain(specdict, a0, b0, a1, b1)
    llx, lly, ulx, uly, lrx, lry = domain_affine

    print("Affine domain:")
    print(f"  LL = ({llx}, {lly})")
    print(f"  UL = ({ulx}, {uly})")
    print(f"  LR = ({lrx}, {lry})")

    def map_uv(u, v):
        A = llx + u * (lrx - llx) + v * (ulx - llx)
        B = lly + u * (lry - lly) + v * (uly - lly)
        return A, B

    samples = [
        (0.0, 0.0, "(0,0)"),
        (1.0, 0.0, "(1,0)"),
        (0.0, 1.0, "(0,1)"),
        (1.0, 1.0, "(1,1)"),
        (0.5, 0.5, "(0.5,0.5)"),
    ]
    print("Sample logical -> physical mapping:")
    for u, v, label in samples:
        A, B = map_uv(u, v)
        print(f"  {label}: (u={u}, v={v}) -> ({A}, {B})")


# ---------------------------------------------------------------------------
# spec2lyapunov: parse spec -> RGB tile
# ---------------------------------------------------------------------------

def get_map_name(spec: str)-> str:
    specdict = specparser.split_chain(spec)
    if not "map" in specdict:
        raise SystemExit(f"No 'map' found in spec {spec}")
    map_spec = specdict["map"]
    if len(map_spec)<1:
        raise SystemExit(f"map needs to specify map name")
    map_name = map_spec[0]
    if not map_name in  maps.MAP_TEMPLATES:
        raise SystemExit(f"{map_name} not in MAP_TEMPLATES")
    return map_name

def make_cfg(spec:str):

    map_name = get_map_name(spec)

    specdict = specparser.split_chain(spec)
    map_spec = specdict["map"]
    map_temp = maps.MAP_TEMPLATES[map_name]

    if not "pardict" in map_temp:
        raise SystemExit(f"{map_name} needs a pardict")

    pardict = map_temp["pardict"]
    for i,(key,value) in enumerate(pardict.items()):
        if specdict.get(key) is not None:
            param_value = specdict.get(key)[0]
        else:
            param_value = value
        pardict[key] = param_value

    map_cfg = _build_map(map_name)

    map_cfg["map_name"] = map_name

    map_type = map_cfg.get("type", "step1d")
    map_cfg["type"] = map_type
    domain = map_cfg["domain"]
    
    use_seq = (map_type=="step1d") or (map_type=="step2d_ab")
    seq_arr = _seq_to_array(DEFAULT_SEQ) if use_seq else None

    if len(map_spec)>1:
        domain_idx = 0
        for i, v in enumerate(map_spec[1:]):

            if use_seq and i == 0 and _looks_like_sequence_token(v):
                seq_str = _decode_sequence_token(v, DEFAULT_SEQ)
                seq_arr = _seq_to_array(seq_str)
                continue
            try:
                domain_component = float(specparser.simple_eval_number(v).real)
            except Exception:
                continue

            if domain_idx < domain.size:
                domain[domain_idx] = domain_component
                domain_idx += 1
    
    map_cfg["seq_arr"]=seq_arr

    a0 = _get_float(specdict, "a0", domain[0])
    b0 = _get_float(specdict, "b0", domain[1])
    a1 = _get_float(specdict, "a1", domain[2])
    b1 = _get_float(specdict, "b1", domain[3])

    map_cfg["domain_affine"] = _build_affine_domain(specdict, a0, b0, a1, b1)

    map_cfg["x0"]    = _get_float(specdict, "x0", map_cfg.get("x0", 0.5))
    map_cfg["y0"]    = _get_float(specdict, "y0", map_cfg.get("y0", 0.5))
    map_cfg["n_tr"]  = _get_int(specdict, "trans", map_cfg.get("trans", DEFAULT_TRANS))
    map_cfg["n_it"]  = _get_int(specdict, "iter", map_cfg.get("iter",  DEFAULT_ITER))
    map_cfg["eps"]   = _get_float(specdict, "eps",   DEFAULT_EPS_LYAP)

    if "entropy" in specdict:
        map_cfg["type"]=map_cfg["type"]+"_entropy"
        K = _get_int(specdict, "k", 32)
        w0 = _get_float(specdict, "w0", 0.1)
        w1 = _get_float(specdict, "w1", math.pi)
        K = max(K,2)
        map_cfg["omegas"] = np.linspace(w0, w1, K, dtype=np.float64)
        map_cfg["entropy_sign"] = int(-1)
        if len(specdict["entropy"])>0:
            map_cfg["entropy_sign"] = int(specdict["entropy"][0])


    if "hist" in specdict:
        map_cfg["type"] = map_cfg["type"] + "_hist"
        map_cfg["vcalc"] = int(0)
        map_cfg["hcalc"] = int(0)
        map_cfg["hbins"] = map_cfg["n_it"]
        if len(specdict["hist"])>0:
            map_cfg["vcalc"] = int(specdict["hist"][0])
        if len(specdict["hist"])>1:
            map_cfg["hcalc"] = int(specdict["hist"][1])
        if len(specdict["hist"])>2:
            map_cfg["hbins"] = int(specdict["hist"][2])

    return map_cfg

def spec2lyapunov(spec: str, pix: int = 5000) -> np.ndarray:

    map_cfg = make_cfg(spec)
 
    if map_cfg["type"] == "step1d":

        print("lyapunov_field_generic_1d")

        field = _lyapunov_field_1d(
            map_cfg["step"],
            map_cfg["deriv"],
            map_cfg["seq_arr"],
            map_cfg["domain_affine"],
            int(pix),
            float(map_cfg["x0"]),
            int(map_cfg["n_tr"]),
            int(map_cfg["n_it"]),
            float(map_cfg["eps"]),
            
        )

    elif map_cfg["type"] == "step2d_ab":

        print("lyapunov_field_generic_2d_ab")

        field = _lyapunov_field_2d_ab(
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
        )

    elif map_cfg["type"] == "step2d":

        print("lyapunov_field_generic_2d")

        field = _lyapunov_field_2d(
            map_cfg["step2"],
            map_cfg["jac2"],
            map_cfg["domain_affine"],
            int(pix),
            float(map_cfg["x0"]), 
            float(map_cfg["y0"]),
            int(map_cfg["n_tr"]), 
            int(map_cfg["n_it"]),
            float(map_cfg.get("eps_floor", 1e-16)),
        )

    elif map_cfg["type"] == "step1d_entropy":

        print("entropy_field_generic_1d")

        raw = _entropy_field_1d(
                map_cfg["step"],
                map_cfg["seq_arr"],
                map_cfg["domain_affine"],
                int(pix),
                float(map_cfg["x0"]),
                int(map_cfg["n_tr"]),
                int(map_cfg["n_it"]),
                map_cfg["omegas"],
        )
        # map H∈[0,1] to [-1,1] so your diverging palettes still work:
        field = map_cfg["entropy_sign"] * (2.0 * raw - 1.0)
    
    elif map_cfg["type"] == "step2d_ab_entropy":

        print("entropy_field_generic_2d_ab")

        raw = _entropy_field_2d_ab(
            map_cfg["step2_ab"],
            map_cfg["seq_arr"],
            map_cfg["domain_affine"],
            int(pix),
            float(map_cfg["x0"]), 
            float(map_cfg["y0"]),
            int(map_cfg["n_tr"]), 
            int(map_cfg["n_it"]),
            map_cfg["omegas"],
        )
        field = map_cfg["entropy_sign"] * (2.0 * raw - 1.0)
    
    elif map_cfg["type"] == "step2d_entropy":

        print("entropy_field_generic_2d")

        raw = _entropy_field_2d(
            map_cfg["step2"],
            map_cfg["domain_affine"],
            int(pix),
            float(map_cfg["x0"]), 
            float(map_cfg["y0"]),
            int(map_cfg["n_tr"]), 
            int(map_cfg["n_it"]),
            map_cfg["omegas"],
        )
        field = map_cfg["entropy_sign"] * (2.0 * raw - 1.0)
    
    elif map_cfg["type"] == "step1d_hist":

        print("hist_field_1d")

        raw = _hist_field_1d(
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
        )

        field = raw-np.median(raw)

    elif map_cfg["type"] == "step2d_ab_hist":

        print("hist_field_2d_ab")

        raw = _hist_field_2d_ab(
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
        )

        field = raw-np.median(raw)

    elif map_cfg["type"] == "step2d_hist":

        print("hist_field_2d")

        raw = _hist_field_2d(
            map_cfg["step2"],
            map_cfg["domain_affine"],
            int(pix),
            float(map_cfg["x0"]),
            float(map_cfg["y0"]),
            int(map_cfg["n_tr"]),
            int(map_cfg["n_it"]),
            int(map_cfg["vcalc"]),
            int(map_cfg["hcalc"]),
        )

        field = raw-np.median(raw)

    else:
        raise SystemExit(f"Unsupported type={map_cfg['type']} for map '{map_cfg['map_name']}'")

    rgb = lyapunov_to_rgb(field, specparser.split_chain(spec))

    return rgb

# ---------------------------------------------------------------------------
# expansion helpers
# ---------------------------------------------------------------------------

def get_all_palettes(palette_regex, maxp):
    print(f"all palette search:{palette_regex}")
    pat = regex.compile(palette_regex)
    out = []
    for k in colors.COLOR_STRINGS.keys():
        if pat.search(k):
            print(f"found palette: {k}")
            out.append(k)
            if len(out) >= maxp:
                break
    return out

def get_long_palettes(palette_regex, maxp):
    print(f"long palette search:{palette_regex}")
    pat = regex.compile(palette_regex)
    out = []
    for k in colors.COLOR_LONG_STRINGS.keys():
        if pat.search(k):
            print(f"found palette: {k}")
            out.append(k)
            if len(out) >= maxp:
                break
    return out

def get_tri_palettes(palette_regex, maxp):
    print(f"tri palette search:{palette_regex}")
    pat = regex.compile(palette_regex)
    out = []
    for k in colors.COLOR_TRI_STRINGS.keys():
        if pat.search(k):
            print(f"found palette: {k}")
            out.append(k)
            if len(out) >= maxp:
                break
    return out

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------



def main() -> None:
    p = argparse.ArgumentParser(
        "lyapounov-cli",
        description=(
            "Lyapunov fractal renderer for 1D maps.\n"
            "Maps are defined symbolically and JIT-compiled; "
            "adding a new map is just editing MAP_TEMPLATES."
        ),
    )

    p.add_argument(
        "--spec",
        required=True,
        help="Lyapunov spec (can include expandspec lists/ranges).",
    )
    p.add_argument(
        "--show-specs",
        action="store_true",
        help="Show expanded specs before rendering.",
    )
    p.add_argument(
        "--pix",
        type=int,
        default=1000,
        help="Tile width/height in pixels (Lyapunov grid resolution).",
    )
    p.add_argument(
        "--out",
        type=str,
        default="lyapunov.jpg",
        help="Output JPG path (can itself be an expandspec template).",
    )
    p.add_argument(
        "--cols",
        type=int,
        default=None,
        help="Columns if chain expands to multiple tiles.",
    )
    p.add_argument(
        "--rows",
        type=int,
        default=None,
        help="Rows if chain expands to multiple tiles.",
    )
    p.add_argument(
        "--invert",
        action="store_true",
        help="Invert final RGB colors (simple negative).",
    )
    p.add_argument(
        "--thumb",
        type=int,
        default=None,
        help="Thumbnail height in pixels (if set, save mosaic as thumbnail).",
    )
    p.add_argument(
        "--const",
        action="append",
        default=[],
        help="Add/override NAME=VALUE (parsed like spec args). Repeatable.",
    )
    p.add_argument(
        "--map",
        type=str,
        default=None,
        help="override map equation",
    )
    p.add_argument(
        "--pal",
        action="append",
        default=[],
        help="add palette",
    )
    p.add_argument(
        "--check-affine",
        action="store_true",
        help="Print affine domain mapping for the first expanded spec and exit.",
    )


    args = p.parse_args()

    if args.map is not None:
        map_name, new_expr = args.map.split("=", 1)
        if map_name in maps.MAP_TEMPLATES:
            new_der_expr = _sympy_deriv(new_expr)
            print(f"map derivative: {new_der_expr}")
            # patch the template; lazy builder will use this
            maps.MAP_TEMPLATES[map_name]["expr"] = new_expr
            maps.MAP_TEMPLATES[map_name]["deriv_expr"] = new_der_expr
            spec_str = f",modify:{map_name}:{new_expr}"
            args.spec = args.spec + spec_str
        else:
            print(f"WARNING: --map refers to unknown map '{map_name}'")

    # Apply constants (like in julia.py)
    for kv in args.const:
        print(f"const {kv}")
        k, v = specparser._parse_const_kv(kv)
        specparser.set_const(k, v)
        expandspec.set_const(k, v)

    for kv in args.pal:
        print(f"adding palette {kv}")
        k, v = kv.split("=", 1)
        colors.COLOR_STRINGS[k]=v

    # Expand output path first
    out_paths = expandspec.expand_cartesian_lists(args.out)
    if not out_paths:
        raise SystemExit("Output expandspec produced no paths")
    outfile = out_paths[0]
    print(f"will save to {outfile}")

    # spec expansion helpers
    expandspec.FUNCS["gap"]=get_all_palettes
    expandspec.FUNCS["glp"]=get_long_palettes
    expandspec.FUNCS["gtp"]=get_tri_palettes

    # Expand the main spec chain
    specs = expandspec.expand_cartesian_lists(args.spec)

    if args.check_affine:
        # just inspect the first expanded spec
        debug_affine_for_spec(specs[0])
        return
    
    if args.show_specs:
        for s in specs:
            print(s)

    if not specs:
        raise SystemExit("Spec expansion produced no tiles")

    for i, spec in enumerate(specs, start=1):
        print(f"{i}/{len(specs)} Rendering {spec}")
        t0 = time.perf_counter()
        rgb = spec2lyapunov(spec, pix=args.pix)
        print(f"field time: {time.perf_counter() - t0:.3f}s")
        # swap A/B axes and flip vertically to match Markus & Hess style
        rgb = np.flipud(rgb)
        fn = raster.add_suffix_number(outfile,i)
        raster.save_jpg_rgb(
            rgb,
            out_path=fn,
            invert=False,
            footer_text=spec,
            footer_pad_lr_px=48,
            footer_dpi=300,
        )
        print(f"saved: {fn}")

    Path(Path(outfile).name).with_suffix(".spec").write_text("\n".join(specs))


if __name__ == "__main__":
    main()

