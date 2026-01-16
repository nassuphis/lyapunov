"""
Map building functions and JIT compilation utilities.

This module contains:
- Symbolic derivative helpers (sympy)
- Function text generation for step/deriv/jacobian
- JIT compilation wrappers
- build_map() - the main map configuration builder
- Sequence handling (A/B patterns)
"""

import re as regex
import sympy as sp
import numpy as np
from numba import njit, types

from . import functions


# ---------------------------------------------------------------------------
# Symbolic derivative helper (x derivative of map expression)
# ---------------------------------------------------------------------------

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


def sympy_deriv(expr_str: str) -> str:
    expr = sp.sympify(expr_str, locals=locs)
    expr_der = sp.diff(expr, x)
    return sp.sstr(expr_der)


def sympy_jacobian_2d(expr_x: str, expr_y: str):
    fx = sp.sympify(expr_x, locals=locs)
    fy = sp.sympify(expr_y, locals=locs)
    dfx_dx = sp.diff(fx, x)
    dfx_dy = sp.diff(fx, y)
    dfy_dx = sp.diff(fy, x)
    dfy_dy = sp.diff(fy, y)
    return tuple(sp.sstr(e) for e in (dfx_dx, dfx_dy, dfy_dx, dfy_dy))


# ---------------------------------------------------------------------------
# Build python function text
# ---------------------------------------------------------------------------

def funtext_1d(name: str, expr: str, dict) -> str:
    lines = [
        f"def {name}(x, forced, params):",
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


def funtext_1d_deriv(name: str, expr: str, dict) -> str:
    """Generate derivative function text WITHOUT params (for sympy compatibility)."""
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


def funtext_2d_ab_step(name: str, expr_x: str, expr_y: str, dict) -> str:
    lines = [
        f"def {name}(x, y, forced, params):",
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


def funtext_2d_ab_jac(
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


def funtext_2d_step(name: str, expr_x: str, expr_y: str, dict) -> str:
    lines = [
        f"def {name}(x, y, first, second, params):",
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


def funtext_2d_jac(
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
# Build python functions from text
# ---------------------------------------------------------------------------

# 1D forced step (with params)
def funpy_1d(expr: str, dict):
    ns = functions.NS.copy()
    src = funtext_1d("impl", expr, dict)
    exec(src, ns, ns)
    return ns["impl"]


# 1D derivative (WITHOUT params - for sympy compatibility)
def funpy_1d_deriv(expr: str, dict):
    ns = functions.NS.copy()
    src = funtext_1d_deriv("impl_deriv", expr, dict)
    exec(src, ns, ns)
    return ns["impl_deriv"]


# 2D forced step
def funpy_2d_ab_step(expr_x: str, expr_y: str, dict):
    ns = functions.NS.copy()
    src = funtext_2d_ab_step("impl2_step", expr_x, expr_y, dict)
    exec(src, ns, ns)
    return ns["impl2_step"]


# 2D forced jacobian
def funpy_2d_ab_jac(dXdx, dXdy, dYdx, dYdy, dict):
    ns = functions.NS.copy()
    src = funtext_2d_ab_jac("impl2_jac", dXdx, dXdy, dYdx, dYdy, dict)
    exec(src, ns, ns)
    return ns["impl2_jac"]


# 2D
def funpy_2d_step(expr_x: str, expr_y: str, dict):
    ns = functions.NS.copy()
    src = funtext_2d_step("impl2_step", expr_x, expr_y, dict)
    exec(src, ns, ns)
    return ns["impl2_step"]


# 2D jacobian
def funpy_2d_jac(dXdx, dXdy, dYdx, dYdy, dict):
    ns = functions.NS.copy()
    src = funtext_2d_jac("impl2_jac", dXdx, dXdy, dYdx, dYdy, dict)
    exec(src, ns, ns)
    return ns["impl2_jac"]


# ---------------------------------------------------------------------------
# JIT function signatures
# ---------------------------------------------------------------------------

STEP_SIG = types.float64(
    types.float64,   # x, the mapped variable
    types.float64,   # forced
    types.Array(types.float64, 1, 'C'),  # params
)

DERIV_SIG = types.float64(
    types.float64,   # x, the mapped variable
    types.float64,   # forced
)

STEP2_AB_SIG = types.UniTuple(types.float64, 2)(
    types.float64,  # x
    types.float64,  # y
    types.float64,  # forced
    types.Array(types.float64, 1, 'C'),  # params
)

JAC2_AB_SIG = types.UniTuple(types.float64, 4)(
    types.float64,  # x
    types.float64,  # y
    types.float64,  # forced
)

STEP2_SIG = types.UniTuple(types.float64, 2)(
    types.float64,  # x
    types.float64,  # y
    types.float64,  # first
    types.float64,  # second
    types.Array(types.float64, 1, 'C'),  # params
)

JAC2_SIG = types.UniTuple(types.float64, 4)(
    types.float64,  # x
    types.float64,  # y
    types.float64,  # first
    types.float64,  # second
)


# ---------------------------------------------------------------------------
# JIT compilation wrappers
# ---------------------------------------------------------------------------

def funjit_1d(expr: str, dict):
    fun = funpy_1d(expr, dict)
    jit = njit(STEP_SIG, cache=False, fastmath=False)(fun)
    return jit


def funjit_1d_deriv(expr: str, dict):
    """JIT compile a 1D derivative function WITHOUT params (for sympy compatibility)."""
    fun = funpy_1d_deriv(expr, dict)
    jit = njit(DERIV_SIG, cache=False, fastmath=False)(fun)
    return jit


def funjit_2d_ab_step(xexpr: str, yexpr: str, dict):
    fun = funpy_2d_ab_step(xexpr, yexpr, dict)
    jit = njit(STEP2_AB_SIG, cache=False, fastmath=False)(fun)
    return jit


def funjit_2d_ab_jag(dxdx: str, dxdy: str, dydx: str, dydy: str, dict):
    fun = funpy_2d_ab_jac(dxdx, dxdy, dydx, dydy, dict)
    jit = njit(JAC2_AB_SIG, cache=False, fastmath=False)(fun)
    return jit


def funjit_2d_step(xexpr: str, yexpr: str, dict):
    fun = funpy_2d_step(xexpr, yexpr, dict)
    jit = njit(STEP2_SIG, cache=False, fastmath=False)(fun)
    return jit


def funjit_2d_jag(dxdx: str, dxdy: str, dydx: str, dydy: str, dict):
    fun = funpy_2d_jac(dxdx, dxdy, dydx, dydy, dict)
    jit = njit(JAC2_SIG, cache=False, fastmath=False)(fun)
    return jit


def substitute_common(x, d):
    if d is None:
        return x
    x = x.format(**d)
    return x


# ---------------------------------------------------------------------------
# build_map - main map configuration builder
# ---------------------------------------------------------------------------

def build_map(name: str, MAP_TEMPLATES: dict) -> dict:
    """
    Build a map configuration from a template.

    Args:
        name: Map name from MAP_TEMPLATES
        MAP_TEMPLATES: The combined map templates dictionary

    Returns:
        dict with compiled step/deriv functions and configuration
    """
    if name not in MAP_TEMPLATES:
        raise KeyError(f"Unknown map '{name}'")

    cfg = MAP_TEMPLATES[name]
    new_cfg = dict(cfg)
    type = cfg.get("type", "step1d")
    pardict = cfg.get("pardict", dict())
    new_cfg["pardict"] = pardict
    new_cfg["domain"] = np.asarray(cfg.get("domain", [0.0, 0.0, 1.0, 1.0]), dtype=np.float64)
    new_cfg["type"] = type

    if type in ("step1d", "step1d_x0"):
        expr = substitute_common(cfg["expr"], cfg.get("expr_common"))
        new_cfg["step"] = funjit_1d(expr, pardict)
        if "deriv_expr" in cfg:
            deriv_expr = substitute_common(cfg["deriv_expr"], cfg.get("expr_common"))
        else:
            deriv_expr = sympy_deriv(substitute_common(cfg.get("expr"), cfg.get("expr_common")))
        new_cfg["deriv"] = funjit_1d_deriv(deriv_expr, pardict)
        new_cfg["eps_floor"] = cfg.get("eps_floor", 1e-16)
        return new_cfg

    if type in ("step2d", "step2d_xy0"):
        if "step2_func" in cfg and "jac2_func" in cfg:
            new_cfg["step2"] = njit(STEP2_SIG, cache=False, fastmath=False)(cfg["step2_func"])
            new_cfg["jac2"] = njit(JAC2_SIG, cache=False, fastmath=False)(cfg["jac2_func"])
        else:
            expr_x = substitute_common(cfg["expr_x"], cfg.get("expr_common"))
            expr_y = substitute_common(cfg["expr_y"], cfg.get("expr_common"))
            new_cfg["step2"] = funjit_2d_step(expr_x, expr_y, pardict)
            if "jac_exprs" in cfg:
                dXdx, dXdy, dYdx, dYdy = cfg["jac_exprs"]
                dXdx = substitute_common(dXdx, cfg.get("expr_common"))
                dXdy = substitute_common(dXdy, cfg.get("expr_common"))
                dYdx = substitute_common(dYdx, cfg.get("expr_common"))
                dYdy = substitute_common(dYdy, cfg.get("expr_common"))
            else:
                dXdx, dXdy, dYdx, dYdy = sympy_jacobian_2d(expr_x, expr_y)
            new_cfg["jac2"] = funjit_2d_jag(dXdx, dXdy, dYdx, dYdy, pardict)
        new_cfg["eps_floor"] = cfg.get("eps_floor", 1e-16)
        return new_cfg

    if type in ("step2d_ab", "step2d_ab_xy0"):
        if "step2_func" in cfg and "jac2_func" in cfg:
            new_cfg["step2_ab"] = njit(STEP2_AB_SIG, cache=False, fastmath=False)(cfg["step2_func"])
            new_cfg["jac2_ab"] = njit(JAC2_AB_SIG, cache=False, fastmath=False)(cfg["jac2_func"])
        else:
            expr_x = substitute_common(cfg["expr_x"], cfg.get("expr_common"))
            expr_y = substitute_common(cfg["expr_y"], cfg.get("expr_common"))
            new_cfg["step2_ab"] = funjit_2d_ab_step(expr_x, expr_y, pardict)
            if "jac_exprs" in cfg:
                dXdx, dXdy, dYdx, dYdy = cfg["jac_exprs"]
                dXdx = substitute_common(dXdx, cfg.get("expr_common"))
                dXdy = substitute_common(dXdy, cfg.get("expr_common"))
                dYdx = substitute_common(dYdx, cfg.get("expr_common"))
                dYdy = substitute_common(dYdy, cfg.get("expr_common"))
            else:
                dXdx, dXdy, dYdx, dYdy = sympy_jacobian_2d(expr_x, expr_y)
            new_cfg["jac2_ab"] = funjit_2d_ab_jag(dXdx, dXdy, dYdx, dYdy, pardict)
        new_cfg["eps_floor"] = cfg.get("eps_floor", 1e-16)
        return new_cfg

    raise ValueError(f"Unsupported type={type} for map '{name}'")


# ---------------------------------------------------------------------------
# Sequence handling (A/B patterns)
# ---------------------------------------------------------------------------

SEQ_ALLOWED_RE = regex.compile(r"^[AaBb0-9()]+$")


def looks_like_sequence_token(tok: str) -> bool:
    s = tok.strip()
    if not s:
        return False
    if not SEQ_ALLOWED_RE.match(s):
        return False
    # must contain at least one A/B or '(' so "123" isn't treated as seq
    return any(ch in "AaBb(" for ch in s)


def decode_sequence_token(tok: str, default_seq: str = "AB") -> str:
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
            group_str = s[i + 1: j]
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
                    count = int(s[j + 1: k])
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


def seq_to_array(seq_str: str) -> np.ndarray:
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
