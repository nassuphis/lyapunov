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
import fields


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
        if use_seq and i == 0 and maps.looks_like_sequence_token(v):
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

    map_cfg = maps.build_map(map_name)

    map_cfg["map_name"] = map_name

    map_type = map_cfg.get("type", "step1d")
    map_cfg["type"] = map_type
    domain = map_cfg["domain"]
    
    use_seq = (map_type=="step1d") or (map_type=="step2d_ab")
    seq_arr = maps.seq_to_array(maps.DEFAULT_SEQ) if use_seq else None

    if len(map_spec)>1:
        domain_idx = 0
        for i, v in enumerate(map_spec[1:]):

            if use_seq and i == 0 and maps.looks_like_sequence_token(v):
                seq_str = maps.decode_sequence_token(v, maps.DEFAULT_SEQ)
                seq_arr = maps.seq_to_array(seq_str)
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
    map_cfg["n_tr"]  = _get_int(specdict, "trans", map_cfg.get("trans", maps.DEFAULT_TRANS))
    map_cfg["n_it"]  = _get_int(specdict, "iter", map_cfg.get("iter",  maps.DEFAULT_ITER))
    map_cfg["eps"]   = _get_float(specdict, "eps",   maps.DEFAULT_EPS_LYAP)

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

        field = fields.lyapunov_field_1d(
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

        field = fields.lyapunov_field_2d_ab(
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

        field = fields.lyapunov_field_2d(
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

        raw = fields.entropy_field_1d(
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

        raw = fields.entropy_field_2d_ab(
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

        raw = fields.entropy_field_2d(
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

        raw = fields.hist_field_1d(
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

        raw = fields.hist_field_2d_ab(
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

        raw = fields.hist_field_2d(
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
            new_der_expr = maps.sympy_deriv(new_expr)
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

    fns = []
    for i, spec in enumerate(specs, start=1):
        fn = raster.add_suffix_number(outfile,i)
        fns.append(fn)
        if not Path(fn).exists():
            print(f"{i}/{len(specs)} Rendering {spec}")
            t0 = time.perf_counter()
            rgb = spec2lyapunov(spec, pix=args.pix)
            print(f"field time: {time.perf_counter() - t0:.3f}s")
            # swap A/B axes and flip vertically to match Markus & Hess style
            rgb = np.flipud(rgb)
            raster.save_jpg_rgb(
                rgb,
                out_path=fn,
                invert=False,
                footer_text=spec,
                footer_pad_lr_px=48,
                footer_dpi=300,
            )
            print(f"saved: {fn}")
        else:
            print(f"{fn} exists, skipping")

    lines = [f"{fn} {spec}" for fn, spec in zip(fns, specs)]
    Path(Path(outfile).name).with_suffix(".spec").write_text("\n".join(lines))


if __name__ == "__main__":
    main()

