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

from pathlib import Path
import time
import math
import argparse
import re as regex
import numpy as np

from specparser import specparser, expandspec
from rasterizer import raster
from rasterizer import colors

import maps
import fields
import affine
import field_color


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
# spec -> map 
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

    map_cfg["domain_affine"] = affine.build_affine_domain(specdict, a0, b0, a1, b1)

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

# ---------------------------------------------------------------------------
# field calls
# ---------------------------------------------------------------------------


def spec2lyapunov(spec: str, pix: int = 5000) -> np.ndarray:

    map_cfg = make_cfg(spec)
 
    if map_cfg["type"] == "step1d":
        print("lyapunov_field_generic_1d")
        field=fields.do_lyapunov_field_1d(map_cfg,pix)

    elif map_cfg["type"] == "step2d_ab":
        print("lyapunov_field_generic_2d_ab")
        field=fields.do_lyapunov_field_2d_ab(map_cfg,pix)

    elif map_cfg["type"] == "step2d":
        print("lyapunov_field_generic_2d")
        field=fields.do_lyapunov_field_2d(map_cfg,pix)

    elif map_cfg["type"] == "step1d_entropy":
        print("entropy_field_generic_1d")
        field=fields.do_entropy_field_1d(map_cfg,pix)
    
    elif map_cfg["type"] == "step2d_ab_entropy":
        print("entropy_field_generic_2d_ab")
        field=fields.do_entropy_field_2d_ab(map_cfg,pix)
    
    elif map_cfg["type"] == "step2d_entropy":
        print("entropy_field_generic_2d")
        field=fields.do_entropy_field_2d(map_cfg,pix)
    
    elif map_cfg["type"] == "step1d_hist":
        print("hist_field_1d")
        field=fields.do_hist_field_1d(map_cfg,pix)

    elif map_cfg["type"] == "step2d_ab_hist":
        print("hist_field_2d_ab")
        field=fields.do_hist_field_2d_ab(map_cfg,pix)

    elif map_cfg["type"] == "step2d_hist":
        print("hist_field_2d")
        field=fields.do_hist_field_2d(map_cfg,pix)

    else:
        raise SystemExit(f"Unsupported type={map_cfg['type']} for map '{map_cfg['map_name']}'")

    rgb = field_color.lyapunov_to_rgb(field, specparser.split_chain(spec))

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
        "--pal",
        action="append",
        default=[],
        help="add palette",
    )
    

    args = p.parse_args()

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

