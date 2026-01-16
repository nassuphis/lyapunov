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

import time
import math
import argparse
import numpy as np

from specparser import chain as specparser
from specparser import expander
from rasterizer import raster

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


def _parse_range(args: list) -> tuple[float, float]:
    """
    Parse range from args list.

    - [] -> (0.0, 1.0)  # default
    - [hi] -> (0.0, hi)  # one number means 0 to that
    - [a, b] -> (min(a,b), max(a,b))  # two numbers define lo and hi
    """
    if len(args) == 0:
        return 0.0, 1.0
    elif len(args) == 1:
        return 0.0, float(args[0])
    else:
        a, b = float(args[0]), float(args[1])
        return min(a, b), max(a, b)


def _make_initial_field(spec_list: list | str, pix: int) -> np.ndarray:
    """
    Generate a (pix, pix) array of initial conditions based on spec.

    spec_list can be a string (legacy) or list of strings.

    Range parsing (applies to noise, grad, image):
      - no args -> [0, 1]
      - one arg (hi) -> [0, hi]
      - two args (a, b) -> [min(a,b), max(a,b)]

    Supported specs:
      - "noise" or "noise:hi" or "noise:lo:hi": uniform random
      - "0.5" (any float): constant value
      - "grad:x" or "grad:x:hi" or "grad:x:lo:hi": horizontal gradient
      - "grad:y" or "grad:y:hi" or "grad:y:lo:hi": vertical gradient
      - "image:filepath" or "image:filepath:hi" or "image:filepath:lo:hi": load grayscale
    """
    # Normalize to list
    if isinstance(spec_list, str):
        spec_list = [spec_list]

    if not spec_list:
        raise ValueError("Empty initial field spec")

    cmd = str(spec_list[0]).strip().lower()
    args = [str(s).strip() for s in spec_list[1:]]

    # noise or noise:hi or noise:lo:hi
    if cmd in ("noise", "random"):
        lo, hi = _parse_range(args)
        return (np.random.rand(pix, pix) * (hi - lo) + lo).astype(np.float64)

    # grad:x or grad:x:hi or grad:x:lo:hi  /  grad:y or grad:y:hi or grad:y:lo:hi
    if cmd == "grad":
        if not args:
            raise ValueError("grad requires axis: grad:x or grad:y")
        axis = args[0].lower()
        lo, hi = _parse_range(args[1:])
        g = np.linspace(lo, hi, pix, dtype=np.float64)
        if axis == "x":
            return np.broadcast_to(g, (pix, pix)).copy()
        elif axis == "y":
            return np.broadcast_to(g[:, np.newaxis], (pix, pix)).copy()
        else:
            raise ValueError(f"grad axis must be 'x' or 'y', got '{axis}'")

    # image:filepath or image:filepath:hi or image:filepath:lo:hi
    if cmd == "image":
        if not args:
            raise ValueError("image requires filepath: image:path/to/file.jpg")
        filepath = args[0]
        lo, hi = _parse_range(args[1:])
        return _load_image_as_field(filepath, pix, lo, hi)

    # Legacy single-word shortcuts
    if cmd == "gradx":
        g = np.linspace(0.0, 1.0, pix, dtype=np.float64)
        return np.broadcast_to(g, (pix, pix)).copy()

    if cmd == "grady":
        g = np.linspace(0.0, 1.0, pix, dtype=np.float64)
        return np.broadcast_to(g[:, np.newaxis], (pix, pix)).copy()

    # Try to parse as a constant float
    try:
        val = float(_eval_number(cmd).real)
        return np.full((pix, pix), val, dtype=np.float64)
    except Exception:
        pass

    raise ValueError(f"Unknown initial field spec: {spec_list}")


def _load_image_as_field(filepath: str, pix: int, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """
    Load an image file, convert to grayscale, resize to pix x pix,
    and return as float64 array scaled to [lo, hi].

    Applies rot90 to align image coordinates with the project's convention
    (image y=0 at top -> mathematical y=0 at bottom).
    """
    try:
        import pyvips
    except ImportError:
        raise ImportError("pyvips required for image loading. Install with: pip install pyvips")

    img = pyvips.Image.new_from_file(filepath, access="sequential")

    # Convert to grayscale if needed
    if img.bands > 1:
        img = img.colourspace("b-w")

    # Rotate 90° CW to align with project coordinate convention
    img = img.rot270()

    # Resize to pix x pix
    scale = pix / max(img.width, img.height)
    img = img.resize(scale)

    # Crop/pad to exact pix x pix (center crop if needed)
    if img.width != pix or img.height != pix:
        # Embed in center of pix x pix canvas
        left = (pix - img.width) // 2
        top = (pix - img.height) // 2
        img = img.embed(left, top, pix, pix, extend="copy")

    # Convert to numpy and scale to [lo, hi]
    arr = np.ndarray(
        buffer=img.write_to_memory(),
        dtype=np.uint8,
        shape=[img.height, img.width]
    )
    normalized = arr.astype(np.float64) / 255.0  # [0, 1]
    return normalized * (hi - lo) + lo  # [lo, hi]



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

def make_cfg(spec:str, pix:int=1000):

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
    
    use_seq = (map_type=="step1d") or (map_type=="step2d_ab") or (map_type=="step1d_ab_x0") or (map_type=="step2d_ab_xy0")
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

    # Default params: empty array (can be set programmatically for precomputed values)
    map_cfg["params"] = np.empty(0, dtype=np.float64)

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

    # Handle x0/y0: arrays for x0/xy0 map types, scalars otherwise
    if "_x0" in map_cfg["type"] or "_xy0" in map_cfg["type"]:
        # For x0/xy0 types, spec can be a list like ["noise", "0", "1"] or ["image", "path.jpg"]
        # map_cfg defaults are strings like "noise" or lists like ["grad", "x"]
        x0_spec = specdict.get("x0") or map_cfg.get("x0", ["noise"])
        y0_spec = specdict.get("y0") or map_cfg.get("y0", ["noise"])

        # Normalize string defaults to list
        if isinstance(x0_spec, str):
            x0_spec = [x0_spec]
        if isinstance(y0_spec, str):
            y0_spec = [y0_spec]

        map_cfg["x0"] = _make_initial_field(x0_spec, pix)
        if "_xy0" in map_cfg["type"]:
            map_cfg["y0"] = _make_initial_field(y0_spec, pix)
    else:
        map_cfg["x0"] = _get_float(specdict, "x0", map_cfg.get("x0", 0.5))
        map_cfg["y0"] = _get_float(specdict, "y0", map_cfg.get("y0", 0.5))

    return map_cfg

# ---------------------------------------------------------------------------
# field calls
# ---------------------------------------------------------------------------


def spec2lyapunov(spec: str, pix: int = 5000) -> np.ndarray:

    map_cfg = make_cfg(spec, pix)
 
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

    elif map_cfg["type"] == "step1d_x0_hist":
        print("hist_field_1d_x0")
        field = fields.do_hist_field_1d_x0(map_cfg)

    elif map_cfg["type"] == "step2d_ab_xy0_hist":
        print("hist_field_2d_ab_xy0")
        field = fields.do_hist_field_2d_ab_xy0(map_cfg)

    elif map_cfg["type"] == "step2d_xy0_hist":
        print("hist_field_2d_xy0")
        field = fields.do_hist_field_2d_xy0(map_cfg)

    elif map_cfg["type"] in ("step1d_x0", "step2d_ab_xy0", "step2d_xy0"):
        raise SystemExit(
            f"type={map_cfg['type']} only supported with hist mode. "
            f"Add 'hist:...' to your spec."
        )

    else:
        raise SystemExit(f"Unsupported type={map_cfg['type']} for map '{map_cfg['map_name']}'")

    rgb = field_color.lyapunov_to_rgb(field, specparser.split_chain(spec))

    return rgb


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def write_used_macros(path: Path, run_spec: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S %z")
    lines = [f"# {ts}", f"@RUN={run_spec}"]

    # preserve macro insertion order (do NOT sort)
    for k, v in expander.MACROS.items():
        lines.append(f"{k}={v}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

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
        "spec",
        help="Lyapunov spec (can include expandspec lists/ranges).",
    )
    p.add_argument(
        "--show-specs",
        action="store_true",
        help="Show expanded specs before rendering.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite move.",
    )
    p.add_argument(
        "--pix",
        type=int,
        default=1000,
        help="Tile width/height in pixels (Lyapunov grid resolution).",
    )
    p.add_argument(
        "--pooled-rgb",
        type=int,
        default=10,
        help="Pooled rgb autilevels.",
    )
    p.add_argument(
        "--out",
        type=str,
        default="lyapunov/lyapunov.jpg",
        help="Output JPG path (can itself be an expandspec template).",
    )
    p.add_argument(
        "--macro-add",
        action="append",
        default=[],
        dest="macro_add",
        help="Add/override MACRO like @NAME=VALUE. Repeatable.",
    )
    p.add_argument(
        "--macro",
        type=str,
        default="macros.txt",
        help="Macro file.",
    )
    p.add_argument(
        "--dry",
        action="store_true",
        help="Dry run.",
    )
    p.add_argument(
        "--macro-only",
        action="store_true",
        help="Only apply macro.",
    )
    p.add_argument(
        "--specs-only",
        action="store_true",
        help="Only apply macro.",
    )
    p.add_argument(
        "--no-auto",
        action="store_true",
        help="No auto-levels.",
    )
    p.add_argument(
        "--no-text",
        action="store_true",
        help="No text.",
    )
    p.add_argument(
        "--resize",
        type=int,
        default=None,
        help="Final resize",
    )
    

    args = p.parse_args()

    raster.autolvlcfg.pooled_rgb=args.pooled_rgb
    expander.macro_init(args.macro)

    # Apply macro overrides/additions
    for kv in args.macro_add:
        print(f"macro {kv}")
        k, v = kv.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not k.startswith("@"):
            k = "@" + k
        expander.MACROS[k] = v

    if args.macro_only:
        print(expander.macro(args.spec))
        return
    
    # result location
    out_schema = Path(args.out)
    outdir = out_schema.resolve().parent
    stem = out_schema.stem
    suffix = out_schema.suffix or ".jpg"
    if not suffix.startswith("."): suffix = "." + suffix
    expander.DICT["outdir"]=outdir
    expander.DICT["outstem"]=stem
    expander.DICT["outsuffix"] = suffix
    expander.DICT["outschema"] = outdir / stem
    specparser.NAMES["outschema"] = outdir / stem


    # Expand the main spec chain
    specs = expander.expand(expander.macro(args.spec))

    if args.show_specs or args.specs_only: 
        for s in specs: print(s)
        if args.specs_only: return

    print(f"will save to {outdir} as {stem}_NNNNN{suffix}")
    outdir.mkdir(parents=True, exist_ok=True)

    if not specs:
        raise SystemExit("Spec expansion produced no tiles")

    for i, spec in enumerate(specs, start=1):
        spec = specparser.add_slot(spec)
        sid = specparser.slot_suffix(spec, width=5)
        out_path = outdir / f"{stem}_{sid}{suffix}"
        spec_path = out_path.with_suffix(".spec")
       
        if out_path.exists() and not args.overwrite:
            print(f"{out_path} exists, skipping")
            continue

        print(f"{i}/{len(specs)} Rendering {spec}")
        
        if args.dry: continue

        t0 = time.perf_counter()
        rgb = spec2lyapunov(spec, pix=args.pix)
        print(f"field time: {time.perf_counter() - t0:.3f}s")

        rgb = np.flipud(rgb)

        if args.no_text:
            footer = None
        else:
            footer = spec

        raster.save_jpg_rgb(
            rgb,
            out_path=str(out_path),
            footer_text=footer,
            footer_pad_lr_px=48,
            footer_dpi=300,
            spec=spec,
            autolvl=(not args.no_auto),
            resize=args.resize,
        )
        spec_path.write_text(spec + "\n", encoding="utf-8")
        print(f"saved: {out_path}")

        #    subprocess.run(
        #        ["bash", "autolevels.sh", str(tmp_path), str(out_path)],
        #        check=True,
        #    )
        #    print(f"autoleveled: {out_path} (deleted {tmp_path})")
        #    tmp_path.unlink()

        
       

    # Save the macro state used for this run
    #if not args.dry:
    #    used_macros_path = outdir / "used_macros.txt"
    #    write_used_macros(used_macros_path, run_spec=args.spec)
    #    print(f"wrote: {used_macros_path}")


if __name__ == "__main__":
    main()


#ffmpeg -framerate 20 -start_number 1 -i 'tst2/tst2_%05d.jpg' -vf "tmix=frames=10:weights='1 1 1 1 1 1 1 1 1 1',format=yuv420p" \ -c:v libx264 -crf 18 -preset medium -movflags +faststart out_avg10.mp4
