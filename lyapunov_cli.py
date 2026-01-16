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
import argparse
import numpy as np

from specparser import chain as specparser
from specparser import expander
from rasterizer import raster

from lyapunov import spec2lyapunov


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

     
    # Save the macro state used for this run
    #if not args.dry:
    #    used_macros_path = outdir / "used_macros.txt"
    #    write_used_macros(used_macros_path, run_spec=args.spec)
    #    print(f"wrote: {used_macros_path}")


if __name__ == "__main__":
    main()


#ffmpeg -framerate 20 -start_number 1 -i 'tst2/tst2_%05d.jpg' -vf "tmix=frames=10:weights='1 1 1 1 1 1 1 1 1 1',format=yuv420p" \ -c:v libx264 -crf 18 -preset medium -movflags +faststart out_avg10.mp4
