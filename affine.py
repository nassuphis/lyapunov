
import sys
from pathlib import Path
parent = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent))
import numpy as np

from specparser import specparser, expandspec

import maps

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


def build_affine_domain(
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

    domain_affine = build_affine_domain(specdict, a0, b0, a1, b1)
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


