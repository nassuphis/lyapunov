"""
Spec string → map configuration builder.

Converts human-readable spec strings into map_cfg dictionaries
ready for field computation.
"""

import math
import numpy as np

from specparser import chain as specparser
import maps
import affine


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


# ---------------------------------------------------------------------------
# spec -> map
# ---------------------------------------------------------------------------

def get_map_name(spec: str) -> str:
    specdict = specparser.split_chain(spec)
    if not "map" in specdict:
        raise SystemExit(f"No 'map' found in spec {spec}")
    map_spec = specdict["map"]
    if len(map_spec) < 1:
        raise SystemExit(f"map needs to specify map name")
    map_name = map_spec[0]
    if not map_name in maps.MAP_TEMPLATES:
        raise SystemExit(f"{map_name} not in MAP_TEMPLATES")
    return map_name


def make_cfg(spec: str, pix: int = 1000):
    """
    Build a map configuration dictionary from a spec string.

    Args:
        spec: Spec string like "map:logistic:AB:2:4:2:4,iter:1000,..."
        pix: Image resolution (needed for x0/y0 field arrays)

    Returns:
        dict with all configuration needed for field computation
    """
    map_name = get_map_name(spec)

    specdict = specparser.split_chain(spec)
    map_spec = specdict["map"]
    map_temp = maps.MAP_TEMPLATES[map_name]

    if not "pardict" in map_temp:
        raise SystemExit(f"{map_name} needs a pardict")

    pardict = map_temp["pardict"]
    for i, (key, value) in enumerate(pardict.items()):
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

    use_seq = (map_type == "step1d") or (map_type == "step2d_ab") or (map_type == "step1d_ab_x0") or (map_type == "step2d_ab_xy0")
    seq_arr = maps.seq_to_array(maps.DEFAULT_SEQ) if use_seq else None

    if len(map_spec) > 1:
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

    map_cfg["seq_arr"] = seq_arr

    a0 = _get_float(specdict, "a0", domain[0])
    b0 = _get_float(specdict, "b0", domain[1])
    a1 = _get_float(specdict, "a1", domain[2])
    b1 = _get_float(specdict, "b1", domain[3])

    map_cfg["domain_affine"] = affine.build_affine_domain(specdict, a0, b0, a1, b1)

    # Default params: empty array (can be set programmatically for precomputed values)
    map_cfg["params"] = np.empty(0, dtype=np.float64)

    map_cfg["n_tr"] = _get_int(specdict, "trans", map_cfg.get("trans", maps.DEFAULT_TRANS))
    map_cfg["n_it"] = _get_int(specdict, "iter", map_cfg.get("iter", maps.DEFAULT_ITER))
    map_cfg["eps"] = _get_float(specdict, "eps", maps.DEFAULT_EPS_LYAP)

    if "entropy" in specdict:
        map_cfg["type"] = map_cfg["type"] + "_entropy"
        K = _get_int(specdict, "k", 32)
        w0 = _get_float(specdict, "w0", 0.1)
        w1 = _get_float(specdict, "w1", math.pi)
        K = max(K, 2)
        map_cfg["omegas"] = np.linspace(w0, w1, K, dtype=np.float64)
        map_cfg["entropy_sign"] = int(-1)
        if len(specdict["entropy"]) > 0:
            map_cfg["entropy_sign"] = int(specdict["entropy"][0])

    if "hist" in specdict:
        map_cfg["type"] = map_cfg["type"] + "_hist"
        map_cfg["vcalc"] = int(0)
        map_cfg["hcalc"] = int(0)
        map_cfg["hbins"] = map_cfg["n_it"]
        if len(specdict["hist"]) > 0:
            map_cfg["vcalc"] = int(specdict["hist"][0])
        if len(specdict["hist"]) > 1:
            map_cfg["hcalc"] = int(specdict["hist"][1])
        if len(specdict["hist"]) > 2:
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
