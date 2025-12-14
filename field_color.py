import sys
from pathlib import Path
parent = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent))
import numpy as np
from rasterizer import colors
from specparser import specparser

# ---------------------------------------------------------------------------
# Color mapping: Lyapunov exponent or Entropy or Custom -> RGB (schemes)
# ---------------------------------------------------------------------------

RGB_SCHEMES: dict[str, dict] = {

    # ------------------------------------------------------------------
    # Markus–Hess (linear, RGB)
    # rgb:mh:gamma:pos:zero:neg
    # ------------------------------------------------------------------
    "mh": dict(
        func=colors.rgb_scheme_mh,
        defaults=dict(
            gamma=0.25,
            pos_color="FF0000",
            zero_color="000000",
            neg_color="FFFF00",
        ),
        args=["gamma","pos_color","zero_color","neg_color",],
    ),

    # ------------------------------------------------------------------
    # Markus–Hess with histogram equalization (RGB)
    # rgb:mh_eq:gamma:pos:zero:neg:nbins
    # ------------------------------------------------------------------
    "mh_eq": dict(
        func=colors.rgb_scheme_mh_eq,
        defaults=dict(
            gamma=1,
            pos_color="FF0000",
            zero_color="000000",
            neg_color="FFFF00",
            nbins=2048,
        ),
        args=["gamma","pos_color","zero_color","neg_color","nbins",],
    ),

    # ------------------------------------------------------------------
    # Markus–Hess with histogram equalization (HSV interpolation)
    # rgb:hsveq:gamma:pos:zero:neg:nbins
    # ------------------------------------------------------------------
    "hsveq": dict(
        func=colors.hsv_scheme_mh_eq,
        defaults=dict(
            gamma=1,
            pos_color="FF0000",
            zero_color="000000",
            neg_color="FFFF00",
            nbins=2048,
        ),
        args=["gamma","pos_color","zero_color","neg_color","nbins",],
    ),

    # ------------------------------------------------------------------
    # Tri-palette, histogram-equalized (RGB)
    # rgb:palette:palette:gamma:nbins
    # ------------------------------------------------------------------
    "palette": dict(
        func=colors.rgb_scheme_palette_eq,
        defaults=dict(
            palette="bauhaus_primaries",
            gamma=1,
            nbins=2048,
        ),
        args=["palette","gamma","nbins",],
    ),

    # ------------------------------------------------------------------
    # Tri-palette, histogram-equalized (HSV)
    # rgb:palhsv:palette:gamma:nbins
    # ------------------------------------------------------------------
    "palhsv": dict(
        func=colors.hsv_scheme_palette_eq,
        defaults=dict(
            palette="bauhaus_primaries",
            gamma=1,
            nbins=2048,
        ),
        args=["palette","gamma","nbins",
        ],
    ),

    # ------------------------------------------------------------------
    # Multi-point palette (non-diverging)
    # rgb:multi:palette:gamma
    # ------------------------------------------------------------------
    "multi": dict(
        func=colors.rgb_scheme_multipoint,
        defaults=dict(
            palette="bauhaus_primaries",
            gamma=1,
        ),
        args=["palette","gamma",
        ],
    ),

    # ------------------------------------------------------------------
    # Palette fields
    # ------------------------------------------------------------------
    "pfg": dict(
         func=colors.rgb_scheme_palette_field,
         defaults=dict(
             paletteA="rg",
             paletteB="ink_prussian_mint",
             norm="eq",
             gamma=1,
             nbins=2048,
             w_feature="grad",
             w_lo=10.0,
             w_hi=99.0,
             w_sigma=2.0,
             w_gamma=1.0,
             w_strength=1.0,
             # feature parameters
             sigma=1, 
         ),
         args=["paletteA","paletteB","w_strength","gamma","nbins",],
     ),

     "pfgx": dict(
         func=colors.rgb_scheme_palette_field,
         defaults=dict(
             paletteA="rg",
             paletteB="ink_prussian_mint",
             norm="eq",
             gamma=1,
             nbins=2048,
             w_feature="gradx",
             w_lo=10.0,
             w_hi=99.0,
             w_sigma=2.0,
             w_gamma=1.0,
             w_strength=1.0,
             # feature parameters
             sigma=1, 
         ),
         args=["paletteA","paletteB","w_strength","gamma","nbins",],
     ),

     "pfgy": dict(
         func=colors.rgb_scheme_palette_field,
         defaults=dict(
             paletteA="rg",
             paletteB="ink_prussian_mint",
             norm="eq",
             gamma=1,
             nbins=2048,
             w_feature="grady",
             w_lo=10.0,
             w_hi=99.0,
             w_sigma=2.0,
             w_gamma=1.0,
             w_strength=1.0,
             # feature parameters
             sigma=1, 
         ),
         args=["paletteA","paletteB","w_strength","gamma","nbins",],
     ),


     "pfd": dict( # the dog
         func=colors.rgb_scheme_palette_field,
         defaults=dict(
             paletteA="rg",
             paletteB="ink_prussian_mint",
             norm="eq",
             gamma=1,
             nbins=2048,
             w_feature="dog",
             w_lo=10.0,
             w_hi=99.0,
             w_sigma=2.0,
             w_gamma=1.0,
             w_strength=1.0,
             # feature parameters
             sigma1=1, 
             sigma2=6, 
             energy_sigma=1.0,
             mode="energy"
         ),
         args=["paletteA","paletteB","w_strength","sigma1","sigma2","energy_sigma","mode","gamma","nbins",],
     ),

    "pfgdx": dict(
        func=colors.rgb_scheme_palette_field,
        defaults=dict(
            paletteA="rg",
            paletteB="ink_prussian_mint",
            norm="eq",
            gamma=1,
            nbins=2048,
            w_feature="grad_dir",
            w_lo=10.0,
            w_hi=99.0,
            w_sigma=2.0,
            w_gamma=1.0,
            w_strength=1.0,
            # feature parameters
            theta=0.0,          # fixed x
            sigma=2.0,
            mode="strength",
        ),
        args=["paletteA", "paletteB", "w_strength", "sigma", "gamma", "nbins"],
    ),

    "pfgdy": dict(
        func=colors.rgb_scheme_palette_field,
        defaults=dict(
            paletteA="rg",
            paletteB="ink_prussian_mint",
            norm="eq",
            gamma=1,
            nbins=2048,
            w_feature="grad_dir",
            w_lo=10.0,
            w_hi=99.0,
            w_sigma=2.0,
            w_gamma=1.0,
            w_strength=1.0,
            # feature parameters
            theta = 0.5 * np.pi,  # fixed y
            sigma = 2.0,
            mode="strength",
            
        ),
        args=["paletteA", "paletteB", "w_strength", "sigma", "gamma", "nbins"],
    ),

    "pfgdt": dict(
        func=colors.rgb_scheme_palette_field,
        defaults=dict(
            paletteA="rg",
            paletteB="ink_prussian_mint",
            norm="eq",
            gamma=1,
            nbins=2048,
            w_feature="grad_dir",
            w_lo=75.0,
            w_hi=99.0,
            w_sigma=2.0,
            w_gamma=1.0,
            w_strength=1.0,
            # feature parameters
            theta=0.0,
            sigma=2,
            mode="align",            
        ),
        args=["paletteA", "paletteB", "w_strength", "theta", "sigma", "mode", "gamma", "nbins"],
    ),

    # Laplacian / curvature energy
    "pfl": dict(
        func=colors.rgb_scheme_palette_field,
        defaults=dict(
            paletteA="rg",
            paletteB="ink_prussian_mint",
            norm="eq",
            gamma=1,
            nbins=2048,
            w_feature="lap",
            w_lo=10.0,
            w_hi=99.0,
            w_sigma=2.0,
            w_gamma=1.0,
            w_strength=1.0,
            # feature parameters
            sigma=2,        # size of local window (box iterations)
        ),
        args=["paletteA", "paletteB", "w_strength", "sigma" ,"gamma", "nbins"],
    ),

    # local variance (“texture energy”)
    "pfv": dict(
        func=colors.rgb_scheme_palette_field,
        defaults=dict(
            paletteA="rg",
            paletteB="ink_prussian_mint",
            norm="eq",
            gamma=1,
            nbins=2048,
            w_feature="lvar",
            w_lo=10.0,
            w_hi=99.0,
            w_sigma=2.0,
            w_gamma=1.0,
            w_strength=1.0,
            # feature parameters
            sigma=2,        # size of local window (box iterations)
        ),
        # expose blur because it changes the “scale” of texture
        args=["paletteA", "paletteB", "w_strength", "sigma", "gamma", "nbins"],
    ),

    # sign coherence (how stable the sign is locally)
    "pfs": dict(
        func=colors.rgb_scheme_palette_field,
        defaults=dict(
            paletteA="rg",
            paletteB="ink_prussian_mint",
            norm="eq",
            gamma=1,
            nbins=2048,
            w_feature="sign_coh",
            w_lo=1.0,        # coherence is already [0,1]-ish; narrower percentiles are nicer
            w_hi=99.0,
            w_sigma=2.0,
            w_gamma=1.0,
            w_strength=1.0,
            # feature parameters
            sigma=2,
        ),
        args=["paletteA", "paletteB", "w_strength", "sigma", "gamma", "nbins"],
    ),

    "pfst": dict(
        func=colors.rgb_scheme_palette_field,
        defaults=dict(
            paletteA="rg",
            paletteB="ink_prussian_mint",
            norm="eq",
            gamma=1,
            nbins=2048,
            w_feature="st_coh",
            w_lo=10.0,         # often you can even try 0.0 here
            w_hi=99.0,
            w_sigma=1.0,        # usually less needed; tensor already smooths
            w_gamma=1.0,
            w_strength=1.0,
            # structure tensor parameters
            sigma_pre=1.0,      # pre-blur before gradients
            sigma_tensor=3.0,   # tensor smoothing scale           
        ),
        args=["paletteA", "paletteB", "w_strength", "sigma_pre", "sigma_tensor", "gamma", "nbins"],
    ),

    "pfgb": dict(
        func=colors.rgb_scheme_palette_field,
        defaults=dict(
            paletteA="rg",
            paletteB="ink_prussian_mint",
            norm="eq",
            gamma=1,
            nbins=2048,

            w_feature="gabor_max",
            w_lo=10.0,
            w_hi=99.0,
            w_sigma=1.0,
            w_gamma=1.0,
            w_strength=1.0,

            # gabor params
            sigma=3.0,
            freq=0.12,
            ntheta=8,
            theta0=0,
            theta1=np.pi,
            gabor_gamma=1.0,
            pre_sigma=0.5,
        ),
        args=["paletteA", "paletteB", "w_strength", "sigma", "freq", "ntheta", "gabor_gamma", "nbins"],
    ),

    "pfgbt": dict(
        func=colors.rgb_scheme_palette_field,
        defaults=dict(
            paletteA="rg",
            paletteB="ink_prussian_mint",
            norm="eq",
            gamma=1,
            nbins=2048,

            w_feature="gabor_theta",
            w_lo=10.0,
            w_hi=99.0,
            w_sigma=1.0,
            w_gamma=1.0,
            w_strength=1.0,

            # gabor params
            sigma=3.0,
            freq=0.12,
            theta=0.0,
            gabor_gamma=1.0,
            pre_sigma=0.5,
        ),
        args=["paletteA", "paletteB", "w_strength", "theta", "sigma", "freq", "gabor_gamma", "nbins"],
    ),

    "pfm": dict(
        func=colors.rgb_scheme_palette_field,
        defaults=dict(
            paletteA="rg",
            paletteB="ink_prussian_mint",
            norm="eq",
            gamma=1,
            nbins=2048,

            w_feature="ms_ratio",
            w_lo=0.0,
            w_hi=100.0,
            w_sigma=1.0,
            w_gamma=1.0,
            w_strength=1.0,

            # multiscale params (coarse mood)
            s1=1.0,
            s2=3.0,
            s3=9.0,
            se=1.0,
            pre_sigma=0.5,
            power=1.0,
        ),
        args=["paletteA", "paletteB", "w_strength", "s1", "s2", "s3", "se", "gamma", "nbins"],
    ),


}

# aliases
RGB_SCHEMES["eq"] = RGB_SCHEMES["mh_eq"]

DEFAULT_RGB_SCHEME = "mh"

def _coerce_param(tok_str: str, default_val):
    """
    Coerce a string token to the type of default_val.

    Returns (value, used_default: bool)
    """
    if tok_str == "*":
        return default_val, True

    if isinstance(default_val, bool):
        low = tok_str.lower()
        if low in ("1", "true", "yes", "y", "on"):
            return True, False
        if low in ("0", "false", "no", "n", "off"):
            return False, False
        return default_val, True

    if isinstance(default_val, int):
        try:
            return int(round(float(specparser.simple_eval_number(tok_str).real))), False
        except Exception:
            return default_val, True

    if isinstance(default_val, float):
        try:
            return float(specparser.simple_eval_number(tok_str).real), False
        except Exception:
            return default_val, True

    if isinstance(default_val, str):
        return tok_str, False

    return default_val, True


def _parse_name_value(tok_str: str):
    # returns (name, value_str) or (None, None)
    if "=" not in tok_str:
        return None, None
    name, value = tok_str.split("=", 1)
    name = name.strip()
    value = value.strip()
    if not name:
        return None, None
    return name, value




def lyapunov_to_rgb(lyap: np.ndarray, specdict: dict) -> np.ndarray:
    rgb_vals = specdict.get("rgb")
    scheme_name = str(rgb_vals[0]).strip().lower() if rgb_vals else DEFAULT_RGB_SCHEME
    scheme_cfg = RGB_SCHEMES.get(scheme_name, RGB_SCHEMES[DEFAULT_RGB_SCHEME])

    params = dict(scheme_cfg["defaults"])

    # global gamma override
    gamma_vals = specdict.get("gamma")
    if gamma_vals and "gamma" in params:
        try:
            params["gamma"] = float(specparser.simple_eval_number(str(gamma_vals[0])).real)
        except Exception:
            pass

    if rgb_vals and len(rgb_vals) > 1:
        tokens = [str(t).strip() for t in rgb_vals[1:]]

        # 1) collect name=value overrides (do not consume positional slots)
        named = []
        positional = []
        for tok in tokens:
            if tok == "*":
                positional.append(tok)
                continue
            k, v = _parse_name_value(tok)
            if k is not None:
                named.append((k, v))
            else:
                positional.append(tok)

        # 2) apply positional tokens in arg order
        arg_order = scheme_cfg.get("args", list(params.keys()))
        pos_i = 0
        for name in arg_order:
            if pos_i >= len(positional):
                break
            tok_str = positional[pos_i]
            pos_i += 1
            val, _ = _coerce_param(tok_str, params[name])
            params[name] = val

        # 3) apply named overrides (independent of position)
        for k, v_str in named:
            if k not in params:
                # unknown keys: ignore silently (or raise if you prefer)
                continue
            val, _ = _coerce_param(v_str, params[k])
            params[k] = val

    return scheme_cfg["func"](lyap, params)



