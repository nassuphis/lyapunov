#!/usr/bin/env python
import sys
import os
import pyvips
import math


def equalize_luminance_hsv(in_path: str,
                           out_path: str | None = None,
                           gamma: float = 1.0) -> str:

    if out_path is None:
        base, ext = os.path.splitext(in_path)
        out_path = f"{base}_eq{ext}"

    image = pyvips.Image.new_from_file(in_path)

    # If grayscale, equalize directly
    if image.bands < 3:
        V = image
        V_eq = V.hist_equal()
        out = V_eq
        if gamma != 1.0:
            out = (out.cast("float") / 255.0) ** gamma
            out = (out * 255.0).cast("uchar")
        out.write_to_file(out_path)
        return out_path

    # Convert to HSV
    hsv = image.colourspace("hsv")
    H, S, V = hsv[0], hsv[1], hsv[2]

    # Normalize V to uchar for hist_equal
    if V.format != "uchar":
        v_min = V.min()
        v_max = V.max()
        rng = max(1e-6, v_max - v_min)
        V_u8 = ((V - v_min) * (255.0 / rng)).cast("uchar")
    else:
        v_min, v_max = 0, 255
        rng = 255
        V_u8 = V

    # Histogram equalization
    V_eq_u8 = V_u8.hist_equal()

    # Convert back to original format/range
    V_eq = V_eq_u8.cast("float")
    V_eq = (V_eq * (rng / 255.0)) + v_min
    V_eq = V_eq.cast(V.format)

    # --- APPLY GAMMA HERE ---
    if gamma != 1.0:
        V_f = V_eq.cast("float")
        V_norm = (V_f - v_min) / rng
        V_gamma = (V_norm ** gamma)
        V_eq = (V_gamma * rng + v_min).cast(V.format)

    # Rebuild HSV and convert to sRGB
    hsv_eq = H.bandjoin([S, V_eq])
    out = hsv_eq.colourspace("srgb")
    out.write_to_file(out_path)

    return out_path


def main(argv: list[str]) -> None:
    if len(argv) < 1 or len(argv) > 3:
        print("Usage: equalize.py input.png [output.png] [gamma]")
        sys.exit(1)

    in_path = argv[0]
    out_path = None
    gamma = 1.0

    if len(argv) >= 2:
        # second arg can be gamma or output
        if argv[1].replace(".", "", 1).isdigit():
            gamma = float(argv[1])
        else:
            out_path = argv[1]

    if len(argv) == 3:
        gamma = float(argv[2])

    written = equalize_luminance_hsv(in_path, out_path, gamma)
    print(f"Wrote: {written}")


if __name__ == "__main__":
    main(sys.argv[1:])
