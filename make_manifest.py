#!/usr/bin/env python3
"""
make_manifest.py

Usage:
  ./make_manifest.py 'gemini2/g*_pair.jpeg' 10 400

Writes (top-level, next to gemini2.html + gemini2.dzi):
  <base_dir_name>_manifest.json   e.g. gemini2_manifest.json

Manifest is for downloading the exact file used to build the DeepZoom mosaic:
- items[] = pair JPEG filenames used by arrayjoin (in order)
- base_url = '<dir_of_glob>/' (e.g. "gemini2/")
- cell_w/cell_h inferred from first pair image
- shim = border (your join_with_border.sh uses the same value)
"""

import json
import re
import sys
from pathlib import Path

import pyvips as vips


def usage() -> None:
    print("usage: make_manifest.py '<glob>' <across> <border>", file=sys.stderr)
    print("example: make_manifest.py 'gemini2/g*_pair.jpeg' 10 400", file=sys.stderr)


def sort_key(p: Path):
    m = re.match(r"^g(\d+)_pair\.(jpe?g)$", p.name, flags=re.IGNORECASE)
    if m:
        return (0, int(m.group(1)))
    return (1, p.name.lower())


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        usage()
        return 2

    glob_pat = argv[0]
    try:
        across = int(argv[1])
        border = int(argv[2])
    except ValueError:
        usage()
        return 2

    if across <= 0 or border < 0:
        usage()
        return 2

    g = Path(glob_pat)
    base_dir = g.parent if str(g.parent) not in ("", ".") else Path(".")
    pattern = g.name

    matches = sorted(base_dir.glob(pattern), key=sort_key)
    if not matches:
        print(f"no matches for glob: {glob_pat}", file=sys.stderr)
        return 1

    # per-cell size = size of the pair jpeg you arrayjoined
    first = vips.Image.new_from_file(str(matches[0]), access="random")
    cell_w, cell_h = first.width, first.height

    # where downloadable items live (the pair jpgs)
    base_url = (str(base_dir).rstrip("/\\") + "/") if str(base_dir) != "." else ""

    # top-level output next to gemini2.html + gemini2.dzi
    out_path = Path(f"{base_dir.name}_manifest.json")

    manifest = {
        "across": across,
        "border": border,   # outer mosaic border you used in embed (also needed for click mapping)
        "shim": border,     # arrayjoin shim (you used 400)
        "cell_w": cell_w,
        "cell_h": cell_h,
        "base_url": base_url,  # e.g. "gemini2/"
        "items": [p.name for p in matches],  # e.g. "g00001_pair.jpeg", ...
    }

    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {out_path} with {len(matches)} items")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

