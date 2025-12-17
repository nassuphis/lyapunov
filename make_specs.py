#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path
parent = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent))

import argparse
from pathlib import Path

from rasterizer.raster import read_spec_exiftool


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Create a lexicographically ordered .spec sidecar from JPEG metadata."
    )
    ap.add_argument("dir", type=Path, help="Directory containing images")
    ap.add_argument("--glob", default="*.jpg")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--include-filename", action="store_true")
    args = ap.parse_args()

    d = args.dir
    if not d.is_dir():
        raise SystemExit(f"Not a directory: {d}")

    files = sorted(d.glob(args.glob), key=lambda p: p.name)
    if not files:
        raise SystemExit("No files matched")

    out = args.out if args.out is not None else Path(f"{d.name}.spec")

    lines: list[str] = []
    print(f"Processing {len(files)} files.")
    for i,p in enumerate(files,start=1):
        spec = read_spec_exiftool(str(p)).strip()
        if not spec:
            spec = "nospec"

        if args.include_filename:
            lines.append(f"{p.name}\t{spec}")
        else:
            lines.append(spec)
        print(".", end="", flush=True)
        if i%10==0: print(f"{i}",end="",flush=True)
        if i%100==0: print()

    print("\nDone.")

    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {out} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
