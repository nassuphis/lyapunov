#!/usr/bin/env bash
# autolevels.sh
#
# ImageMagick auto-level preset:
#   -contrast-stretch 0.15%x0.15%
#   -sigmoidal-contrast 5x50%
#   -modulate 100,108,100
#
# Usage:
#   ./autolevels.sh input.jpg [output.jpg]
# If output is omitted, writes: <stem>_autolvl.<ext>

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 input.jpg [output.jpg]" >&2
  exit 2
fi

in="$1"
if [[ ! -f "$in" ]]; then
  echo "Input not found: $in" >&2
  exit 1
fi

if [[ $# -eq 2 ]]; then
  out="$2"
else
  dir="$(dirname "$in")"
  base="$(basename "$in")"
  stem="${base%.*}"
  ext="${base##*.}"
  out="${dir}/${stem}_autolvl.${ext}"
fi

magick "$in" \
  -contrast-stretch 0.15%x0.15% \
  -sigmoidal-contrast 5x50% \
  -modulate 100,108,100 \
  "$out"

echo "Wrote: $out"
