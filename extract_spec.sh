#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "usage: $0 <image-file> [height-percent]" >&2
  exit 1
fi

in="$1"
height="${2:-10}"   # default = 10%

dir="$(dirname "$in")"
base="$(basename "$in")"
name="${base%.*}"
out="ocr_${name}.png"

magick "$in" \
  -gravity south -crop "100%x${height}%+0+0" +repage \
  -despeckle \
  -resize 300% -filter point \
  -colorspace sRGB \
  -fuzz 10% -fill white -opaque white \
  -fill black +opaque white \
  -alpha off \
  -statistic Median 3x3 -negate -threshold 99% \
  -units PixelsPerInch -density 300 \
  "$out"

python ocr.py "$out"


