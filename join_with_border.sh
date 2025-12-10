#!/usr/bin/env bash
# join_with_border.sh
#
# Usage:
#   ./join_with_border.sh "pattern" target.jpg shim_px across BG_HEX
#
# Example:
#   ./join_with_border.sh "test/tst18_*.jpg" nn14loc49.jpg 800 4 FFADFF

set -euo pipefail

if [ "$#" -ne 5 ]; then
    echo "Usage: $0 \"pattern\" target.jpg shim across BG_HEX" >&2
    exit 1
fi

pattern="$1"   # e.g. "test/tst18_*.jpg"
target="$2"    # e.g. nn14loc49.jpg
shim="$3"      # e.g. 800
across="$4"    # e.g. 4
bg_hex="$5"    # e.g. FFADFF

# Convert hex RR GG BB → "R G B"
R=$((16#${bg_hex:0:2}))
G=$((16#${bg_hex:2:2}))
B=$((16#${bg_hex:4:2}))
bg_rgb="$R $G $B"

# Derive a _raw name from target, keeping extension
ext="${target##*.}"
base="${target%.*}"
raw="${base}_raw.${ext}"

echo "Joining images matching: $pattern"
echo "Intermediate (raw) image: $raw"
echo "Final image: $target"
echo "Inner shim: ${shim}px"
echo "Outer border: ${shim}px"
echo "Background RGB: $bg_rgb"

# 1) Join images with inner shim and background color
vips arrayjoin "$(ls $pattern)" "$raw" \
    --across "$across" \
    --shim "$shim" \
    --background "$bg_rgb"

# 2) Get width/height of raw image
W=$(vipsheader -f width "$raw")
H=$(vipsheader -f height "$raw")

border="$shim"
OUTW=$((W + 2 * border))
OUTH=$((H + 2 * border))

echo "Raw size: ${W}x${H}"
echo "Output size: ${OUTW}x${OUTH}"

# 3) Embed with same background color for outer border
vips embed "$raw" "$target" \
    "$border" "$border" "$OUTW" "$OUTH" \
    --extend background \
    --background "$bg_rgb"

echo "Done: $target"

