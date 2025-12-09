#!/usr/bin/env bash
# join_with_border.sh
#
# Usage:
#   ./join_with_border.sh "test/tst18_*.jpg" nn14loc49.jpg 800 16

set -euo pipefail

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 \"pattern\" target.jpg shim_px across" >&2
    exit 1
fi

pattern="$1"   # e.g. "test/tst18_*.jpg"
target="$2"    # e.g. nn14loc49.jpg
shim="$3"      # e.g. 800
across="$4"    # e.g 4

# Derive a _raw name from target, keeping extension
ext="${target##*.}"
base="${target%.*}"
raw="${base}_raw.${ext}"

echo "Joining images matching: $pattern"
echo "Intermediate (raw) image: $raw"
echo "Final image: $target"
echo "Inner shim: ${shim}px, outer border: ${shim}px"

# 1) Join images with inner shim
#    IMPORTANT: pass the *single* argument "$(ls ...)" like you do manually
vips arrayjoin "$(ls $pattern)" "$raw" --across "$across" --shim "$shim"

# 2) Get width/height of raw image
W=$(vipsheader -f width "$raw")
H=$(vipsheader -f height "$raw")

# 3) Compute output size with border = shim on all sides
border="$shim"
OUTW=$((W + 2 * border))
OUTH=$((H + 2 * border))

echo "Raw size: ${W}x${H}"
echo "Output size: ${OUTW}x${OUTH}"

# 4) Embed with black border
vips embed "$raw" "$target" \
    "$border" "$border" "$OUTW" "$OUTH" \
    --extend background --background "0 0 0"

echo "Done: $target"
