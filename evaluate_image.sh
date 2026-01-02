#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <image-file>" >&2
  exit 1
fi

in="$1"

tmp="$(mktemp -t eval_image_XXXXXX).jpg"
cleanup() {
  rm -f "$tmp"
}
trap cleanup EXIT

vips thumbnail "$in" "$tmp" 500

python img_evaluate.py "$tmp"

