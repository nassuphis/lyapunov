#!/usr/bin/env bash
set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 image.png" >&2
    exit 1
fi

PNG="$1"
BASE="${PNG%.*}"

echo "Fetching $PNG from polyvec/lyapunov."
s5cmd cp "s3://polyvec/lyapunov/$BASE.png" .

echo "Creating DeepZoom from $PNG → $BASE.dzi ..."
vips dzsave "$PNG" "$BASE"

echo "Uploading to s3://polynomiography ..."
s5cmd cp "$BASE.dzi" s3://polynomiography
s5cmd cp "${BASE}_files" s3://polynomiography

echo "Cleaning up local files ..."
rm -f "$PNG"
rm -f "$BASE.dzi"
rm -rf "${BASE}_files"

echo "Done."

