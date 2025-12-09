#!/usr/bin/env bash
set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 image.jpg" >&2
    exit 1
fi

JPG="$1"
BASE="${JPG%.*}"
TARGET_DZI="s3://polynomiography/${BASE}.dzi"

# s5cmd ls returns 0 if it finds the object, non-zero otherwise.
if s5cmd ls "$TARGET_DZI" >/dev/null 2>&1; then
    echo "DeepZoom already exists on S3. Skipping: $TARGET_DZI"
    exit 0
fi

echo "No existing .dzi found. Proceeding."

echo "Fetching $JPG from polyvec/lyapunov."
s5cmd cp "s3://polyvec/lyapunov/$BASE.jpg" .

echo "Creating DeepZoom from $JPG → $BASE.dzi ..."
vips dzsave "$JPG" "$BASE"

echo "Uploading to s3://polynomiography ..."
s5cmd cp "$BASE.dzi" s3://polynomiography
s5cmd cp "${BASE}_files" s3://polynomiography

echo "Cleaning up local files ..."
rm -f "$JPG"
rm -f "$BASE.dzi"
rm -rf "${BASE}_files"

echo "Done."

