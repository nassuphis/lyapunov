#!/usr/bin/env bash
# autolevels_dir.sh
#
# Apply autolevels.sh to all JPGs in a directory.
#
# Usage:
#   ./autolevels_dir.sh nn14loc67
#
# Output:
#   nn14loc67_autolvl/nn14loc67_00001_autolvl.jpg
#   ...

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 input_directory" >&2
  exit 2
fi

indir="$1"
if [[ ! -d "$indir" ]]; then
  echo "Not a directory: $indir" >&2
  exit 1
fi

outdir="${indir}_autolvl"
mkdir -p "$outdir"

shopt -s nullglob

count=0
for inpath in "$indir"/*.jpg; do
  base="$(basename "$inpath")"
  stem="${base%.*}"
  ext="${base##*.}"
  outpath="${outdir}/${stem}_autolvl.${ext}"

  ./autolevels.sh "$inpath" "$outpath"
  count=$((count + 1))
done

echo "Processed $count file(s)"


