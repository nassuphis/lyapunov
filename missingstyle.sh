#!/usr/bin/env bash
set -euo pipefail

t1="$(mktemp)"
t2="$(mktemp)"
trap 'rm -f "$t1" "$t2"' EXIT

ls -1 tst/*.jpg | sort > "$t1"
for i in gemini2/g*_pair.jpeg; do
  exiftool -s3 -UserComment "$i"
done | sed '/^$/d' | sort > "$t2"

comm -23 "$t1" "$t2"
