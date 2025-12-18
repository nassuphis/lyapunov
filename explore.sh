#!/usr/bin/env bash
set -euo pipefail

# defaults
KILL=1
TOP=1
PIX=2000

usage() {
    echo "usage: $0 <run_dir> <slot_count> [--kill N] [--top N] [--pix N]"
    exit 1
}

# --- parse args ---
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --kill)
            KILL="$2"
            shift 2
            ;;
        --top)
            TOP="$2"
            shift 2
            ;;
        --pix)
            PIX="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

set -- "${POSITIONAL[@]}"

if [ $# -ne 2 ]; then
    usage
fi

RUN_DIR="$1"
SLOTS="$2"

OUT_DIR="$RUN_DIR"
OUT_JPG="$RUN_DIR/$(basename "$RUN_DIR").jpg"
MACROS="$RUN_DIR/used_macros.txt"

# --- preflight ---
if [ ! -f "$MACROS" ]; then
    echo "ERROR: macro file not found:"
    echo "  $MACROS"
    echo
    echo "Run lyapunov_cli.py once to generate used_macros.txt first."
    exit 1
fi

echo "=== explore ==="
echo "dir      : $RUN_DIR"
echo "slots    : $SLOTS"
echo "kill     : $KILL"
echo "top      : $TOP"
echo "macros   : $MACROS"
echo

while true; do
    echo "--- render pass ---"
    python lyapunov_cli.py '@RUN' \
        --pix "$PIX" \
        --out "$OUT_JPG" \
        --macro "$MACROS" \
        --macro-add "@SLOTS0=$SLOTS"

    echo "--- prune pass ---"
    python delete_boring.py "$OUT_DIR" --hot
    python rank_boring.py "$OUT_DIR" --hot --kill "$KILL"
    python dedupe_top.py "$OUT_DIR" --hot --top "$TOP"

    n_files=$(ls "$OUT_DIR"/*.jpg 2>/dev/null | wc -l | tr -d ' ')
    echo "files: $n_files / $SLOTS"

    if [ "$n_files" -ge "$SLOTS" ]; then
        echo "=== stable: no more boring images ==="
        break
    fi

    echo "=== continuing exploration ==="
    echo
done