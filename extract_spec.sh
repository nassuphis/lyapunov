#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <image-file>" >&2
  exit 1
fi

in="$1"
#tmp="$(mktemp -t ocr_ready_XXXXXX.png)"
tmp="ocr_ready.png"

magick "$in" \
  -gravity south -crop 100%x20%+0+0 +repage \
  -despeckle \
  -resize 300% -filter point \
  -colorspace sRGB \
  -fuzz 1% -fill white -opaque white \
  -fill black +opaque white \
  -alpha off \
  -statistic Median 3x3 -negate -threshold 99% \
  -units PixelsPerInch -density 300 \
  "$tmp"

exit 0

# OCR and normalize to one line
tesseract ocr_ready.png stdout --oem 1 --psm 6 --dpi 300 \
  -c tessedit_char_whitelist="_abcdefghijklmnopqrstuvwxyz0123456789:,.=()-+*AB" \
  -c load_system_dawg=0 -c load_freq_dawg=0 -c load_punc_dawg=0 -c load_number_dawg=0 \
  -c user_words_file=spec.words \
  -c user_patterns_file=spec.patterns \
| tr '\n' ' ' \
| sed -E 's/[[:space:]]+/ /g; s/^ //; s/ $//' \
| tr ' ' ',' ; printf '\n'
#| sed 's/(1)/(l)/g' \
#| sed 's/(l1)/(l)/g' \
#| sed 's/(1l)/(l)/g' \
#| sed 's/(zj)/(x)/g' \
#| sed 's/m:10(/m:i0(/g' \
#| sed 's/m:chul(/m:chu1(/g' \
#| sed 's/m:chti(/m:cht1(/g' \
#| sed 's/m:11(/m:i1(/g' \
#| sed 's/\*41(/\*j1(/g' \
#| sed 's/\*chul(/\*chu1(/g' \
#| sed 's/\*eht2(/\*cht2(/g' \
#| sed 's/(chul(/(chu1(/g' \
#| sed 's/\+17)/\+1j/g' \
#| sed 's/\+145\*/\+1j\*/g' \
#| sed 's/\+li\*/\+1j\*/g' \
#| sed 's/71(/j1(/g' \
#| sed 's/,rgo/\,rgb/g' \


#rm -f "$tmp"

