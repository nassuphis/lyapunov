#!/usr/bin/env bash
# seq 254 262 | xargs -I{} bash -c 'f={}; ./upscale.sh renderings/output${f}.png upscale/output${f}.png'
# Exit on error
set -e

# Usage message
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <input_image_path> <output_upscale_path>"
  exit 1
fi

INPUT_PATH="$1"
OUTPUT_PATH="$2"

# Check that RECRAFT token is set
if [ -z "$RECRAFT" ]; then
  echo "Error: RECRAFT token is not set. Run: export RECRAFT=your_token"
  exit 1
fi

# Check if jq is installed
if ! command -v jq >/dev/null 2>&1; then
  echo "Error: jq is not installed. You can install it with: brew install jq"
  exit 1
fi

# Call the vectorize API
echo "Uploading $INPUT_PATH to Recraft for upscaling..."
RESPONSE=$(curl -s -X POST https://external.api.recraft.ai/v1/images/crispUpscale \
  -H "Content-Type: multipart/form-data" \
  -H "Authorization: Bearer $RECRAFT" \
  -F "file=@${INPUT_PATH}")

# Extract the SVG URL using jq
URL=$(echo "$RESPONSE" | jq -r '.image.url')

if [ "$URL" == "null" ] || [ -z "$URL" ]; then
  echo "Error: Failed to extract URL from response:"
  echo "$RESPONSE"
  exit 1
fi

# Use a temp file to store the raw download
TMPFILE=$(mktemp -t recraft-upscale-XXXXXXXX.png)

# Download the SVG
echo "Downloading upscaled image to $TMPFILE..."
curl -s -L -o "$TMPFILE" "$URL"

# Re-encode to a clean PNG using vips
echo "Converting to clean PNG with vips..."
vips copy "$TMPFILE" "$OUTPUT_PATH"

# Remove the temp file
rm -f "$TMPFILE"

echo "✅ Done: upscaled image saved to $OUTPUT_PATH"


