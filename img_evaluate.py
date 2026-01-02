#!/usr/bin/env python
from __future__ import annotations

import os
import sys
from pathlib import Path

parent = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent))

import pyvips  # pip install pyvips (requires libvips installed)
from google import genai
from google.genai import types

def read_api_key(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8").strip()

PROMPT_TEXT = """TASK
Evaluate the provided image for aesthetic value.

DEFINITION OF AESTHETIC VALUE
Aesthetic value measures visual interest arising from structure, complexity,
and novelty. It does NOT measure beauty, taste, emotional appeal, symbolism,
or subject matter.

PROCESS
1. Examine the global structure of the image:
   - coherence, symmetry, organization, large-scale forms
2. Examine local detail:
   - fine structure, variation, multiscale detail vs uniformity
3. Assess novelty:
   - departure from pure randomness or trivial geometry
4. Ignore semantics, meaning, labels, and emotional interpretation.

SCORING (0–10)
Assign a single integer score using these anchors:
0  — Visually trivial or unstructured
      Examples: white noise, flat color, uniform texture, black image
1–2 — Minimal structure
      Examples: weak gradients, slight variation, mostly noise, mostly black
3–4 — Simple, low-complexity structure
      Examples: basic geometric shapes, clean lines, little multiscale detail
5–6 — Moderately complex
      Examples: clear organization with some internal variation
7–8 — High complexity and structure
      Examples: rich detail across multiple scales, nontrivial patterns
9–10 — Very high aesthetic complexity
      Examples: fractal-like, self-similar, or mathematically rich structure
      with strong global coherence

OUTPUT FORMAT
Return ONLY the following line, with no explanation:
score: <integer 0–10>
"""


def make_thumb_jpeg_bytes(image_path: str, size: int = 500, quality: int = 85) -> bytes:
    """
    Make a size×size thumbnail (center-cropped) and return JPEG bytes.
    """
    # thumbnail() does decode + shrink efficiently
    im = pyvips.Image.thumbnail(
        image_path,
        size,
        height=size,
        crop="centre",  # "cover" + center crop to exact size×size
    )

    # Ensure 3-band for JPEG; flatten alpha if present
    if im.hasalpha():
        im = im.flatten(background=[0, 0, 0])
    if im.bands > 3:
        im = im[:3]
    elif im.bands == 1:
        im = im.bandjoin([im, im])  # -> 3 bands total

    # Encode to JPEG in-memory
    # (Vips option string syntax)
    buf = im.write_to_buffer(f".jpg[Q={quality},strip]")
    return buf


def evaluate_image(image_path: str) -> str:
    client = genai.Client(api_key=read_api_key("gemini_key.txt"))

    thumb_jpeg = make_thumb_jpeg_bytes(image_path, size=500, quality=85)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            PROMPT_TEXT,
            types.Part.from_bytes(data=thumb_jpeg, mime_type="image/jpeg"),
        ],
        config=types.GenerateContentConfig(
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_NONE",
                ),
            ]
        ),
    )
    return (response.text or "").strip()


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python img_evaluate.py <path_to_image>", file=sys.stderr)
        return 2

    image_path = sys.argv[1]
    if not os.path.exists(image_path):
        print(f"img_evaluate.py error: File '{image_path}' not found.", file=sys.stderr)
        return 2

    print(evaluate_image(image_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


