#!/usr/bin/env python
"""
gemini_paint.py

Same as before, but output files are g1.jpeg, g2.jpeg, ... (no leading zeroes).
Also writes the INPUT filename into EXIF UserComment (tag 37510) of the output JPEG.
"""

import sys
import os
import io
import re
import mimetypes
import argparse
from pathlib import Path

from google import genai
from google.genai import types

from PIL import Image  # <-- added


# --------------------------------------------------------------------------------
# UTILS
# --------------------------------------------------------------------------------

USERCOMMENT = 37510  # EXIF tag 0x9286


def read_api_key(path: str | Path) -> str:
    p = Path(path)
    try:
        key = p.read_text(encoding="utf-8").strip()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"API key file not found: {p.resolve()}") from e
    if not key:
        raise ValueError(f"API key file is empty: {p.resolve()}")
    return key


def get_api_key() -> str:
    env_key = os.getenv("GEMINI_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()

    key_path = Path(__file__).resolve().with_name("gemini_key.txt")
    return read_api_key(key_path)


def guess_mime_type(path: Path) -> str:
    mt = mimetypes.guess_type(str(path))[0]
    return mt or "application/octet-stream"


def get_next_output_path(output_dir: Path) -> Path:
    """
    Return next available gemini/gN.jpeg path (no leading zeroes).
    Caller should open with "xb" to avoid overwrites.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, 1_000_000):
        p = output_dir / f"g{i}.jpeg"
        if not p.exists():
            return p
    raise RuntimeError(f"Too many outputs in: {output_dir.resolve()}")


def extract_image_bytes(response) -> bytes:
    cands = getattr(response, "candidates", None) or []
    for cand in cands:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            inline = getattr(part, "inline_data", None)
            if inline is not None:
                data = getattr(inline, "data", None)
                if data:
                    return data

            as_image = getattr(part, "as_image", None)
            if callable(as_image):
                img = as_image()
                buf = io.BytesIO()
                img.save(buf, format="JPEG")
                return buf.getvalue()

    txt = getattr(response, "text", None)
    raise ValueError(f"No image data found in response. candidates={len(cands)} text={txt!r}")


def exif_user_comment_bytes(s: str) -> bytes:
    # EXIF UserComment wants an 8-byte encoding prefix.
    try:
        b = s.encode("ascii")
        return b"ASCII\0\0\0" + b
    except UnicodeEncodeError:
        b = s.encode("utf-16le")
        # Commonly understood: UNICODE prefix + BOM + UTF-16LE payload
        return b"UNICODE\0" + b"\xff\xfe" + b


def inject_usercomment_into_jpeg(jpeg_bytes: bytes, comment: str) -> bytes:
    """
    Adds/overwrites EXIF UserComment (37510) in a JPEG byte stream via Pillow.
    Note: this re-encodes the JPEG once.
    """
    im = Image.open(io.BytesIO(jpeg_bytes))
    exif = im.getexif()
    exif[USERCOMMENT] = exif_user_comment_bytes(comment)

    out = io.BytesIO()
    im.save(out, format="JPEG", exif=exif.tobytes(), quality=95)
    return out.getvalue()


# --------------------------------------------------------------------------------
# CORE GENERATION LOGIC
# --------------------------------------------------------------------------------

def generate_image(input_path: Path, resolution: str, aspect_ratio: str, temperature: float) -> bytes:
    client = genai.Client(api_key=get_api_key())

    input_bytes = input_path.read_bytes()
    mime_type = guess_mime_type(input_path)

    prompt_text = (
        "Remove the letters at the bottom. "
        "Make an abstract expressionist oil painting based on this composition. "
        "Use appropriate brush strokes, but preserve the geometry. "
        "The composition must be identical, not approximate. "
        "Try to map every visual element of the input image to a visual element in the output imege. "
        "The output image should contain no visual elements that do not correspond to something in the input image. "
        "No text, no signatures, no borders, no cropping. "
        "Make sure the painting can be executed without fine brushwork."
    )

    print("--- Calling gemini-3-pro-image-preview ---")
    print(f"Input: {input_path.name} | MIME: {mime_type} | Res: {resolution} | AR: {aspect_ratio} | Temp: {temperature}")

    response = client.models.generate_content(
        model="gemini-3-pro-image-preview",
        contents=[
            prompt_text,
            types.Part.from_bytes(data=input_bytes, mime_type=mime_type),
        ],
        config=types.GenerateContentConfig(
            temperature=temperature,
            image_config=types.ImageConfig(
                aspect_ratio=aspect_ratio,
                image_size=resolution,
            ),
            safety_settings=[
                types.SafetySetting(
                    category="HARM_CATEGORY_DANGEROUS_CONTENT",
                    threshold="BLOCK_NONE",
                ),
            ],
        ),
    )

    jpeg_bytes = extract_image_bytes(response)

    # --- NEW: write input filename into EXIF UserComment ---
    # Use input_path.name (just the filename). If you want full path, use str(input_path.resolve()).
    jpeg_bytes = inject_usercomment_into_jpeg(jpeg_bytes, str(input_path))

    return jpeg_bytes


# --------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Generate painting-style art with Gemini image model")
    parser.add_argument("input_file", type=Path, help="Path to the input source image.")
    parser.add_argument("--res", choices=["1K", "2K", "4K"], default="4K")
    parser.add_argument("--ar",choices = ["1:1","2:3","3:2","3:4","4:3","4:5","5:4","9:16","16:9","21:9"], type=str, default="1:1")
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--outdir", type=Path, default=Path("gemini"))

    args = parser.parse_args(argv)

    if not args.input_file.exists():
        print(f"Error: file not found: {args.input_file}", file=sys.stderr)
        return 1
    if not args.input_file.is_file():
        print(f"Error: not a file: {args.input_file}", file=sys.stderr)
        return 1

    if not re.match(r"^\d+:\d+$", args.ar):
        print("Error: --ar must look like 'W:H' (e.g. 1:1, 16:9)", file=sys.stderr)
        return 1

    if not (0.0 <= args.temp <= 2.0):
        print("Error: --temp out of range (expected 0..2)", file=sys.stderr)
        return 1

    try:
        image_bytes = generate_image(args.input_file, args.res, args.ar, min(max(args.temp,0.0),1.0))
        save_path = get_next_output_path(args.outdir)

        with open(save_path, "xb") as f:
            f.write(image_bytes)

        print(f"Success! Image saved to: {save_path}")
        return 0

    except Exception as e:
        print(f"Generation failed: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
