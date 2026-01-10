#!/usr/bin/env python3
"""
img2expr2.py  (aka gemini_paint.py)

- Generates a Gemini style-transfer image (JPEG).
- Writes EXIF UserComment (source filename) into the output.
- Copies XMP blob ("xmp-data") from the INPUT IMAGE into the output (if present).
- Also writes a side-by-side "pair" image:
    [resized input NxN] 400px gap [styled NxN]
  with 400px border around everything, background color FDF5E6.
"""

import sys
import os
import io
import re
import json
import mimetypes
import argparse
from pathlib import Path

import pyvips as vips
import subprocess
from google import genai
from google.genai import types


# --------------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------------
BORDER_PX = 400
BG_HEX = "FDF5E6"


# -------------------------------------
# spec -> xml
# -------------------------------------
def _make_xmp_packet(spec: str) -> bytes:
    ns_uri = "https://example.com/lyapunov/1.0/"
    xml = f"""<?xpacket begin='' id='W5M0MpCehiHzreSzNTczkc9d'?>
<x:xmpmeta xmlns:x='adobe:ns:meta/'>
 <rdf:RDF xmlns:rdf='http://www.w3.org/1999/02/22-rdf-syntax-ns#'>
  <rdf:Description xmlns:lyapunov='{ns_uri}'
                   lyapunov:spec='{spec}'/>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end='w'?>"""
    return xml.encode("utf-8")


# -------------------------------------
# EXIF UserComment payload (with encoding prefix)
# -------------------------------------
def _exif_usercomment_payload(s: str) -> bytes:
    # EXIF UserComment requires an 8-byte character code prefix.
    # Prefer ASCII when possible; else use UNICODE (UTF-16LE).
    try:
        b = s.encode("ascii")
        return b"ASCII\0\0\0" + b
    except UnicodeEncodeError:
        b = s.encode("utf-16le")
        return b"UNICODE\0" + b


# -------------------------------------
# embed value into image
# -------------------------------------
def spec2image(base,spec):
    base1 = base.copy()  # ensure writable metadata
    base1.set_type(vips.GValue.blob_type, "xmp-data", _make_xmp_packet(spec))
    base1.set_type(vips.GValue.gstr_type,"exif-ifd0-UserComment",spec)
    return base1


# -------------------------------------
# read value from image
# -------------------------------------
def read_spec_exiftool(path: str) -> str:
    out = subprocess.check_output(
        ["exiftool", "-s3", "-XMP-lyapunov:spec", path],
        text=True,
    )
    if out.strip():
        return out.strip()

    out = subprocess.check_output(
        ["exiftool", "-s3", "-UserComment", path],
        text=True,
    )
    return out.strip()


# --------------------------------------------------------------------------------
# UTILS
# --------------------------------------------------------------------------------
def hex_to_rgb_u8(hex_str: str) -> list[int]:
    s = hex_str.strip().lstrip("#")
    if len(s) != 6:
        raise ValueError(f"Expected 6 hex digits, got: {hex_str!r}")
    return [int(s[i : i + 2], 16) for i in (0, 2, 4)]


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
    Return first free slot by probing g00001..g10000 (no directory scan).
    Also treats gNNNNN_pair as occupying the slot.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(1, 10_001):
        main = output_dir / f"g{i:05d}.jpeg"
        pair = output_dir / f"g{i:05d}_pair.jpeg"
        if (not main.exists()) and (not pair.exists()):
            return main

    raise RuntimeError(f"No free slots in {output_dir.resolve()} (1..10000)")



def summarize_gemini_response(resp, *, max_json_chars: int = 4000) -> str:
    """
    Best-effort diagnostic summary for the "no image data found" case.
    """
    lines: list[str] = []
    lines.append(f"response.text={getattr(resp, 'text', None)!r}")

    cands = getattr(resp, "candidates", None) or []
    lines.append(f"candidates={len(cands)}")

    for ci, cand in enumerate(cands):
        lines.append(f"cand[{ci}].finish_reason={getattr(cand, 'finish_reason', None)!r}")
        lines.append(f"cand[{ci}].safety_ratings={getattr(cand, 'safety_ratings', None)!r}")

        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) or []
        lines.append(f"cand[{ci}].parts={len(parts)}")

        for pi, part in enumerate(parts):
            inline = getattr(part, "inline_data", None)
            ptxt = getattr(part, "text", None)

            if inline is not None:
                data = getattr(inline, "data", None)
                mime = getattr(inline, "mime_type", None)
                dlen = len(data) if isinstance(data, (bytes, bytearray)) else None
                lines.append(f"  part[{pi}]: inline_data mime={mime!r} data_len={dlen}")
            elif ptxt is not None:
                lines.append(f"  part[{pi}]: text_len={len(ptxt)} head={ptxt[:120]!r}")
            else:
                lines.append(f"  part[{pi}]: (no inline_data, no text) type={type(part)}")

    # Optional dump (often useful)
    try:
        d = resp.model_dump()
        js = json.dumps(d, indent=2, default=str)
        lines.append("model_dump(head):")
        lines.append(js[:max_json_chars])
        if len(js) > max_json_chars:
            lines.append(f"... (truncated, {len(js)} chars total)")
    except Exception as e:
        lines.append(f"model_dump failed: {e!r}")

    return "\n".join(lines)


def extract_image_bytes(response, *, debug: bool = False) -> bytes:
    """
    Extract image bytes from a Gemini response. If no image is present, raise
    with useful diagnostics.
    """
    cands = getattr(response, "candidates", None) or []

    # Try candidates->content->parts->inline_data first
    for cand in cands:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            inline = getattr(part, "inline_data", None)
            if inline is not None:
                data = getattr(inline, "data", None)
                if data:
                    return data

            # Fallback: as_image()
            as_image = getattr(part, "as_image", None)
            if callable(as_image):
                img = as_image()
                if img is not None:
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG")
                    return buf.getvalue()

    # Nothing found: give concrete feedback
    msg = f"No image data found in response. candidates={len(cands)} text={getattr(response, 'text', None)!r}"

    # Add quick finish_reason/parts counts even without --debug
    try:
        if cands:
            fr = getattr(cands[0], "finish_reason", None)
            content = getattr(cands[0], "content", None)
            parts = getattr(content, "parts", None) or []
            msg += f" finish_reason={fr!r} parts={len(parts)}"
    except Exception:
        pass

    if debug:
        msg += "\n\n" + summarize_gemini_response(response)

    raise ValueError(msg)


def to_rgb8(im: vips.Image, bg_rgb: list[int]) -> vips.Image:
    # Convert to sRGB, remove alpha (if any), force 3-band uchar.
    try:
        if im.interpretation != "srgb":
            im = im.colourspace("srgb")
    except Exception:
        im = im.copy(interpretation="srgb")

    if im.hasalpha():
        im = im.flatten(background=bg_rgb)

    # Ensure 3 bands
    if im.bands == 1:
        im = im.bandjoin([im, im])
    elif im.bands > 3:
        im = im.extract_band(0, n=3)

    im = im.cast("uchar").copy(interpretation="srgb")
    return im


def resize_center_crop_square(im: vips.Image, size: int) -> vips.Image:
    # Resize (up or down) preserving aspect, then center-crop to size x size.
    w, h = im.width, im.height
    if w <= 0 or h <= 0:
        raise ValueError(f"Bad image dimensions: {w}x{h}")

    scale = max(size / w, size / h)
    im2 = im.resize(scale, kernel="lanczos3")

    left = max(0, (im2.width - size) // 2)
    top = max(0, (im2.height - size) // 2)
    return im2.crop(left, top, size, size)


def save_jpeg_exclusive(path: Path, im: vips.Image, *, Q: int = 95) -> None:
    data = im.jpegsave_buffer(Q=Q, strip=False)
    with open(path, "xb") as f:
        f.write(data)


def make_side_by_side(
    orig_im: vips.Image,
    styled_im: vips.Image,
    *,
    size: int,
    border: int,
    bg_rgb: list[int],
) -> vips.Image:
    # Prepare tiles
    o = resize_center_crop_square(orig_im, size)
    o = to_rgb8(o, bg_rgb)

    s = resize_center_crop_square(styled_im, size)
    s = to_rgb8(s, bg_rgb)

    total_w = border + size + border + size + border
    total_h = border + size + border

    canvas = vips.Image.black(total_w, total_h).new_from_image(bg_rgb)
    canvas = canvas.cast("uchar").copy(interpretation="srgb")

    x0, y0 = border, border
    x1 = border + size + border

    canvas = canvas.insert(o, x0, y0)
    canvas = canvas.insert(s, x1, y0)
    return canvas


# --------------------------------------------------------------------------------
# CORE GENERATION LOGIC
# --------------------------------------------------------------------------------

def make_inference_jpeg_bytes_square(src_path: Path, *, side: int = 2048, Q: int = 85) -> tuple[bytes, str]:
    im = vips.Image.new_from_file(str(src_path), access="random", autorotate=True)

    # force RGB8 sRGB
    try:
        if im.interpretation != "srgb":
            im = im.colourspace("srgb")
    except Exception:
        im = im.copy(interpretation="srgb")

    if im.hasalpha():
        im = im.flatten(background=[255, 255, 255])

    if im.bands == 1:
        im = im.bandjoin([im, im])
    elif im.bands > 3:
        im = im.extract_band(0, n=3)

    im = im.cast("uchar").copy(interpretation="srgb")

    # square downsample
    if im.width != side:
        scale = side / im.width
        im = im.resize(scale, kernel="lanczos3")

    # encode smaller payload; strip metadata
    data = im.jpegsave_buffer(Q=Q, strip=True)
    return data, "image/jpeg"


def generate_image_bytes(
    input_path: Path,
    resolution: str,
    aspect_ratio: str,
    temperature: float,
    *,
    debug: bool = False,
    data_size: int = 1024,
) -> bytes:
    client = genai.Client(api_key=get_api_key())

    #input_bytes = input_path.read_bytes()
    #mime_type = guess_mime_type(input_path)

    input_bytes, mime_type = make_inference_jpeg_bytes_square(input_path, side=data_size, Q=85)
    print(f"inference payload: {len(input_bytes)} bytes (side=2048 Q=85)")

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
    print(
        f"Input: {input_path.name} | bytes={len(input_bytes)} | MIME: {mime_type} | "
        f"Res: {resolution} | AR: {aspect_ratio} | Temp: {temperature}"
    )

    response = client.models.generate_content(
        model="gemini-3-pro-image-preview",
        contents=[
            prompt_text,
            types.Part.from_bytes(data=input_bytes, mime_type=mime_type),
        ],
        config=types.GenerateContentConfig(
            temperature=temperature,
            # Explicitly ask for an image back
            response_modalities=["IMAGE"],
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

    return extract_image_bytes(response, debug=debug)


# --------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------
def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Generate painting-style art with Gemini image model")
    parser.add_argument("input_file", type=Path, help="Path to the input source image.")
    parser.add_argument("--res", choices=["1K", "2K", "4K"], default="4K")
    parser.add_argument("--data", type=int, default=1024)
    parser.add_argument("--ar", type=str, default="1:1")
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--outdir", type=Path, default=Path("gemini"))
    parser.add_argument("--no-pair", action="store_true", help="Do not write the side-by-side pair image.")
    parser.add_argument("--debug", action="store_true", help="Print verbose Gemini response diagnostics on failure.")

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

    size = {"1K": 1024, "2K": 2048, "4K": 4096}[args.res]
    bg_rgb = hex_to_rgb_u8(BG_HEX)

    try:
        # Load original once (for metadata + for the pair image)
        print(f"reading file: {args.input_file}")
        orig = vips.Image.new_from_file(str(args.input_file), access="random", autorotate=True)
        source_fn = str(args.input_file)
        print(f"source_fn: {source_fn}")

        # Generate styled bytes, decode to vips
        print("calling gemini.")
        styled_bytes = generate_image_bytes(
            args.input_file,
            args.res,
            args.ar,
            min(max(args.temp, 0.0), 1.0),
            debug=args.debug,
            data_size=args.data,
        )
        print("gemini call completed.")
        styled = vips.Image.new_from_buffer(styled_bytes, "", access="random")
        styled_out = spec2image(to_rgb8(styled, bg_rgb), source_fn)

        # Pick output name
        save_path = get_next_output_path(args.outdir)

        # Write styled output (exclusive)
        save_jpeg_exclusive(save_path, styled_out, Q=95)
        print(f"Success! Styled image saved to: {save_path}")
        print(f"Check UserData: {read_spec_exiftool(save_path)}")

        # Write side-by-side pair
        if not args.no_pair:
            pair = make_side_by_side(orig, styled, size=size, border=BORDER_PX, bg_rgb=bg_rgb)
            pair = spec2image(pair, source_fn)

            pair_path = save_path.with_name(save_path.stem + "_pair.jpeg")
            save_jpeg_exclusive(pair_path, pair, Q=95)
            print(f"Pair image saved to: {pair_path}")

        return 0

    except Exception as e:
        print(f"Generation failed: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

