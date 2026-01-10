import sys
import argparse
import re
from pathlib import Path
from google import genai
from google.genai import types
import io

# --------------------------------------------------------------------------------
# UTILS
# --------------------------------------------------------------------------------

def read_api_key(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8").strip()

def get_next_output_path(output_dir: Path) -> Path:
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / "g01.jpeg"

    pattern = re.compile(r"^g(\d+)\.jpeg$", re.IGNORECASE)
    max_index = 0
    
    for file in output_dir.iterdir():
        if file.is_file():
            match = pattern.match(file.name)
            if match:
                index = int(match.group(1))
                if index > max_index:
                    max_index = index
    
    return output_dir / f"g{max_index + 1:02d}.jpeg"

# --------------------------------------------------------------------------------
# CORE GENERATION LOGIC
# --------------------------------------------------------------------------------

def generate_image(input_path: Path, resolution: str, aspect_ratio: str, temperature: float):
    client = genai.Client(api_key=read_api_key("gemini_key.txt"))

    with open(input_path, "rb") as f:
        input_image_bytes = f.read()

    prompt_text = (
        "remove the letters at the bottom. "
        "make an abstract expressionist painting based on this composition. "
        "use the most appropriate brush strokes, but preserve geometry. "
        "the composition must be identical, not approximate. "
        "make sure the painting can be executed without fine brushwork"
    )

    print(f"--- Calling gemini-3-pro-image-preview ---")
    print(f"Input: {input_path.name} | Res: {resolution} | AR: {aspect_ratio} | Temp: {temperature}")

    # Correct Usage: Nesting ImageConfig inside GenerateContentConfig
    response = client.models.generate_content(
        model="gemini-3-pro-image-preview",
        contents=[
            prompt_text,
            types.Part.from_bytes(data=input_image_bytes, mime_type="image/jpeg")
        ],
        config=types.GenerateContentConfig(
            temperature=temperature,
            image_config=types.ImageConfig(
                aspect_ratio=aspect_ratio,
                image_size=resolution  # Valid values: "1K", "2K", "4K"
            ),
            safety_settings=[
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
            ]
        )
    )

    if response.candidates:
        for part in response.candidates[0].content.parts:
            # Check for inline raw bytes
            if part.inline_data:
                return part.inline_data.data
            
            # Check for standard image object handling (as per your snippet)
            # The SDK wraps this differently in some versions, but this handles the 'as_image' pattern
            if hasattr(part, 'as_image'):
                img = part.as_image()
                buf = io.BytesIO()
                img.save(buf, format='JPEG')
                return buf.getvalue()

    raise ValueError("No image data found in response.")

# --------------------------------------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Art with Gemini 3 Pro Image")
    
    parser.add_argument("input_file", type=Path, help="Path to the input source image.")
    # Defaulting to 4K as requested. The API expects uppercase "4K".
    parser.add_argument("--res", type=str, default="4K", help="Output resolution: '1K', '2K', or '4K'.")
    parser.add_argument("--ar", type=str, default="1:1", help="Aspect Ratio (e.g., '1:1', '16:9').")
    parser.add_argument("--temp", type=float, default=0.7, help="Temperature (default: 0.7).")

    args = parser.parse_args()

    if not args.input_file.exists():
        print(f"Error: File '{args.input_file}' not found.")
        sys.exit(1)

    # Ensure resolution is uppercase for the API (e.g. "4k" -> "4K")
    clean_res = args.res.upper()

    try:
        image_bytes = generate_image(args.input_file, clean_res, args.ar, args.temp)
        
        output_dir = Path("gemini")
        save_path = get_next_output_path(output_dir)
        
        with open(save_path, "wb") as f:
            f.write(image_bytes)
            
        print(f"Success! Image saved to: {save_path}")

    except Exception as e:
        print(f"Generation failed: {e}")
        