
from diffusers import StableDiffusionInstructPix2PixPipeline
from PIL import Image
import torch
import os
import uuid

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the instruct-pix2pix pipeline
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    "timbrooks/instruct-pix2pix",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    safety_checker=None  # Optional: disables NSFW filter
).to(device)

def apply_style_filter(input_image: Image.Image, style: str) -> str:
    """
    Applies the selected style to the input image using instruct-pix2pix.

    Args:
        input_image (PIL.Image): Original uploaded image.
        style (str): Style name like 'anime', 'sketch', etc.

    Returns:
        str: Path to the saved output image.
    """

    # Map styles to prompts
    prompt_map = {
        "anime": "turn this photo into anime style",
        "sketch": "make this image a pencil sketch drawing",
        "cartoon": "make this look like a cartoon character",
        "painting": "convert this into a painting",
        "pixar": "convert this face to pixar movie style"
    }

    prompt = prompt_map.get(style.lower(), f"turn this into {style} style")
    
    # Resize and convert image
    image = input_image.convert("RGB").resize((512, 512))

    # Run image-to-image inference
    result = pipe(prompt=prompt, image=image, num_inference_steps=30).images[0]

    # Save result
    # os.makedirs("outputs", exist_ok=True)
    # output_filename = f"outputs/{style}_{uuid.uuid4().hex[:8]}.png"
    # result.save(output_filename)

    return result
