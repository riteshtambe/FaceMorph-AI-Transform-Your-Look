# # from diffusers import KandinskyImg2ImgPipeline, KandinskyPriorPipeline
# # from transformers import CLIPTokenizer, CLIPTextModelWithProjection
# # from PIL import Image
# # import torch
# # import os

# # # Download and load Kandinsky model components
# # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # # 1. Prior Pipeline (text -> latent)
# # prior = KandinskyPriorPipeline.from_pretrained(
# #     "kandinsky-community/kandinsky-2-2-prior",
# #     torch_dtype=torch.float32
# # ).to(device)

# # # 2. Img2Img pipeline
# # pipe = KandinskyImg2ImgPipeline.from_pretrained(
# #     "kandinsky-community/kandinsky-2-2",
# #     torch_dtype=torch.float32
# # ).to(device)

# # def apply_style_filter(input_image: Image.Image, style: str) -> Image.Image:
# #     prompt_map = {
# #         "anime": "anime-style portrait of a person",
# #         "sketch": "pencil sketch of a face",
# #         "cartoon": "cartoon character portrait",
# #         "painting": "artistic oil painting of a person",
# #         "pixar": "Pixar animation style face of a young person"
# #     }

# #     prompt = prompt_map.get(style.lower(), "portrait of a face")

# #     input_image = input_image.resize((512, 512)).convert("RGB")

# #     # Step 1: Encode the prompt into latent embeddings
# #     prior_output = prior(prompt, guidance_scale=1.0)
# #     image_embeds = prior_output.image_embeds

# #     # Step 2: Generate the stylized image using the img2img pipeline
# #     result = pipe(
# #         image=input_image,
# #         image_embeds=image_embeds,
# #         strength=0.7,
# #         guidance_scale=4.0
# #     ).images[0]

# #     return result
# from diffusers import StableDiffusionImg2ImgPipeline
# from PIL import Image
# import torch
# import os

# # Load a public img2img model from Hugging Face
# pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
#     "nitrosocke/Arcane-Diffusion",
#     torch_dtype=torch.float32, # float32 is best for CPU
#     safety_checker=None  # <-- Disable NSFW filtering
# ).to("cpu")

# def apply_style_filter(input_image: Image.Image, style: str) -> Image.Image:
#     prompt_map = {
#         "anime": "anime style portrait of a young person",
#         "sketch": "black and white sketch of a face",
#         "cartoon": "cartoon style character with detailed lines",
#         "painting": "digital painting portrait",
#         "pixar": "pixar style face of a person, big eyes, clean lighting"
#     }

#     prompt = prompt_map.get(style.lower(), "portrait in arcane style")

#     # Resize image to standard size (SD requirement)
#     input_image = input_image.resize((512, 512)).convert("RGB")

#     # Generate stylized image
#     result = pipe(
#         prompt=prompt,
#         image=input_image,
#         strength=0.75,
#         guidance_scale=7.5
#     ).images[0]

#     return result
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
