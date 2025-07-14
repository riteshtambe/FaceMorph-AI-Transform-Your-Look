from diffusers import StableDiffusionPipeline
from PIL import Image
import torch
import os

# Load model (make sure you have a GPU or adjust for CPU)
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16,
    revision="fp16"
).to("cuda")

def apply_style(image_path: str, prompt: str):
    image = Image.open(image_path).convert("RGB")
    
    # NOTE: This model accepts a prompt and generates an image. It doesn't take input images.
    # For the sake of this example, we treat the prompt as style
    # You can fine-tune or implement controlnet for real style transfer
    
    result = pipe(prompt=prompt, num_inference_steps=30).images[0]
    
    os.makedirs("outputs", exist_ok=True)
    output_file = f"outputs/{prompt}_{os.path.basename(image_path)}"
    result.save(output_file)

    return output_file
