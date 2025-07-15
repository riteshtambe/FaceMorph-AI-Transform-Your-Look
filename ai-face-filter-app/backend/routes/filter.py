from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image, UnidentifiedImageError
import os
from io import BytesIO

from backend.models.stable_diffusion import apply_style_filter

router = APIRouter()

# Directory to store processed images
OUTPUT_DIR = "temp"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@router.post("/")
async def filter_image(style: str, file: UploadFile = File(...)):
    try:
        # Reset file pointer
        file.file.seek(0)

        # Check file type and load image
        image = Image.open(file.file)

        # Optional: verify image is RGB (some filters expect this)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Save original temporarily (optional for logging/debugging)
        input_path = os.path.join(OUTPUT_DIR, "original_input.png")
        image.save(input_path)

        # ✅ Apply the AI filter
        output_image = apply_style_filter(image, style)

        # Save final result
        output_path = os.path.join(OUTPUT_DIR, "styled_output.png")
        output_image.save(output_path)

        # ✅ Return the output path to frontend
        return {"output_path": output_path}

    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Something went wrong: {str(e)}")

