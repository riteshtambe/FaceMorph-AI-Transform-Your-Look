from fastapi import APIRouter, UploadFile, File
from backend.models.stable_diffusion import apply_style
import shutil
import os

router = APIRouter()

@router.post("/")
async def style_transfer(style: str, file: UploadFile = File(...)):
    temp_path = f"temp/{file.filename}"
    os.makedirs("temp", exist_ok=True)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    output_path = apply_style(temp_path, style)
    return {"output_path": output_path}
