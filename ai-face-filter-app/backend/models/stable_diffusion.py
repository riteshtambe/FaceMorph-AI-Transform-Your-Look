
from diffusers import StableDiffusionInstructPix2PixPipeline
from PIL import Image
import torch
import os
import uuid
import cv2
import numpy as np
import mediapipe as mp

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the instruct-pix2pix pipeline
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    "timbrooks/instruct-pix2pix",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    safety_checker=None  # Optional: disables NSFW filter
).to(device)


def detect_and_crop_face(pil_image):
    """ Detects face and crops the image to the face region using Mediapipe. """
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    # Convert PIL to OpenCV
    image_np = np.array(pil_image.convert("RGB"))
    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB))

        if results.detections:
            # Get first detected face bounding box
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box

            h, w, _ = image_cv2.shape
            xmin = int(bboxC.xmin * w)
            ymin = int(bboxC.ymin * h)
            width = int(bboxC.width * w)
            height = int(bboxC.height * h)

            # Add some padding
            pad = 40
            x1 = max(xmin - pad, 0)
            y1 = max(ymin - pad, 0)
            x2 = min(xmin + width + pad, w)
            y2 = min(ymin + height + pad, h)

            cropped_face = image_cv2[y1:y2, x1:x2]
            pil_cropped = Image.fromarray(cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB))
            return pil_cropped
        else:
            return pil_image  # fallback if no face detected

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
