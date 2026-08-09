from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

Image.MAX_IMAGE_PIXELS = 50_000_000
ALLOWED_FORMATS = {"JPEG", "PNG"}

class ImageValidationError(ValueError):
    pass

@dataclass
class PreprocessedImage:
    display: Image.Image
    processed: Image.Image
    model_input: np.ndarray
    metadata: dict

def _open(source: bytes | str | Path) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(source) if isinstance(source, bytes) else source)
        image.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ImageValidationError("The uploaded file is not a readable retinal image.") from exc
    if image.format not in ALLOWED_FORMATS:
        raise ImageValidationError("Decoded image format must be JPEG or PNG.")
    return ImageOps.exif_transpose(image).convert("RGB")

def crop_retinal_field(rgb: np.ndarray, black_threshold: int = 10) -> tuple[np.ndarray, dict]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    mask = gray > black_threshold
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return rgb, {"field_detected": False, "crop_box": [0, 0, rgb.shape[1], rgb.shape[0]]}
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    if w * h < rgb.shape[0] * rgb.shape[1] * 0.15:
        return rgb, {"field_detected": False, "crop_box": [0, 0, rgb.shape[1], rgb.shape[0]]}
    return rgb[y:y+h, x:x+w], {"field_detected": True, "crop_box": [x, y, x+w, y+h]}

def preprocess_image(source, image_size: int = 224, clahe: bool = False,
                     min_dimension: int = 64, black_threshold: int = 10) -> PreprocessedImage:
    display = _open(source)
    if min(display.size) < min_dimension:
        raise ImageValidationError(f"Image dimensions must be at least {min_dimension} pixels.")
    rgb = np.asarray(display)
    cropped, metadata = crop_retinal_field(rgb, black_threshold)
    h, w = cropped.shape[:2]
    side = max(h, w)
    square = np.zeros((side, side, 3), dtype=np.uint8)
    square[(side-h)//2:(side-h)//2+h, (side-w)//2:(side-w)//2+w] = cropped
    resized = cv2.resize(square, (image_size, image_size), interpolation=cv2.INTER_AREA)
    if clahe:
        lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8)).apply(l)
        resized = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
    # EfficientNet in Keras includes input rescaling internally; preserve 0..255 floats.
    model_input = resized.astype(np.float32)[None, ...]
    metadata.update({
        "original_size": list(display.size),
        "processed_size": [image_size, image_size],
        "clahe": bool(clahe),
    })
    return PreprocessedImage(display, Image.fromarray(resized), model_input, metadata)

def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, "PNG", optimize=True)
    return buffer.getvalue()

