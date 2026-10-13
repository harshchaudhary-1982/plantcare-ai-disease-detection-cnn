"""Image loading and preprocessing shared by training and inference."""
from pathlib import Path
import numpy as np
from PIL import Image, UnidentifiedImageError
from config import IMAGE_SIZE, SUPPORTED_EXTENSIONS


def load_rgb_image(image_source):
    """Open a path or file-like object and return a resized RGB PIL image."""
    try:
        with Image.open(image_source) as image:
            image = image.convert("RGB")
            return image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("This file is not a valid, readable image.") from exc


def image_to_array(image_source):
    """Return a float32 image array in [0, 255]; model handles rescaling."""
    image = load_rgb_image(image_source)
    array = np.asarray(image, dtype=np.float32)
    if array.shape != (IMAGE_SIZE[0], IMAGE_SIZE[1], 3):
        raise ValueError(f"Expected an RGB image of size {IMAGE_SIZE}.")
    return array


def prepare_batch(image_source):
    """Prepare one image as a batch of shape (1, height, width, 3)."""
    return np.expand_dims(image_to_array(image_source), axis=0)


def is_supported_image(path):
    return Path(path).suffix.lower() in SUPPORTED_EXTENSIONS
