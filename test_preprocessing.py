from io import BytesIO
import numpy as np
import pytest
from PIL import Image
from src.preprocessing import image_to_array, prepare_batch, load_rgb_image


def sample_png():
    image = Image.new("RGB", (80, 60), color=(20, 120, 30))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer


def test_image_to_array_shape_and_type():
    arr = image_to_array(sample_png())
    assert arr.shape == (224, 224, 3)
    assert arr.dtype == np.float32
    assert arr.min() >= 0 and arr.max() <= 255


def test_prepare_batch_shape():
    batch = prepare_batch(sample_png())
    assert batch.shape == (1, 224, 224, 3)


def test_invalid_image_raises_value_error():
    with pytest.raises(ValueError):
        load_rgb_image(BytesIO(b"not an image"))
