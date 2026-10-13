"""Prediction contract tests that do not require a downloaded dataset."""
import json
import numpy as np
import pytest
from pathlib import Path
from predict import predict_image


def test_missing_model_fails_clearly(tmp_path):
    image_path = tmp_path / "leaf.png"
    image_path.write_bytes(b"not needed because model check happens first")
    with pytest.raises(FileNotFoundError, match="Train the model first"):
        predict_image(image_path, tmp_path / "missing.keras", tmp_path / "classes.json")


def test_mapping_file_is_a_list(tmp_path):
    mapping = tmp_path / "classes.json"
    mapping.write_text(json.dumps(["Tomato_healthy", "Tomato_Late_blight"]), encoding="utf-8")
    assert json.loads(mapping.read_text(encoding="utf-8")) == ["Tomato_healthy", "Tomato_Late_blight"]
