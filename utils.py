"""Small shared helpers."""
import json
from pathlib import Path


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def load_class_names(path):
    with Path(path).open(encoding="utf-8") as file:
        names = json.load(file)
    if not isinstance(names, list) or not names or not all(isinstance(x, str) for x in names):
        raise ValueError("Class mapping must be a non-empty JSON list of class names.")
    return names
