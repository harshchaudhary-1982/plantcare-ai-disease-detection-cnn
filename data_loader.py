"""Reproducible train/validation/test split for folder-based image datasets."""
import json
import shutil
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
from config import RAW_DATA_DIR, SPLIT_DATA_DIR, SEED, VALIDATION_FRACTION, TEST_FRACTION, SUPPORTED_EXTENSIONS


def discover_classes(dataset_dir=RAW_DATA_DIR):
    """Find immediate subdirectories containing supported image files."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_dir}\n"
            "Read data/README.md and place the extracted class folders there."
        )
    classes = sorted(
        folder.name for folder in dataset_dir.iterdir()
        if folder.is_dir() and any(
            file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS
            for file in folder.rglob("*")
        )
    )
    if len(classes) < 2:
        raise ValueError(f"Expected at least 2 class folders with images in {dataset_dir}. Found: {classes}")
    return classes


def _collect_files(dataset_dir, classes):
    paths, labels = [], []
    for label, class_name in enumerate(classes):
        class_dir = Path(dataset_dir) / class_name
        class_files = sorted(
            p for p in class_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not class_files:
            raise ValueError(f"No supported images found in class folder: {class_dir}")
        paths.extend(class_files)
        labels.extend([label] * len(class_files))
    return np.asarray(paths, dtype=object), np.asarray(labels, dtype=np.int32)


def create_split(dataset_dir=RAW_DATA_DIR, output_dir=SPLIT_DATA_DIR, force=False):
    """Create class-preserving train/validation/test folders using fixed seed.

    Split is performed at file level before copying; each source file is assigned
    to exactly one split. For near-duplicate images, use a dataset-specific
    duplicate audit before training.
    """
    dataset_dir, output_dir = Path(dataset_dir), Path(output_dir)
    marker = output_dir / "split_manifest.json"
    if marker.exists() and not force:
        with marker.open(encoding="utf-8") as f:
            return json.load(f)

    classes = discover_classes(dataset_dir)
    paths, labels = _collect_files(dataset_dir, classes)
    test_fraction = TEST_FRACTION
    val_fraction_of_remaining = VALIDATION_FRACTION / (1.0 - test_fraction)

    try:
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            paths, labels, test_size=test_fraction + VALIDATION_FRACTION,
            random_state=SEED, stratify=labels
        )
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels, test_size=test_fraction / (test_fraction + VALIDATION_FRACTION),
            random_state=SEED, stratify=temp_labels
        )
    except ValueError as exc:
        raise ValueError(
            "Could not create a stratified split. Each class needs enough images "
            "for train, validation, and test sets. Add more images or choose a smaller class-balanced subset."
        ) from exc

    if output_dir.exists():
        shutil.rmtree(output_dir)
    manifest = {"seed": SEED, "classes": classes, "counts": {}}
    for split_name, split_paths, split_labels in (
        ("train", train_paths, train_labels),
        ("validation", val_paths, val_labels),
        ("test", test_paths, test_labels),
    ):
        manifest["counts"][split_name] = {}
        for class_index, class_name in enumerate(classes):
            destination = output_dir / split_name / class_name
            destination.mkdir(parents=True, exist_ok=True)
            selected = split_paths[split_labels == class_index]
            manifest["counts"][split_name][class_name] = int(len(selected))
            for index, source in enumerate(selected):
                # Unique stable names avoid collisions if original names repeat.
                target = destination / f"{index:06d}_{Path(source).name}"
                shutil.copy2(source, target)

    manifest["total_images"] = int(len(paths))
    manifest["split_fractions"] = {"validation": VALIDATION_FRACTION, "test": TEST_FRACTION}
    output_dir.mkdir(parents=True, exist_ok=True)
    with marker.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest
