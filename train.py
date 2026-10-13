"""Train PlantCare AI from class-folder images."""
import argparse
import json
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from config import (
    RAW_DATA_DIR, SPLIT_DATA_DIR, MODEL_DIR, RESULTS_DIR, MODEL_PATH,
    CLASS_NAMES_PATH, METADATA_PATH, IMAGE_SIZE, BATCH_SIZE, SEED, EPOCHS
)
from src.data_loader import create_split
from src.model import build_model
from src.utils import save_json


def make_dataset(directory, shuffle):
    dataset = keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="int",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        seed=SEED if shuffle else None,
    )
    return dataset.prefetch(tf.data.AUTOTUNE)


def main():
    parser = argparse.ArgumentParser(description="Train the PlantCare AI CNN.")
    parser.add_argument("--data-dir", default=str(RAW_DATA_DIR), help="Folder containing one subfolder per class")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--force-split", action="store_true", help="Rebuild train/validation/test split")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    tf.keras.utils.set_random_seed(SEED)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest = create_split(args.data_dir, SPLIT_DATA_DIR, force=args.force_split)
    classes = manifest["classes"]
    train_ds = make_dataset(SPLIT_DATA_DIR / "train", shuffle=True)
    val_ds = make_dataset(SPLIT_DATA_DIR / "validation", shuffle=False)

    # Keep class ordering consistent with the dataset loader and saved mapping.
    if list(train_ds.class_names) != classes or list(val_ds.class_names) != classes:
        raise RuntimeError("Class order changed between split creation and TensorFlow loading.")

    model = build_model(num_classes=len(classes))
    model.summary()
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(str(MODEL_PATH), monitor="val_loss", save_best_only=True, verbose=1),
        keras.callbacks.CSVLogger(str(RESULTS_DIR / "training_history.csv")),
    ]
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)
    # Checkpoint contains best val_loss weights; save that best-restored model explicitly.
    model.save(MODEL_PATH)
    save_json(classes, CLASS_NAMES_PATH)
    save_json({
        "seed": SEED, "image_size": list(IMAGE_SIZE), "batch_size": BATCH_SIZE,
        "requested_epochs": args.epochs, "completed_epochs": len(history.history["loss"]),
        "class_names": classes, "dataset_manifest": manifest,
        "model_file": MODEL_PATH.name
    }, METADATA_PATH)

    # Plot actual training history.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for metric, ylabel in (("accuracy", "Accuracy"), ("loss", "Loss")):
        plt.figure(figsize=(8, 5))
        plt.plot(history.history[metric], label=f"Training {metric}")
        plt.plot(history.history[f"val_{metric}"], label=f"Validation {metric}")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.title(f"PlantCare AI: {ylabel} by Epoch")
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"{metric}_curves.png", dpi=150)
        plt.close()

    print(f"\nTraining finished. Model saved to: {MODEL_PATH}")
    print(f"Class mapping saved to: {CLASS_NAMES_PATH}")
    print("Next: run `python evaluate.py` to measure held-out test performance.")


if __name__ == "__main__":
    main()
