"""Evaluate the saved model on the held-out test set and save real metrics."""
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from config import MODEL_PATH, CLASS_NAMES_PATH, SPLIT_DATA_DIR, RESULTS_DIR, IMAGE_SIZE, BATCH_SIZE, SEED
from src.utils import load_class_names


def main():
    if not MODEL_PATH.is_file() or not CLASS_NAMES_PATH.is_file():
        raise FileNotFoundError("Trained model/class mapping missing. Run `python train.py` first.")
    test_dir = SPLIT_DATA_DIR / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError("Test split missing. Run `python train.py` first.")
    class_names = load_class_names(CLASS_NAMES_PATH)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, labels="inferred", label_mode="int", image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE, shuffle=False
    )
    if list(test_ds.class_names) != class_names:
        raise ValueError("Test class order does not match saved class mapping.")
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
    model = tf.keras.models.load_model(MODEL_PATH)

    y_true = np.concatenate([labels.numpy() for _, labels in test_ds])
    probabilities = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)
    report = classification_report(y_true, y_pred, labels=range(len(class_names)),
                                   target_names=class_names, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "classification_report": report,
        "class_names": class_names,
        "confusion_matrix": matrix.tolist(),
        "test_sample_count": int(len(y_true))
    }
    with (RESULTS_DIR / "evaluation.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Save a readable text report.
    text_report = classification_report(y_true, y_pred, labels=range(len(class_names)),
                                        target_names=class_names, zero_division=0)
    (RESULTS_DIR / "classification_report.txt").write_text(
        f"Test accuracy: {results['test_accuracy']:.4f}\n"
        f"Test samples: {len(y_true)}\n\n{text_report}\n", encoding="utf-8"
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 0.65), max(6, len(class_names) * 0.55)))
    image = ax.imshow(matrix, interpolation="nearest")
    fig.colorbar(image, ax=ax)
    ax.set(title="PlantCare AI — Test Confusion Matrix", xlabel="Predicted label", ylabel="True label")
    ax.set_xticks(range(len(class_names)), class_names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(class_names)), class_names, fontsize=7)
    if len(class_names) <= 20:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    print(f"Test accuracy: {results['test_accuracy']:.4f}")
    print(f"Test samples: {len(y_true)}")
    print(text_report)
    print(f"Reports and confusion matrix saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
