"""Command-line prediction using the saved PlantCare AI model."""
import argparse
from pathlib import Path
import numpy as np
from config import MODEL_PATH, CLASS_NAMES_PATH
from src.preprocessing import prepare_batch
from src.utils import load_class_names


def predict_image(image_path, model_path=MODEL_PATH, class_names_path=CLASS_NAMES_PATH):
    if not Path(model_path).is_file():
        raise FileNotFoundError(f"Model not found: {model_path}. Train the model first with: python train.py")
    if not Path(class_names_path).is_file():
        raise FileNotFoundError(f"Class mapping not found: {class_names_path}. Train the model first.")
    # Import TensorFlow only after checking paths, so missing-model errors stay clear
    # even in environments where the optional deep-learning dependency is not installed.
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is not installed. Install requirements.txt in the project virtual environment.") from exc
    model = tf.keras.models.load_model(model_path)
    class_names = load_class_names(class_names_path)
    batch = prepare_batch(image_path)
    probabilities = model.predict(batch, verbose=0)[0]
    if len(probabilities) != len(class_names):
        raise ValueError("Model output count does not match saved class mapping.")
    index = int(np.argmax(probabilities))
    return {"class_name": class_names[index], "confidence": float(probabilities[index]),
            "probabilities": probabilities.tolist()}


def main():
    parser = argparse.ArgumentParser(description="Predict a plant leaf class.")
    parser.add_argument("image", help="Path to a JPG, JPEG, or PNG leaf image")
    args = parser.parse_args()
    try:
        result = predict_image(args.image)
        print(f"Prediction: {result['class_name']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print("Note: experimental prediction; confirm with an agricultural expert.")
    except (ValueError, FileNotFoundError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
