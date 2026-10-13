"""CNN model definition."""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from config import IMAGE_SIZE, LEARNING_RATE


def build_model(num_classes: int) -> keras.Model:
    """Build a modest CNN. Input rescaling is embedded in the saved model."""
    if num_classes < 2:
        raise ValueError("At least two classes are required to train a classifier.")

    augmentation = keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.10),
        layers.RandomContrast(0.10),
    ], name="training_augmentation")

    inputs = keras.Input(shape=(*IMAGE_SIZE, 3), name="leaf_image")
    x = layers.Rescaling(1.0 / 255.0, name="rescale")(inputs)
    x = augmentation(x)

    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Dropout(0.15)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.35)(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="class_probabilities")(x)

    model = keras.Model(inputs, outputs, name="plantcare_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
