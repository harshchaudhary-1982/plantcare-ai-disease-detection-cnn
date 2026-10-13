"""Shared configuration for PlantCare AI."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw" / "PlantVillage"
SPLIT_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_PATH = MODEL_DIR / "plantcare_cnn.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
METADATA_PATH = MODEL_DIR / "training_metadata.json"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15
EPOCHS = 20
LEARNING_RATE = 1e-3
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
