from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


CLASS_NAMES = [
    "buildings",
    "forest",
    "glacier",
    "mountain",
    "sea",
    "street",
]

MODEL_PATHS = {
    "EfficientNetB0": Path(
        "efficientnet_transfer_learning_model.keras"
    ),
    "ResNet50": Path(
        "resnet_transfer_learning_model.keras"
    ),
}


@st.cache_resource
def load_model(model_name: str):
    model_path = MODEL_PATHS.get(model_name)

    if model_path is None:
        raise ValueError(f"Unknown model: {model_name}")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file was not found: {model_path}"
        )

    return tf.keras.models.load_model(model_path)


def prepare_image(
    image: Image.Image,
    image_size: tuple[int, int] = (150, 150),
) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize(image_size)

    image_array = np.asarray(image, dtype=np.float32)

    # Batch dimension:
    # (150, 150, 3) -> (1, 150, 150, 3)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def predict_image(model, image_array: np.ndarray):
    probabilities = model.predict(
        image_array,
        verbose=0,
    )[0]

    predicted_index = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index])

    return predicted_class, confidence, probabilities



def prepare_image(image: Image.Image) -> np.ndarray:
    """
    Convert uploaded image into model input.
    """

    image = image.convert("RGB")
    image = image.resize((150, 150))

    image_array = np.array(image, dtype=np.float32)

    image_array = np.expand_dims(image_array, axis=0)

    return image_array