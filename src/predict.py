from pathlib import Path

import numpy as np

from PIL import Image

from tensorflow.keras.models import load_model

from src.logger import get_logger

from src.unknown import (
    detect_unknown,
    calculate_entropy
)

logger = get_logger(__name__)

CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
    "unknown"
]


def load_trained_model(
        model_path="models/resnet_cifar10.keras"
):
    """
    Load trained CNN model.
    """

    logger.info(
        f"Loading model from {model_path}"
    )

    model = load_model(
        model_path
    )

    logger.info(
        "Model loaded successfully"
    )

    return model


def load_external_image(
        image_path
):
    """
    Load external image and
    preprocess it for CIFAR-10.
    """

    logger.info(
        f"Loading image: {image_path}"
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize(
        (32, 32)
    )

    image = np.array(
        image
    )

    image = image.astype(
        "float32"
    ) / 255.0

    logger.info(
        f"Image shape: {image.shape}"
    )

    return image


def predict_single_image(
        model,
        image,
        confidence_threshold=0.60
):
    """
    Predict a single image.

    Parameters
    ----------
    model : keras model

    image : ndarray
        Shape (32,32,3)

    Returns
    -------
    predicted_class : str

    confidence : float
    """

    logger.info(
        "Running prediction"
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    probabilities = model.predict(
        image,
        verbose=0
    )

    predicted_idx = np.argmax(
        probabilities
    )

    confidence = float(
        np.max(probabilities)
    )

    entropy = calculate_entropy(
        probabilities
    )

    logger.info(
        f"Entropy={entropy:.4f}"
    )

    if detect_unknown(
            probabilities,
            threshold=confidence_threshold
    ):
        predicted_class = "unknown"

    else:
        predicted_class = CLASS_NAMES[
            predicted_idx
        ]

    logger.info(
        f"Prediction={predicted_class}"
    )

    logger.info(
        f"Confidence={confidence:.4f}"
    )

    print("\nTop 3 Predictions\n")

    top3 = np.argsort(
        probabilities[0]
    )[-3:][::-1]

    for idx in top3:

        print(
            f"{CLASS_NAMES[idx]:12s}"
            f" : {probabilities[0][idx]:.4f}"
        )

    print(
        f"\nFinal Prediction : {predicted_class}"
    )

    print(
        f"Confidence       : {confidence:.4f}"
    )

    print(
        f"Entropy          : {entropy:.4f}"
    )

    Path(
        "outputs/predictions"
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        "outputs/predictions/predictions.log",
        "a",
        encoding="utf-8"
    ) as f:

        f.write(
            f"Prediction={predicted_class}, "
            f"Confidence={confidence:.4f}, "
            f"Entropy={entropy:.4f}\n"
        )

    return (
        predicted_class,
        confidence,
        probabilities
    )


if __name__ == "__main__":

    model = load_trained_model()

    image = load_external_image(
        "inputs/sample_image.jpg"
    )

    predicted_class, confidence = (
        predict_single_image(
            model,
            image
        )
    )

    print(
        f"\nResult: {predicted_class}"
    )

    print(
        f"Confidence: {confidence:.4f}"
    )