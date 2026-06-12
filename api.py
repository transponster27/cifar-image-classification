from pathlib import Path
import tempfile

from fastapi import FastAPI
from fastapi import UploadFile
from fastapi import File
from fastapi import HTTPException
import numpy as np

from src.logger import get_logger

from src.predict import (
    load_trained_model,
    load_external_image,
    predict_single_image,
    CLASS_NAMES
)

logger = get_logger(__name__)

# FastAPI Application

app = FastAPI(
    title="CIFAR-10 Image Classification API",
    description="CNN-based image classification using TensorFlow/Keras",
    version="1.0.0"
)

# Load model once when API starts

logger.info(
    "Loading trained model..."
)

model = load_trained_model()

logger.info(
    "API ready"
)

# Root Endpoint

@app.get("/")
def home():
    """
    API landing page.
    """

    return {
        "message":
        "CIFAR-10 Image Classification API Running"
    }


# Health Check

@app.get("/health")
def health():
    """
    Used by monitoring systems
    to verify API availability.
    """

    return {
        "status": "healthy",
        "model_loaded": True
    }


# Prediction Endpoint

@app.post("/predict")
async def predict(
        file: UploadFile = File(...)
):
    """
    Upload image and return prediction.
    """

    try:

        logger.info(
            f"Received file: {file.filename}"
        )

        # Save uploaded image temporarily

        suffix = Path(
            file.filename
        ).suffix

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            contents = await file.read()

            temp_file.write(
                contents
            )

            temp_path = temp_file.name

        # Preprocess image

        image = load_external_image(
            temp_path
        )

        # Predict

        predicted_class, confidence, probabilities = (
            predict_single_image(
                model,
                image
            )
        )

        # Top 3 predictions

        top_predictions = []

        top3 = np.argsort(
            probabilities[0]
        )[-3:][::-1]

        for idx in top3:

            top_predictions.append({

                "class":
                CLASS_NAMES[idx],

                "probability":
                round(
                    float(
                        probabilities[0][idx]
                    ),
                    4
                )
            })

        logger.info(
            f"Prediction={predicted_class}"
        )

        # API Response

        return {

            "filename":
            file.filename,

            "prediction":
            predicted_class,

            "confidence":
            round(
                confidence,
                4
            ),

            "top_predictions":
            top_predictions
        }

    except Exception as e:

        logger.exception(
            "Prediction failed"
        )

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# Run

# uvicorn api:app --reload