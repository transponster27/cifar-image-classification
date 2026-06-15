import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from src.data import load_dataset
from src.train import train
from src.evaluate import evaluate
from src.predict import load_trained_model
from src.predict import predict_single_image
from src.predict import load_external_image
from src.visualisation import (
    show_dataset_samples,
    show_correct_predictions,
    show_misclassifications,
    plot_training_history,
    plot_accuracy_loss,
)
from src.error import show_errors
import numpy as np
from src.logger import get_logger
import pickle
from pathlib import Path

logger = get_logger(__name__)

logger.info("Application started")


# Load Dataset

X_train, X_test, y_train, y_test = load_dataset()

#Train Model
history = train(
    X_train,
    y_train,
    target_epochs=20
)

# Evaluate Model

evaluate(
    X_test,
    y_test
)

# Load Model

model = load_trained_model()

# Predict Entire Test Set

predictions = model.predict(
    X_test,
    verbose=0
)

y_pred = np.argmax(
    predictions,
    axis=1
)

# Single Image Prediction Example

image = load_external_image(
    "inputs/sample_image.jpg"
)

predicted_class, confidence, probabilities = predict_single_image(
    model,
    image
    # X_test[0]
)

# Visualizations

show_dataset_samples(
    X_test,
    y_test
)

show_correct_predictions(
    X_test,
    y_test,
    y_pred
)

show_misclassifications(
    X_test,
    y_test,
    y_pred
)

with open(
    "outputs/history.pkl",
    "rb"
) as f:
    history = pickle.load(f)


plot_training_history(
    history
)

plot_accuracy_loss(
    history
)

# Error Analysis

show_errors(
    X_test,
    y_test
)

logger.info("Application finished")