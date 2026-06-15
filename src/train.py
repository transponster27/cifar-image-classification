from pathlib import Path
import pickle
import json

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from src.logger import get_logger
from src.model import build_model

logger = get_logger(__name__)


def train(
        X_train,
        y_train,
        target_epochs=20
):

    logger.info("Training pipeline started")

    # Ensure directories exist
    Path("models").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)

    # Load last epoch state
    epoch_file = Path("outputs/last_epoch.json")
    initial_epoch = 0

    if epoch_file.exists():
        with open(epoch_file, "r") as f:
            initial_epoch = json.load(f)["last_epoch"]

        logger.info(f"Resuming from epoch {initial_epoch}")
    else:
        logger.info("Starting from scratch")

    # Load or create model
    model_path = Path("models/resnet_cifar10.keras")

    if model_path.exists():
        logger.info("Existing model found. Loading...")
        model = load_model(model_path)
    else:
        logger.info("No model found. Creating new one...")
        model = build_model()

    # Compile model
    optimizer = Adam(learning_rate=1e-4)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    logger.info("Model compiled")

    # Callbacks
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        verbose=1
    )

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(
        filepath="models/resnet_cifar10.keras",
        save_freq="epoch",
        save_best_only=False
    )

    # Skip if already trained
    if initial_epoch >= target_epochs:
        logger.info(
            f"Already trained ({initial_epoch}/{target_epochs})"
        )
        return None

    # TRAIN MODEL
    logger.info("Training started")

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        initial_epoch=initial_epoch,
        epochs=target_epochs,
        batch_size=256,
        callbacks=[early_stop, reduce_lr, checkpoint],
        verbose=1
    )

    # SAVE MODEL
    model.save(model_path)
    logger.info("Model saved")

    # SAVE HISTORY (APPEND SAFE)
    history_file = Path("outputs/history.pkl")

    if history_file.exists():
        with open(history_file, "rb") as f:
            old_history = pickle.load(f)
    else:
        old_history = {}

    for key, values in history.history.items():
        if key not in old_history:
            old_history[key] = []
        old_history[key].extend(values)

    with open(history_file, "wb") as f:
        pickle.dump(old_history, f)

    logger.info("History saved")

    # SAVE MODEL SUMMARY
    with open("outputs/reports/model_summary.txt", "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    logger.info("Model summary saved")

    # UPDATE EPOCH TRACKER
    completed_epoch = initial_epoch + len(history.history["loss"])

    with open(epoch_file, "w") as f:
        json.dump(
            {"last_epoch": completed_epoch},
            f
        )

    logger.info(f"Training completed up to epoch {completed_epoch}")

    return history