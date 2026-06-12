from pathlib import Path
import pickle

from tensorflow.keras.callbacks import (
    EarlyStopping,
    LearningRateScheduler,
    ReduceLROnPlateau
)

from tensorflow.keras.models import (
    load_model
)
from tensorflow.keras.optimizers import Adam

from src.logger import get_logger
from src.model import build_model

logger = get_logger(__name__)


def train(
        X_train,
        y_train,
        epochs=35
):

    logger.info(
        "Training pipeline started"
    )

    Path(
        "models"
    ).mkdir(
        exist_ok=True
    )

    Path(
        "outputs"
    ).mkdir(
        exist_ok=True
    )

    Path(
        "outputs/reports"
    ).mkdir(
        parents=True,
        exist_ok=True
    )

    model_path = Path(
        "models/resnet_cifar10.keras"
    )

    if model_path.exists():

        logger.info(
            "Existing model found. Resuming training."
        )

        model = load_model(
            model_path
        )

    else:

        logger.info(
            "No existing model found. Creating new model."
        )

        model = build_model()

    optimizer = Adam(
        learning_rate=0.0001
        )
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    logger.info(
        "Model compilation complete"
    )
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

    logger.info(
        "Training started"
    )

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=64,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    model.save(
        model_path
    )

    logger.info(
        "Model saved successfully"
    )

    with open(
        "outputs/history.pkl",
        "wb"
    ) as f:

        pickle.dump(
            history.history,
            f
        )

    logger.info(
        "Training history saved"
    )

    with open(
        "outputs/reports/model_summary.txt",
        "w",
        encoding="utf-8"
    ) as f:

        model.summary(
            print_fn=lambda x: f.write(
                x + "\n"
            )
        )

    logger.info(
        "Model summary saved"
    )

    logger.info(
        "Training completed"
    )

    return history