import matplotlib.pyplot as plt
import numpy as np
from src.predict import CLASS_NAMES

from src.logger import get_logger

logger = get_logger(__name__)

from pathlib import Path

Path(
    "outputs/plots"
).mkdir(
    parents=True,
    exist_ok=True
)

def show_dataset_samples(
        X,
        y,
        rows=3,
        cols=3
):

    logger.info(
        "Displaying sample images"
    )

    fig = plt.figure(
        "Sample Dataset Images",
        figsize=(10, 10)
    )
    print(y.shape)
    print(type(y[0]))
    print(y[0])

    for i in range(rows * cols):

        plt.subplot(
            rows,
            cols,
            i + 1
        )

        plt.imshow(X[i])

        plt.title(
            CLASS_NAMES[int(y[i])]
        )

        plt.axis("off")

    fig.tight_layout()

    fig.savefig(
        "outputs/plots/dataset_samples.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

def show_correct_predictions(
        X_test,
        y_test,
        y_pred,
        n=9
):

    logger.info(
        "Displaying correct predictions"
    )

    correct = np.where(
        y_pred == y_test.flatten()
    )[0]

    fig = plt.figure(
        "Correct Predictions",
        figsize=(10,10)
    )

    for i, idx in enumerate(correct[:n]):

        plt.subplot(
            3,
            3,
            i+1
        )

        plt.imshow(
            X_test[idx]
        )

        plt.title(
            f"{CLASS_NAMES[y_pred[idx]]}"
        )

        plt.axis("off")

    fig.tight_layout()
    fig.savefig(
        "outputs/plots/correct_predictions.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

def show_misclassifications(
        X_test,
        y_test,
        y_pred,
        n=9
):

    logger.info(
        "Displaying errors"
    )

    errors = np.where(
        y_pred != y_test.flatten()
    )[0]

    fig = plt.figure(
        "Misclassifications",        
        figsize=(12,10)
    )

    for i, idx in enumerate(errors[:n]):

        plt.subplot(
            3,
            3,
            i+1
        )

        plt.imshow(
            X_test[idx]
        )

        actual = CLASS_NAMES[
            y_test[idx]]

        predicted = CLASS_NAMES[
            y_pred[idx]
        ]

        plt.title(
            f"A:{actual}\nP:{predicted}"
        )

        plt.axis("off")

    fig.tight_layout()

    fig.savefig(
        "outputs/plots/misclassifications.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)


def plot_training_history(
        history
):

    logger.info(
        "Plotting training history"
    )

    fig = plt.figure(
        "Training History",
        figsize=(12,5)
    )

    plt.plot(
        history["accuracy"],
        label="Train Accuracy"
    )

    plt.plot(
        history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.title(
        "Training vs Validation Accuracy"
    )

    plt.legend()
    fig.savefig(
    "outputs/plots/training_history.png",
    dpi=300,
    bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

def plot_accuracy_loss(
        history
):

    logger.info(
        "Plotting accuracy and loss"
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14,5)
    )

    axes[0].plot(
        history["accuracy"]
    )

    axes[0].plot(
        history["val_accuracy"]
    )

    axes[0].set_title(
        "Accuracy"
    )

    axes[0].legend(
        ["Train", "Validation"]
    )

    axes[1].plot(
        history["loss"]
    )

    axes[1].plot(
        history["val_loss"]
    )

    axes[1].set_title(
        "Loss"
    )

    axes[1].legend(
        ["Train", "Validation"]
    )
    fig.savefig(
    "outputs/plots/accuracy_loss.png",
    dpi=300,
    bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

    