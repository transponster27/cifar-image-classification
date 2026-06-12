import numpy as np

import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

from src.logger import get_logger
from src.data import load_dataset
from src.predict import CLASS_NAMES

logger = get_logger(__name__)



def show_errors(
        X_test,
        y_test
):

    logger.info(
        "Error analysis started"
    )

    model = load_model(
        "models/cifar10_cnn.keras"
    )

    predictions = model.predict(
        X_test,
        verbose=0
    )

    y_pred = np.argmax(
        predictions,
        axis=1
    )

    errors = np.where(
        y_pred != y_test.flatten()
    )[0]

    logger.info(
        f"Misclassified images: {len(errors)}"
    )

    for i in range(9):

        idx = errors[i]

        plt.subplot(3, 3, i + 1)

        plt.imshow(X_test[idx])

        plt.title(
        f"Actual:{CLASS_NAMES[y_test[idx]]}\n"
        f"Predicted:{CLASS_NAMES[y_pred[idx]]}"
    )

        plt.axis("off")

    plt.savefig(
        "outputs/plots/error_analysis.png",
        dpi=300,
        bbox_inches="tight"
        )
    plt.show()

    logger.info(
        "Error visualization completed"
    )


if __name__ == "__main__":
    show_errors()