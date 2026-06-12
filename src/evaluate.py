import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)
from src.logger import get_logger
from src.data import load_dataset
from pathlib import Path
import matplotlib.pyplot as plt
from src.predict import CLASS_NAMES

logger = get_logger(__name__)


def evaluate(
        X_test,
        y_test
):

    logger.info(
        "Evaluation started"
    )

    # _, X_test, _, y_test = load_dataset()

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

    logger.info(
        "Classification report generated"
    )
    report = classification_report(
        y_test.flatten(),
        y_pred,
        target_names=CLASS_NAMES)

    print(report)

    Path(
        "outputs/reports"
    ).mkdir(
        parents=True,
        exist_ok=True
    )
    with open(
        "outputs/reports/classification_report.txt",
        "w"
    ) as f :
        f.write(report)        

    logger.info(
        "Confusion matrix generated"
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    fig, ax = plt.subplots(
    figsize=(10,8)
    )

    im = ax.imshow(cm)

    ax.set_xticks(
        np.arange(len(CLASS_NAMES))
    )

    ax.set_yticks(
        np.arange(len(CLASS_NAMES))
    )

    ax.set_xticklabels(
        CLASS_NAMES,
        rotation=45
    )

    ax.set_yticklabels(
        CLASS_NAMES
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.title(
        "Confusion Matrix"
    )

    plt.colorbar(im)

    plt.tight_layout()

    plt.savefig(
        "outputs/reports/confusion_matrix.png"
    )

    plt.close()

    logger.info(
        "Evaluation completed"
    )
    
if __name__ == "__main__":
    evaluate()