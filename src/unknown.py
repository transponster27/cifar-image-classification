import numpy as np

from src.logger import get_logger

logger = get_logger(__name__)


def detect_unknown(
        probabilities,
        threshold=0.60
):
    """
    Detect whether prediction
    should be considered unknown.

    Parameters
    ----------
    probabilities : ndarray
        Softmax output

    threshold : float
        Minimum confidence required

    Returns
    -------
    bool
    """

    confidence = float(
        np.max(probabilities)
    )

    logger.info(
        f"Max confidence: {confidence:.4f}"
    )

    return confidence < threshold

def calculate_entropy(
        probabilities
):
    """
    Measures uncertainty.
    Higher entropy means
    more uncertainty.
    """

    probs = probabilities[0]

    return -np.sum(
        probs *
        np.log(probs + 1e-10)
    )