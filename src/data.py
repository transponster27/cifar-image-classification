from tensorflow.keras.datasets import cifar10
from tensorflow.keras.datasets import cifar100
from sklearn.utils import shuffle

import numpy as np

from src.logger import get_logger

logger = get_logger(__name__)

def load_dataset():

    logger.info(
        "Loading CIFAR10 and CIFAR100"
    )

    (x10_train, y10_train), (x10_test, y10_test) = (
        cifar10.load_data()
    )

    (x100_train, y100_train), (x100_test, y100_test) = (
        cifar100.load_data(label_mode="fine")
    )

    # Unknown 5000 random classes used in training

    np.random.seed(42)

    unknown_train_idx = np.random.choice(
        len(x100_train),
        5000,
        replace=False
    )

    unknown_test_idx = np.random.choice(
        len(x100_test),
        1000,
        replace=False
    )
    #Extract unknown subset
    x_unknown_train = x100_train[unknown_train_idx]
    x_unknown_test = x100_test[unknown_test_idx]

    #create unknown label 10
    UNKNOWN_LABEL = 10

    y_unknown_train = np.full(
        (len(x_unknown_train), 1),
        UNKNOWN_LABEL
    )

    y_unknown_test = np.full(
        (len(x_unknown_test), 1),
        UNKNOWN_LABEL
    )
    # Merge train

    X_train = np.concatenate([
        x10_train,
        x_unknown_train
    ], axis=0)

    y_train = np.concatenate([
        y10_train,
        y_unknown_train
    ], axis=0)

    # Merge test

    X_test = np.concatenate([
        x10_test,
        x_unknown_test
    ], axis=0)

    y_test = np.concatenate([
        y10_test,
        y_unknown_test
    ], axis=0)

    # Normalize

    X_train = (
        X_train.astype("float32")
        / 255.0
    )

    X_test = (
        X_test.astype("float32")
        / 255.0
    )
    #shuffle
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    X_test, y_test = shuffle(X_test, y_test, random_state=42)
    #flatten
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    logger.info(
        f"Train Shape: {X_train.shape}"
    )

    logger.info(
        f"Test Shape: {X_test.shape}"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )