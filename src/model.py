# src/model.py

from tensorflow.keras.models import Model

from tensorflow.keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    ReLU,
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    Add
)

from src.logger import get_logger

logger = get_logger(__name__)


def residual_block(
    x,
    filters,
    stride=1
):

    shortcut = x

    x = Conv2D(
        filters,
        (3, 3),
        strides=stride,
        padding="same",
        use_bias=False
    )(x)

    x = BatchNormalization()(x)

    x = ReLU()(x)

    x = Conv2D(
        filters,
        (3, 3),
        padding="same",
        use_bias=False
    )(x)

    x = BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:

        shortcut = Conv2D(
            filters,
            (1, 1),
            strides=stride,
            padding="same",
            use_bias=False
        )(shortcut)

        shortcut = BatchNormalization()(shortcut)

    x = Add()([
        x,
        shortcut
    ])

    x = ReLU()(x)

    return x


def build_model():

    logger.info(
        "Building ResNet18"
    )

    inputs = Input(
        shape=(32, 32, 3)
    )

    x = Conv2D(
        64,
        (3, 3),
        padding="same",
        use_bias=False
    )(inputs)

    x = BatchNormalization()(x) #std1 mean0

    x = ReLU()(x)

    # Stage 1
    x = residual_block(x, 64)
    x = residual_block(x, 64)

    # Stage 2
    x = residual_block(
        x,
        128,
        stride=2
    )

    x = residual_block(
        x,
        128
    )

    # Stage 3
    x = residual_block(
        x,
        256,
        stride=2
    )

    x = residual_block(
        x,
        256
    )

    # Stage 4
    x = residual_block(
        x,
        512,
        stride=2
    )

    x = residual_block(
        x,
        512
    )

    x = GlobalAveragePooling2D()(x)

    x = Dropout(
        0.5
    )(x)

    outputs = Dense(
        11,
        activation="softmax"
    )(x)

    model = Model(
        inputs,
        outputs,
        name="ResNet18"
    )

    logger.info(
        "ResNet18 created successfully"
    )

    return model


if __name__ == "__main__":

    model = build_model()

    model.summary()
