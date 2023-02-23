import glob
import os
from io import BytesIO
from typing import Iterable, List, Tuple

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

model = Sequential(
    [
        tf.keras.Input(shape=(100,)),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(10, activation="linear"),
    ],
    name="numbers_classification_model",
)

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
)


def get_label_from_filename(filename: str) -> int:
    return int(os.path.split(filename)[-1][0])


def get_bw_value(pixel: Tuple[int]) -> int:
    if all([x == 255 for x in pixel]):
        return 1
    return 0


def transform_to_black_and_white_array(image_data: Iterable) -> List[int]:
    return [get_bw_value(pixel) for pixel in image_data]


def load_training_data() -> Tuple[list, list]:
    X_train = []
    y_train = []
    test_filename = ""
    for filename in sorted(glob.glob(os.path.join("src/training_data/", "*.png"))):
        with Image.open(filename) as image:
            test_filename == filename
            image_data = image.getdata()
            X_train.append(transform_to_black_and_white_array(image_data))
            y_train.append(get_label_from_filename(filename))

    return X_train, y_train


def train_model(X_train: list, y_train: list) -> None:
    model.fit(X_train, y_train, epochs=3000)
    model.summary()


def categorize_image(decoded_image: bytes):
    image = Image.open(BytesIO(decoded_image))
    bw_array = transform_to_black_and_white_array(image.getdata())

    prediction_array = model.predict([bw_array])
    prediction_array_with_softmax = tf.nn.softmax(prediction_array)
    prediction = np.argmax(prediction_array_with_softmax)

    return prediction
