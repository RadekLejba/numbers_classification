import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

from .neural_network import predict_label

# Models with some more or less random values
model1 = Sequential(
    [
        tf.keras.Input(shape=(40000,)),
        Dense(128, activation="sigmoid"),
        Dense(64, activation="sigmoid"),
        Dense(32, activation="sigmoid"),
        Dense(10, activation="linear"),
    ],
    name="numbers_classification_model1",
)
model2 = Sequential(
    [
        tf.keras.Input(shape=(40000,)),
        Dense(128, activation="relu"),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(10, activation="linear"),
    ],
    name="numbers_classification_model2",
)
model3 = Sequential(
    [
        tf.keras.Input(shape=(40000,)),
        Dense(32, activation="relu"),
        Dense(32, activation="relu"),
        Dense(32, activation="relu"),
        Dense(10, activation="linear"),
    ],
    name="numbers_classification_model3",
)
model4 = Sequential(
    [
        tf.keras.Input(shape=(40000,)),
        Dense(16, activation="relu"),
        Dense(15, activation="relu"),
        Dense(14, activation="relu"),
        Dense(13, activation="relu"),
        Dense(12, activation="relu"),
        Dense(11, activation="relu"),
        Dense(10, activation="linear"),
    ],
    name="numbers_classification_model4",
)
model5 = Sequential(
    [
        tf.keras.Input(shape=(40000,)),
        Dense(10, activation="relu"),
        Dense(16, activation="relu"),
        Dense(16, activation="relu"),
        Dense(10, activation="linear"),
    ],
    name="numbers_classification_model5",
)

models = [model1, model2, model3, model4, model5]


def separate_data_to_train_dev_test(x: list, y: list) -> tuple:
    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=0.40, random_state=1
    )
    x_dev, x_test, y_dev, y_test = train_test_split(
        x_temp, y_temp, test_size=0.50, random_state=1
    )

    return x_train, y_train, x_test, y_test, x_dev, y_dev


def compare_models(x: list, y: list):
    x_train, y_train, x_test, y_test, x_dev, y_dev = separate_data_to_train_dev_test(
        x, y
    )
    messages = []
    for i, model in enumerate(models):
        predictions = []
        model.compile(
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        )
        model.fit(
            x_train,
            y_train,
            epochs=100,
        )
        predictions = model.predict(x_dev)
        for prediction in predictions:
            prediction_array_with_softmax = tf.nn.softmax(prediction)
            prediction = np.argmax(prediction_array_with_softmax)
            predictions.append(prediction)

        train_mse = mean_squared_error(y_dev, predictions) / 2
        messages.append(f"error for model {i}: {train_mse}")

    for message in messages:
        print(message)
