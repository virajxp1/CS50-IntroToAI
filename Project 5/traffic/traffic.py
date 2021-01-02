import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
# noinspection PyUnresolvedReferences
from tensorflow.compat.v1 import ConfigProto
# noinspection PyUnresolvedReferences
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D, Flatten, Dropout

EPOCHS = 10
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def main():
    # Check command-line arguments

    data_dir = os.getcwd() + "\\gtsrb"
    # Get image arrays and labels for all image files
    images, labels = load_data(data_dir)

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test, y_test, verbose=2)

    # Save model to file
    filename = "model"
    model.save(filename)
    print(f"Model saved to {filename}.")


def load_data(data_dir):
    data = []
    for i in range(0, NUM_CATEGORIES):
        dir_of_folder = data_dir + "\\" + str(i)
        if os.path.isdir(dir_of_folder):
            print(f"Loading files from {dir_of_folder}...")
        pics_in_dir = os.listdir(dir_of_folder)
        for filename in pics_in_dir:
            img = cv2.imread(dir_of_folder + "\\" + filename, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
            data.append((img, i))
    images, labels = map(list, zip(*data))
    return images, labels


def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    input_size = (IMG_WIDTH, IMG_HEIGHT, 3)

    model = tf.keras.models.Sequential([

        # 1st Conv block
        Conv2D(filters=64, kernel_size=3, padding='same', activation='relu', input_shape=input_size),
        Conv2D(filters=64, kernel_size=3, padding='same', activation='relu'),
        MaxPool2D(pool_size=2, strides=2, padding='same'),

        # 2nd Conv block
        Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
        Conv2D(filters=128, kernel_size=3, padding='same', activation='relu'),
        MaxPool2D(pool_size=2, strides=2, padding='same'),

        # Flatten units
        Flatten(),
        Dropout(0.2),

        # Add a hidden layer with dropout
        tf.keras.layers.Dense(4096, activation="relu"),
        tf.keras.layers.Dropout(0.5),

        # Add a hidden layer
        tf.keras.layers.Dense(NUM_CATEGORIES * 16, activation="relu"),

        # Add an output layer with output units for all categories
        tf.keras.layers.Dense(NUM_CATEGORIES, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


if __name__ == "__main__":
    main()
