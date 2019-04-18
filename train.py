import matplotlib.pyplot as plt
import cv2 as opencv
import numpy as np
import random
import os

# Tensorflow for training
import tensorflow.keras as keras
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print("Using ", tf.__version__)


def get_data(values, labels):
    values = np.load(values)
    labels = np.load(labels)
    return (values, labels)


def showimageplots():
    values, labels = get_data("numeral_value.npy", "numeral_label.npy")
    print("Length of training data : %d\nLength of training labels : %d" %
          (len(values), len(labels)))
    grid = 2
    for i in range(1, grid*grid+1):
        index = random.randint(0, len(values))
        plt.subplot(grid, grid, i).set_title(labels[index])
        plt.imshow(values[index], cmap="gray", interpolation="nearest")
    plt.show()


def get_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model():
    values, labels = get_data("data/numeral_value.npy", "data/numeral_label.npy")
    values = tf.keras.utils.normalize(values, axis=1)
    model = get_model()
    print(values.shape, labels.shape)
    session = model.fit(values, labels, epochs=20)
    model.save("data/hindi-num.model")
    print("----\nTesting\n----")
    history = model.fit(values, labels, epochs=5)
    print("Test Loss :", history)
    print("Test Accuracy :", history)


def test_model():
    values, labels = get_data("data/numeral_value.npy", "data/numeral_label.npy")
    saved_model = tf.keras.models.load_model('data/hindi-num.model')
    index = random.randint(0, len(values))
    prediction = saved_model.predict(values)
    grid = 3
    for i in range(1, grid*grid+1):
        index = random.randint(0, len(values))
        plt.subplot(grid, grid, i).set_title(np.argmax(prediction[index]))
        plt.imshow(values[index], cmap="gray", interpolation="nearest")
    plt.show()


train_model()
test_model()
