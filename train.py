import matplotlib.pyplot as plt
import cv2 as opencv
import numpy as np
import random


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
