import matplotlib.pyplot as plt
import cv2 as opencv
import numpy as np
import random


# Load Numpy and Show values
values = np.load("numeral_value.npy")
labels = np.load("numeral_label.npy")

val = random.randint(0, len(values))

print("Length of training data : %d\nLength of training labels : %d" %
      (len(values), len(labels)))


def showimageplots():
    imglen = 4
    figure = plt.figure()
    for i in range(1, imglen*imglen+1):
        index = random.randint(0, len(values))
        plt.subplot(imglen, imglen, i).set_title(labels[index])
        plt.imshow(values[index], cmap="gray", interpolation="nearest")
    plt.show()
