import random
import os
import sys

import matplotlib.pyplot as plt
import cv2
import numpy as np

import tensorflow.keras as keras
import tensorflow as tf

class Data_Trainer():
    def __init__(self):
        self.training_dataset_path = "./processed/training_dataset.npy"
        self.testing_dataset_path = "./processed/testing_dataset.npy"
        self.labels_path = "./processed/labels.csv"
        self.model = self.get_model()
        self.labels = self.get_labels()
        self.training_set = []
        self.training_labels = []
        self.testing_set = []
        self.testing_labels = []
        self.training_records = 0
        self.testing_records = 0

    def get_labels(self):
        with open(self.labels_path,"r") as csvfile:
            contents = csvfile.read().replace("\n","").split(",")
            print("Labels : Normalized")
            return contents

    def get_model(self):
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(28, 28)),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(46, activation=tf.nn.softmax)
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        print("Model : Created")
        return model

    def make_one_hot(self,label_name):
        one_hot_array = np.zeros(46)
        index = self.labels.index(label_name)
        one_hot_array[index] = 1
        return one_hot_array

    def train_the_model(self):
        print("Training data : Processing training records ..")
        compound_array = np.load("{}".format(self.training_dataset_path))
        for records in compound_array:
            self.training_set.append(records[0])
            self.training_labels.append(self.make_one_hot(records[1]))
            self.training_records += 1
        print("Training data : Processed {} records".format(self.training_records))

    def test_the_model(self):
        print("Testing data : Processing testing records ..")
        compound_array = np.load("{}".format(self.testing_dataset_path))
        for records in compound_array:
            self.testing_set.append(records[0])
            self.testing_labels.append(self.make_one_hot(records[1]))
            self.testing_records += 1
        print("Testing data : Processed {} records".format(self.training_records))

    def train_model_with_data(self):
        self.train_the_model()
        self.test_the_model()

if __name__ == "__main__":
    data_trainer = Data_Trainer()
    data_trainer.train_model_with_data()
    plt.imshow(data_trainer.training_set[0])
    print(data_trainer.training_labels[0])
    plt.show()
