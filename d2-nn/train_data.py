import random
import os
import sys
import time
import datetime

import matplotlib.pyplot as plt
import cv2
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
        self.model_name = "trained_hindi_chars.model"
        self.training_start_time = 0
        self.training_end_time = 0
        self.elapsed_time = 0
        self.accuracy = 0
        self.loss = 1
        self.training_epochs = 3
        self.trained_directory_name = "trained"
        self.create_trained_directory()

    def create_trained_directory(self):
        if not os.path.exists(self.trained_directory_name):
            os.mkdir(self.trained_directory_name)

    def get_labels(self):
        with open(self.labels_path,"r") as csvfile:
            contents = csvfile.read().replace("\n","").split(",")
            print("Labels : Normalized")
            return contents

    def get_model(self):
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(32, 32)),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(128, activation=tf.nn.relu),
            keras.layers.Dense(46, activation=tf.nn.softmax)
        ])
        model.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
        print("Model : Created")
        return model

    def log_train_data(self):
        logstring = "\n"
        start_time_in_seconds = datetime.datetime.fromtimestamp(self.training_start_time).strftime('%M:%S.%f')[:-4]
        end_time_in_seconds = datetime.datetime.fromtimestamp(self.training_end_time).strftime('%M:%S.%f')[:-4]
        self.elapsed_time = self.training_start_time - self.training_end_time
        logstring += "Training TRN_{}\n".format(self.training_start_time)
        logstring += "Training Status : Completed\n"
        logstring += "Process ID : TRN_{}\n".format(self.training_start_time)
        logstring += "Training start time : {}\n".format(start_time_in_seconds)
        logstring += "Training end time : {}\n".format(end_time_in_seconds)
        logstring += "Total training time : {}\n".format(self.elapsed_time)
        logstring += "Training epochs : {}\n".format(self.training_epochs)
        logstring += "Loss : {}\n".format(self.loss)
        logstring += "Accuracy : {}\n".format(self.accuracy)
        print(logstring)
        with open("{}/train_logfile.txt".format(self.trained_directory_name), "a") as trainfile:
            trainfile.write(logstring)

    def process_training_data(self):
        print("Training data : Processing training records ..")
        compound_array = np.load("{}".format(self.training_dataset_path))
        for records in compound_array:
            self.training_set.append(records[0])
            self.training_labels.append(self.labels.index(records[1]))
            self.training_records += 1
        self.training_set = np.asarray(self.training_set)
        self.training_labels = np.asarray(self.training_labels)
        print("Training data : Processed {} records".format(self.training_records))
        print("Training data values of shape {} and labels of shape {} is generated".format(self.training_set.shape,self.training_labels.shape))

    def process_testing_data(self):
        print("Testing data : Processing testing records ..")
        compound_array = np.load("{}".format(self.testing_dataset_path))
        for records in compound_array:
            self.testing_set.append(records[0])
            self.testing_labels.append(self.labels.index(records[1]))
            self.testing_records += 1
        self.testing_set = np.asarray(self.testing_set)
        self.testing_labels = np.asarray(self.testing_labels)
        print("Testing data : Processed {} records".format(self.training_records))
        print("Testing data values of shape {} and labels of shape {} is generated".format(self.testing_set.shape,self.testing_labels.shape))

    def train_the_model(self):
        sess = self.model.fit(self.training_set, self.training_labels, epochs=self.training_epochs)
        print("\nEvaluating the model : \n")
        loss, accuracy = self.model.evaluate(self.testing_set, self.testing_labels)
        self.loss = loss
        self.accuracy = accuracy
        self.model.save("{}/{}".format(self.trained_directory_name,self.model_name))
        self.log_train_data()

    def train_model_with_data(self):
        self.training_start_time = time.time()
        self.process_training_data()
        self.training_end_time = time.time()
        self.process_testing_data()
        self.train_the_model()

if __name__ == "__main__":
    data_trainer = Data_Trainer()
    data_trainer.train_model_with_data()
