# System Imports
import random, os, sys, time, datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Library Imports
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf

def get_data():
    training_dataset = np.load("./processed/training_dataset.npy", allow_pickle=True)
    testing_dataset = np.load("./processed/testing_dataset.npy", allow_pickle=True)
    labels = open('./processed/labels.csv', 'r').read().replace('\n','').split(',')    
    training_values = [element[0] for element in training_dataset]
    training_labels = [labels.index(element[1]) for element in training_dataset]
    testing_values = [element[0] for element in testing_dataset]
    testing_labels = [labels.index(element[1]) for element in testing_dataset]
    return training_values, training_labels, testing_values, testing_labels

def get_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 1)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))
    model.add(tf.keras.layers.Dense(46, activation='softmax'))
    return model

def train_data():
    model = get_model()
    train_images, train_labels, test_images, test_labels = get_data()
    train_images = np.reshape(train_images, (78200, 32, 32, 1))
    test_images = np.reshape(test_images, (13800, 32, 32, 1))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    model.summary()
    
    model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Accuracy : {}'.format(test_acc))
    print('Loss : {}'.format(test_loss))

if __name__ == "__main__":
    train_data()