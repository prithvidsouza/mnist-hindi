# System Imports
import random, os, sys, time, datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Library Imports
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import tensorflow.keras as keras
import tensorflow as tf

def draw_graph(history, epochs):
    # summarize history for accuracy
    plt.style.use('seaborn-whitegrid')
    
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.yticks(np.arange(0, 1.05, step=0.05))
    plt.xticks(np.arange(0, epochs, step=1))
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('train-test.png', bbox_inches='tight')
    plt.close()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.yticks(np.arange(0, 1.05, step=0.05))
    plt.xticks(np.arange(0, epochs, step=1))
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('train-loss.png', bbox_inches='tight')
    plt.close()

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
    # model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(46, activation='softmax'))
    return model

def train_data():
    epochs = 10
    model = get_model()
    train_images, train_labels, test_images, test_labels = get_data()
    train_images = np.reshape(train_images, (78200, 32, 32, 1))
    test_images = np.reshape(test_images, (13800, 32, 32, 1))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    model.summary()
    
    history = model.fit(train_images, train_labels, epochs=epochs, validation_data=(test_images, test_labels))
    draw_graph(history, epochs)
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print('Accuracy : {}'.format(test_acc))
    print('Loss : {}'.format(test_loss))

if __name__ == "__main__":
    train_data()