from tkinter import *
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as opencv
import numpy
import time
import os

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
print("Using tensorflow :", tf.__version__)

from package.data import get_character
from package.image_processing import convert_image

class MainWindow():
    def __init__(self, width, height, title):
        self.window = Tk()
        self.window.bind("<B1-Motion>", self.draw)
        self.window.geometry("%sx%s"%(width,height))
        self.window.title(title)
        self.canvas = Canvas(self.window,width=width,height=height-50,bg="#fff")
        self.canvas.pack()
        self.predictbtn = Button(self.window,text="Predict",command=self.get_image)
        self.predictbtn.pack()
        self.clearbtn = Button(self.window,text="Clear",command=self.clear_canvas)
        self.clearbtn.pack()
        self.thickness = 8
        self.draw_path = "drawings"
        self.current_model = "n"
        self.model_names = { "n" : ("numeral_model.model","numerals"), "v" : ("vowel_model.model","vowels"), "c" : ("consonant_model.model","consonants") }

    def draw(self,event):
        self.canvas.create_oval(event.x, event.y, event.x + self.thickness, event.y + self.thickness, fill="black")

    def clear_canvas(self):
        self.canvas.delete("all")

    def predict(self,values):
        model = self.model_names[self.current_model][0]
        values = values.reshape(1,28,28)
        print("Loaded Model : ",model)
        saved_model = tf.keras.models.load_model('data/%s'%(model))
        prediction = saved_model.predict(values)
        return prediction

    def get_image(self):
        resized = convert_image(self.draw_path, self.canvas) # Get resized image from the canvas
        predict_arr = self.predict(resized) # Predict values from it
        # plt.plot()
        xvals = [i for i in range(len(predict_arr[0]))]
        print(xvals,predict_arr[0])
        getmax = numpy.argmax(predict_arr) # Get greatest index
        prediction = get_character(getmax,  self.model_names[self.current_model][1])
        print(prediction)
        plt.plot(xvals,predict_arr[0])
        plt.show()

    def start(self):
        self.window.mainloop()

window = MainWindow(400,400,"Hindi MNIST")
window.start()
