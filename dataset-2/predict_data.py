import os
import random
import time

import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from PIL import Image
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

print("Using tensorflow :", tf.__version__)

class MainWindow():
    def __init__(self, width, height, title):
        self.window = Tk()
        self.window.bind("<B1-Motion>", self.draw)
        self.window.geometry("{}x{}".format(width,height))
        self.window.title(title)
        self.canvas = Canvas(self.window,width=width,height=height-50,bg="#fff")
        self.canvas.pack()
        self.predictbtn = Button(self.window,text="Predict",command=self.predict_from_image)
        self.predictbtn.pack()
        self.clearbtn = Button(self.window,text="Clear",command=self.clear_canvas)
        self.clearbtn.pack()
        self.pen_thickness = 8
        self.drawings_path = "drawings"
        self.model_name = "trained_hindi_chars.model"
        self.model = self.get_model()
        self.label = self.get_labels()
        self.create_drawings_path()

    def get_model(self):
        return tf.keras.models.load_model('trained/{}'.format(self.model_name))

    def get_labels(self):
        with open("processed/labels.csv","r") as label:
            content = label.read().replace("\n","")
            content = content.split(",")
            return content

    def create_drawings_path(self):
        if not os.path.exists(self.drawings_path):
            os.mkdir(self.drawings_path)

    def draw(self, event):
        self.canvas.create_oval(event.x, event.y, event.x + self.pen_thickness, event.y + self.pen_thickness, fill="black")

    def clear_canvas(self):
        self.canvas.delete("all")

    def save_image(self):
        timestamp = time.time()
        name = "{}/drawnimg.{}".format(self.drawings_path, timestamp)
        self.canvas.postscript(file="{}.eps".format(name),colormode="color")
        img = Image.open("{}.eps".format(name))
        img.save("{}.png".format(name),"png")
        os.remove("{}.eps".format(name))
        print("Saved canvas as : {}.png".format(name))
        return name + ".png"

    def convert_image(self):
        filename = self.save_image()
        loadedimg = cv2.imread("{}".format(filename))
        os.remove("{}".format(filename))
        loadedimg = cv2.cvtColor(loadedimg, cv2.COLOR_BGR2GRAY)
        img = 255 - loadedimg
        contours,hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            xe,ye,we,he = cv2.boundingRect(cnt)
            print(xe,ye,we,he)
            crop_img = img[ye:ye+he, xe:xe+we]
            resized_img = cv2.resize(crop_img, (28,28))
            resized_img = np.pad(resized_img,((2,2),(2,2)), 'constant')
        return resized_img

    def predict_from_image(self):
        img = self.convert_image()
        predict_data = np.asarray([img])
        prediction = self.model.predict(predict_data)
        index = np.argmax(prediction)
        print(self.label[index])
        img_to_save = Image.fromarray(img)
        img_to_save.save("drawings/predict_{}.png".format(self.label[index]))
        plt.imshow(img)
        plt.show()

if __name__ == "__main__":
    mainwindow = MainWindow(400,400,"Devnagari")
    mainwindow.window.mainloop()
