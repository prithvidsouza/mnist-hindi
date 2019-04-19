from tkinter import *
import matplotlib.pyplot as plt
from PIL import Image
import cv2 as opencv
import numpy
import time
import os

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

    def draw(self,event):
        self.canvas.create_oval(event.x, event.y, event.x + self.thickness, event.y + self.thickness, fill="black")

    def clear_canvas(self):
        print("Clearing canvas")

    def convert_image(self, pngimg):
        loadedimg = opencv.cvtColor(pngimg, opencv.COLOR_BGR2GRAY)
        img = 255 - loadedimg
        contours,hierarchy = opencv.findContours(img, opencv.RETR_EXTERNAL, opencv.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            xe,ye,we,he = opencv.boundingRect(cnt)
            print(xe,ye,we,he)
            crop_img = loadedimg[ye:ye+he, xe:xe+we]
            resized_img = opencv.resize(crop_img, (28,28))
            print(resized_img.shape)
            plt.imshow(resized_img, cmap="gray", interpolation="nearest")
            plt.show()

    def get_image(self):
        if not os.path.isdir(self.draw_path):
            os.mkdir(self.draw_path)
        timestamp = time.time()
        name = "{}/drawnimg.{}".format(self.draw_path,timestamp)
        self.canvas.postscript(file="{}.eps".format(name),colormode="color")
        img = Image.open("{}.eps".format(name))
        img.save("{}.png".format(name),"png")
        os.remove("{}.eps".format(name))
        print("{}.png".format(name))
        loadedimg = opencv.imread("{}.png".format(name))
        os.remove("{}.png".format(name))
        # print(type(loadedimg), loadedimg.shape)
        self.convert_image(loadedimg)

    def start(self):
        self.window.mainloop()

window = MainWindow(400,400,"Hindi MNIST")
window.start()
