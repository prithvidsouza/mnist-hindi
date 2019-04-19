import cv2 as opencv
import os
from PIL import Image
import time

def save_image(path, canvas):
    if not os.path.isdir(path):
        os.mkdir(path)
    timestamp = time.time()
    name = "{}/drawnimg.{}".format(path, timestamp)
    canvas.postscript(file="{}.eps".format(name),colormode="color")
    img = Image.open("{}.eps".format(name))
    img.save("{}.png".format(name),"png")
    # os.remove("{}.eps".format(name))
    print("Saved canvas as : {}.png".format(name))
    return name + ".png"

def convert_image(path, canvas):
    filename = save_image(path, canvas)
    loadedimg = opencv.imread("{}".format(filename))
    os.remove("{}".format(filename)) # This deletes the png image
    loadedimg = opencv.cvtColor(loadedimg, opencv.COLOR_BGR2GRAY)
    img = 255 - loadedimg # Inverting values
    contours,hierarchy = opencv.findContours(img, opencv.RETR_EXTERNAL, opencv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        xe,ye,we,he = opencv.boundingRect(cnt)
        print(xe,ye,we,he)
        crop_img = loadedimg[ye:ye+he, xe:xe+we]
        resized_img = opencv.resize(crop_img, (28,28))
        # plt.imshow(resized_img, cmap="gray", interpolation="nearest")
        # plt.show()
    return resized_img
