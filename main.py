import csv
import cv2 as opencv
import numpy as np
import os


def clean_data(labels_path, labels_name):
    with open("%s/%s" % (labels_path, labels_name)) as labels:
        content = [value.replace(",,,", "")
                   for value in labels.read().split("\n") if value != ",,,"]
        clean_data = {}
        clean_data["labels"] = content[1]
        clean_data["numbers"] = content[2:12]
        clean_data["vowels"] = content[14:26]
        clean_data["consonants"] = content[28:64]
        return clean_data


labels_path = 'devanagari-character-dataset'
labels_name = 'labels.csv'
newlist = clean_data(labels_path, labels_name)

consonent_path = 'devanagari-character-dataset/nhcd/consonants'
numerals_path = 'devanagari-character-dataset/nhcd/numerals'
vowels_path = 'devanagari-character-dataset/nhcd/vowels'

# Just looking at Numbers
directories = sorted(os.listdir(numerals_path))
files = sorted(os.listdir("%s/%s" % (numerals_path, directories[0])))
print(directories, files)

# Loading the image and grayscaling to reduce 3 channels to 1
loadedimg = opencv.imread("%s/%s/%s" %
                          (numerals_path, directories[0], files[0]))
img = opencv.cvtColor(loadedimg, opencv.COLOR_BGR2GRAY)
height, width = img.shape
