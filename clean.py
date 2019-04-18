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

consonent_path = 'data/devanagari-character-dataset/nhcd/consonants'
numerals_path = 'data/devanagari-character-dataset/nhcd/numerals'
vowels_path = 'data/devanagari-character-dataset/nhcd/vowels'
labels_path = 'data/devanagari-character-dataset'
labels_name = 'labels.csv'

def convert_to_npy():
    newlist = clean_data(labels_path, labels_name)
    numeral_value = []
    numeral_label = []
    directories = sorted(os.listdir(numerals_path))
    for current_dir in directories:
        files = sorted(os.listdir("%s/%s" %
                                  (numerals_path, current_dir)))
        for image in files:
            print("reading : %s/%s/%s" %
                  (numerals_path, current_dir, image))
            loadedimg = opencv.imread("%s/%s/%s" %
                                      (numerals_path, current_dir, image))
            img = opencv.cvtColor(loadedimg, opencv.COLOR_BGR2GRAY)
            numeral_value.append(img)
            numeral_label.append(int(current_dir))
    np.save("data/numeral_value.npy", numeral_value)
    np.save("data/numeral_label.npy", numeral_label)

convert_to_npy()
