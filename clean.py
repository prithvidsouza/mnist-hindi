import csv
import cv2 as opencv
import numpy as np
import os

pathinfo = dict()
pathinfo["consonent_path"] = 'data/devanagari-character-dataset/nhcd/consonants'
pathinfo["numerals_path"] = 'data/devanagari-character-dataset/nhcd/numerals'
pathinfo["vowels_path"] = 'data/devanagari-character-dataset/nhcd/vowels'
pathinfo["labels_path"] = 'data/devanagari-character-dataset/labels.csv'

def clean_data(labels_path):
    '''
    This method will import labels.csv from the dataset and retreive the clean label classes
    and return them as a dictionary
    '''
    with open("%s" % (labels_path)) as labels:
        content = [value.replace(",,,", "")
                   for value in labels.read().split("\n") if value != ",,,"]
        clean_data = {}
        clean_data["labels"] = content[1]
        clean_data["numbers"] = content[2:12]
        clean_data["vowels"] = content[14:26]
        clean_data["consonants"] = content[28:64]
        return clean_data

def convert_to_npy():
    '''
    This mthod will scan the given directory and looks for images
    then it in turn converts it to numpy arrays as labels and values and stores them
    '''
    newlist = clean_data(pathinfo['labels_path'])
    value = []
    label = []
    directories = sorted(os.listdir(pathinfo['numerals_path']))
    for current_dir in directories:
        files = sorted(os.listdir("%s/%s" % (pathinfo['numerals_path'], current_dir)))
        for image in files:
            print("reading : %s/%s/%s" % (pathinfo['numerals_path'], current_dir, image))
            loadedimg = opencv.imread("%s/%s/%s" % (pathinfo['numerals_path'], current_dir, image))
            img = opencv.cvtColor(loadedimg, opencv.COLOR_BGR2GRAY)
            value.append(img)
            label.append(int(current_dir))
    np.save("data/numeral_value.npy", value)
    np.save("data/numeral_label.npy", label)

convert_to_npy()
