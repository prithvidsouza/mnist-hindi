import csv
import cv2 as opencv
import numpy as np
import os

pathinfo = dict()
pathinfo["consonent_path"] = 'data/devanagari-character-dataset/nhcd/consonants'
pathinfo["numerals_path"] = 'data/devanagari-character-dataset/nhcd/numerals'
pathinfo["vowels_path"] = 'data/devanagari-character-dataset/nhcd/vowels'

def convert_to_npy(current_selection,valuename,labelname):
    '''
    This mthod will scan the given directory and looks for images
    then it in turn converts it to numpy arrays as labels and values and stores them
    '''
    value = []
    label = []
    directories = sorted(os.listdir(current_selection))
    for current_dir in directories:
        files = sorted(os.listdir("%s/%s" % (current_selection, current_dir)))
        for image in files:
            print("reading : %s/%s/%s" % (current_selection, current_dir, image))
            loadedimg = opencv.imread("%s/%s/%s" % (current_selection, current_dir, image))
            img = opencv.cvtColor(loadedimg, opencv.COLOR_BGR2GRAY)
            value.append(img)
            label.append(int(current_dir))
    np.save("data/%s.npy"%(valuename), value)
    np.save("data/%s.npy"%(labelname), label)

convert_to_npy(pathinfo['numerals_path'],'numeral_value',"numeral_label") # Cleaning the numeral data
convert_to_npy(pathinfo['consonent_path'],'consonant_value',"consonant_label") # Cleaning the consonant data
convert_to_npy(pathinfo['vowels_path'],'vowel_value',"vowel_label") # Cleaning the vowel data
