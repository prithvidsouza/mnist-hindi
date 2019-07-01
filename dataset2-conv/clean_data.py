# Author : Velan Salis
# NMAM Institute of Technology, Nitte
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random
import time
import datetime
import csv
import tensorflow as tf

class Data_Cleaner():
    def __init__(self):
        self.dataset_directory = "DevanagariHandwrittenCharacterDataset"
        self.training_set_path = "data/{}/training_data".format(self.dataset_directory)
        self.testing_set_path = "data/{}/testing_data".format(self.dataset_directory)
        self.training_set_directory = sorted(os.listdir(self.training_set_path))
        self.testing_set_directory = sorted(os.listdir(self.testing_set_path))
        self.output_training_set_name = "training_dataset"
        self.output_testing_set_name = "testing_dataset"
        self.training_instances = 0
        self.testing_instances = 0
        self.reading_start_time = 0
        self.reading_end_time = 0
        self.total_time_elapsed = 0
        self.clean_directory = "processed"
        self.labels = []
        self.create_clean_directory()

    def create_clean_directory(self):
        if not os.path.exists(self.clean_directory):
            os.mkdir(self.clean_directory)

    def add_label(self, label_name):
        if label_name not in self.labels:
            print("Added label {}".format(label_name))
            self.labels.append(label_name)

    def write_labels(self):
        with open("{}/labels.csv".format(self.clean_directory),"w") as csvfile:
            csvptr = csv.writer(csvfile)
            csvptr.writerows([self.labels])

    def log_data_cleaning_stats(self):
        logstring = "\n"
        start_time_in_seconds = datetime.datetime.fromtimestamp(self.reading_start_time).strftime('%M:%S.%f')[:-4]
        end_time_in_seconds = datetime.datetime.fromtimestamp(self.reading_end_time).strftime('%M:%S.%f')[:-4]
        logstring += "Data Cleaning Stats for CLEAN_{} : \n".format(self.reading_start_time)
        logstring += "Process ID : CLEAN_{}\n".format(self.reading_start_time)
        logstring += "Reading started : {}\n".format(start_time_in_seconds)
        logstring += "Reading end : {}\n".format(end_time_in_seconds)
        logstring += "Total time elapsed : {}\n".format(self.total_time_elapsed)
        logstring += "Clean directory : ./{}\n".format(self.clean_directory)
        logstring += "Clean training dataset name : {}.npy\n".format(self.output_training_set_name)
        logstring += "Clean testing dataset name : {}.npy\n".format(self.output_testing_set_name)
        logstring += "Total instances : {}\n".format(self.training_instances + self.testing_instances)
        logstring += "Total training instances : {}\n".format(self.training_instances)
        logstring += "Total testing instances : {}\n".format(self.testing_instances)
        with open("{}/clean_logfile.txt".format(self.clean_directory),"a") as file:
            file.write(logstring)
        print(logstring)

    def generate_training_data(self):
        numpy_array_list = []
        for character_dir_pointer in self.training_set_directory:
            current_directory = "{}/{}".format(self.training_set_path,character_dir_pointer)
            image_file_list = os.listdir(current_directory)
            for image_file_pointer in image_file_list:
                print("Reading {}/{}".format(current_directory,image_file_pointer))
                image = cv2.imread(current_directory + "/" + image_file_pointer)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                label = character_dir_pointer.split("_")
                if len(label) == 2:
                    label = label[1]
                else:
                    label = label[2]
                image = tf.keras.utils.normalize(image, axis=1)
                self.training_instances += 1
                numpy_array_list.append((image,label))
                self.add_label(label)
        random.shuffle(numpy_array_list)
        np.save("{}/{}".format(self.clean_directory,self.output_training_set_name), numpy_array_list)

    def generate_testing_data(self):
        numpy_array_list = []
        for character_dir_pointer in self.testing_set_directory:
            current_directory = "{}/{}".format(self.testing_set_path,character_dir_pointer)
            image_file_list = os.listdir(current_directory)
            for image_file_pointer in image_file_list:
                print("Reading {}/{}".format(current_directory,image_file_pointer))
                image = cv2.imread(current_directory + "/" + image_file_pointer)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                label = character_dir_pointer.split("_")
                if len(label) == 2:
                    label = label[1]
                else:
                    label = label[2]
                image = tf.keras.utils.normalize(image, axis=1)
                self.testing_instances += 1
                numpy_array_list.append((image,label))
        random.shuffle(numpy_array_list)
        np.save("{}/{}".format(self.clean_directory,self.output_testing_set_name), numpy_array_list)

    def clean_and_convert_data(self):
        if not os.path.exists("data/{}".format(self.dataset_directory)):
            print("No dataset found in directory ./data for conversion and cleaning")
            print("Download the dataset from the following link : http://archive.ics.uci.edu/ml/machine-learning-databases/00389/")
            print("Place it in the ./data folder. (if not exists, create one)")
            print("Then try running the command again.")
            return
        else:
            self.reading_start_time = time.time()
            self.generate_training_data()
            self.generate_testing_data()
            self.reading_end_time = time.time()
            self.total_time_elapsed = self.reading_end_time - self.reading_start_time
            self.log_data_cleaning_stats()
            self.write_labels()

    def check_for_consistency(self):
        numpy_array_list = np.load("{}.npy".format(self.output_training_set_name))
        random_index = random.randint(0,len(numpy_array_list))
        print(numpy_array_list[random_index][0].shape)
        plt.imshow(numpy_array_list[random_index][0])
        print(numpy_array_list[random_index][1])
        plt.show()

# Main Loop
if __name__ == "__main__":
    data_cleaner = Data_Cleaner()
    data_cleaner.clean_and_convert_data()
