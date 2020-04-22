#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 10:29:31 2020

Prepares data in classes

@author: odrec
"""
from glob import glob
from os import path
import matplotlib.pyplot as plt
import numpy as np
from tika import parser
import json, random

def find_ext(dr, ext):
    return glob(path.join(dr,"*.{}".format(ext)))

def plot_bar_x(unique_classes, classes_counts):
    # this is for plotting purpose
    index = np.arange(len(unique_classes))
    plt.bar(index, classes_counts)
    plt.xlabel('Classes', fontsize=5)
    plt.ylabel('Samples in each class', fontsize=5)
    plt.xticks(index, unique_classes, fontsize=5, rotation=30)
    plt.title('Classes and their amount of samples')
    plt.show()
    
def get_train_test_split(data, split=0.8):
    num_training_samples = int(len(data) * 0.8)
    random_keys = random.sample(list(data), num_training_samples)
    all_keys = list(data.keys())
    training_data = {}
    testing_data = {}
    for key in all_keys:
        if key in random_keys:
            training_data[key] = data[key]
        else: testing_data[key] = data[key]
    return training_data, testing_data
    
def load_extracted_data(path_to_extracted_data='extracted_data/'):
    data_file = open(path_to_extracted_data+'saved_data.json')
    data_str = data_file.read()
    data = json.loads(data_str)
    return data

def extract_data(path_to_data='files/', path_to_extracted_data='extracted_data/'):
    pdf_files = find_ext(path_to_data, 'pdf')
    data = {}
    classes = []
    for i,f in enumerate(pdf_files):
        pdf_name = pdf_files[i].split('/')[1]
        pdf_name = pdf_name.split('.')[0]
        data[pdf_name] = {}
        raw = parser.from_file(f)
        raw_content = raw['content']
        raw = raw_content.replace("\n"," ")
        raw = raw_content.replace("\t","    ")
        data[pdf_name]['content'] = raw_content
        
        tmp = pdf_files[i].split('_')[-1]
        filling_type = tmp.split('.')[0]
        #remove all unnecessary characters
        filling_type = filling_type.replace("-","")
        filling_type = filling_type.replace(" ","")
        data[pdf_name]['label'] = filling_type
        classes.append(filling_type)
        
    with open(path_to_extracted_data+'saved_data.json', 'w+') as fp:
        json.dump(data, fp)
    
    unique_classes = set(classes)
    
    classes_counts = []
    for uc in unique_classes:
        classes_counts.append(classes.count(uc))
        
    #Visualize data differences
    plot_bar_x(unique_classes, classes_counts)
    

if __name__ == "__main__":
    extract_data()