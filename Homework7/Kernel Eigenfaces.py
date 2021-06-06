"""
Machine Learning Homework 7
    Kernel Eigenfaces

"""
import os
import numpy as np
from matplotlib import pyplot as plt

subject_num = 15
image_num = 11
train_num  = 9
test_num = 2
expression = ['centerlight', 'glasses', 'happy', 'leftlight',
              'noglasses', 'normal', 'rightlight', 'sad',
              'sleepy', 'surprised', 'wink']

def readPGM(file):
    if not os.path.isfile(file):
        return None
    with open(file, 'rb') as f:
        f.readline() # P5
        f.readline() # Comment line
        width, height = [int(i) for i in f.readline().split()]
        assert int(f.readline()) <= 255 # Depth
        img = np.zeros((height, width))
        for r in range(height):
            for c in range(width):
                img[r][c] = ord(f.read(1))
        return img

def readData():
    train_data = []
    file_path = "./Yale_Face_Database/Training/"
    for sub in range(subject_num):
        for i in range(image_num):
            file_name = f"subject{sub+1:02d}.{expression[i]}.pgm"
            d = readPGM(file_path + file_name)
            if d is not None:
                train_data.append(d)
    
    test_data = []
    file_path = "./Yale_Face_Database/Testing/"
    for sub in range(subject_num):
        for i in range(image_num):
            file_name = f"subject{sub+1:02d}.{expression[i]}.pgm"
            d = readPGM(file_path + file_name)
            if d is not None:
                test_data.append(d)
    print("File read.")
    return train_data, test_data

if __name__ == "__main__":
    train_data, test_data = readData()
    
    # PCA, LDA
    
    # Kernel PCA, LDA
    
    # Different kernels