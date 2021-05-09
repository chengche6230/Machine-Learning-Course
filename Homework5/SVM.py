"""
Machine Learning Homework 5
    SVM

"""
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

train_num = 5000
test_num = 2500
img_size = 28
img_length = img_size * img_size

def loadData(filepath):
    with open(filepath + 'X_train.csv', 'r') as file:
        train_img = []
        for k in range(train_num):
            train_img.append(np.array([float(i) for i in file.readline().strip().split(',')]))
    with open(filepath + 'Y_train.csv', 'r') as file:
        train_label = []
        for k in range(train_num):
            train_label.append([float(i) for i in file.readline().strip()])
    with open(filepath + 'X_test.csv', 'r') as file:
        test_img = []
        for k in range(test_num):
            test_img.append(np.array([float(i) for i in file.readline().strip().split(',')]))
    with open(filepath + 'Y_test.csv', 'r') as file:
        test_label = []
        for k in range(test_num):
            test_label.append([float(i) for i in file.readline().strip()])
    return train_img, train_label, test_img, test_label

def printImg(img):
    for i in range(img_size):
        for j in range(img_size):
            index = i * img_size + j
            print('. ', end='') if img[index] < 0.5 else print('* ', end='')
        print()

if __name__ == "__main__":
    train_img, train_label, test_img, test_label = loadData('./data/')
    
    for i in range(train_num):
        printImg(train_img[i])
        input()