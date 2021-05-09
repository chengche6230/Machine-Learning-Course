"""
Machine Learning Homework 5
    SVM

"""
import numpy as np
import matplotlib.pyplot as plt

def loadData(filename):
     with open(filename, 'r') as file:
        for k in range(4):
            file.read(4)
        train_img = []
        for k in trange(train_num):
            tmp = np.zeros((img_size, img_size), np.uint8)
            for i in range(img_size):
                for j in range(img_size):
                    int_bytes = file.read(1)
                    tmp[i][j] = int.from_bytes(int_bytes, 'big')
            train_img.append(tmp)

if __name__ == "__main__":
    1