"""
Machine Learning Homework 6
    Spectral Clustering

"""
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import image
from tqdm import trange

img_length = 100
img_size = 10000
K = 2
gamma_s = 1 / img_size
gamma_c = 1 / (256 * 256)
kernel = None

COLOR_2 = [[153, 102, 51], [0, 153, 255]]

#%%

def readImg(fileName):
    img = cv2.imread('./data/' + fileName)
    return img.reshape(img_size, 3)

def spatialSimi(p1, p2):
    p = (p1 // img_length - p2 // img_length) ** 2
    p += (p1 % img_length - p2 % img_length) ** 2
    return p

def colorSimi(c1, c2):
    c = 0
    for i in range(3):
        c += (c1[i] - c2[i]) ** 2
    return c

def k(x1, x2):
    s = np.exp(-gamma_s * spatialSimi(x1[0], x2[0]))
    c = np.exp(-gamma_c * colorSimi(x1[1], x2[1]))
    return s * c
    
def computeKernel(img):
    kernel = np.zeros((img_size, img_size))
    for p in trange(img_size):
        for q in range(p, img_size):
            kernel[p][q] = kernel[q][p] = k([p, img[p]], [q, img[q]])
    return kernel

#%%
if __name__ == "__main__":
    img = readImg("image1.png")