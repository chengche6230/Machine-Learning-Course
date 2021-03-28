"""
Machine Learning Homework 2

"""
import numpy as np
from tqdm import trange

def loadData(train_num, test_num, img_size):
    # Train image
    with open(FILE_PATH + TRAIN_IMAGE_FILE, 'rb') as file:
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
        print("Train image data loaded.")
    
    # Train label
    with open(FILE_PATH + TRAIN_LABEL_FILE, 'rb') as file:
        for i in range(2):
            file.read(4)
        train_label = []
        for k in trange(train_num):
            int_bytes = file.read(1)
            train_label.append(int.from_bytes(int_bytes, 'big'))
        
        print("Train label data loaded.")
            
    # Test image
    with open(FILE_PATH + TEST_IMAGE_FILE , 'rb') as file:
        for k in range(4):
            file.read(4)
        test_img = []
        for k in trange(test_num):
            tmp = np.zeros((img_size, img_size), np.uint8)
            for i in range(img_size):
                for j in range(img_size):
                    int_bytes = file.read(1)
                    tmp[i][j] = int.from_bytes(int_bytes, 'big')
            test_img.append(tmp)
        
        print("Test image data loaded.")
    
    # Test label
    with open(FILE_PATH + TEST_LABEL_FILE, 'rb') as file:
        for i in range(2):
            file.read(4)
        test_label = []
        for k in trange(test_num):
            int_bytes = file.read(1)
            test_label.append(int.from_bytes(int_bytes, 'big'))
        
        print("Test label data loaded.")
    
    return train_img, train_label, test_img, test_label

def printImg(img, img_size):
    for i in range(img_size):
        for j in range(img_size):
            print("*", end=" ") if img[i][j] > 127 else print(".", end=" ")
        print()

#%%

FILE_PATH = "./dataset/"
TRAIN_IMAGE_FILE = "train-images.idx3-ubyte"
TRAIN_LABEL_FILE = "train-labels.idx1-ubyte"
TEST_IMAGE_FILE = "t10k-images.idx3-ubyte"
TEST_LABEL_FILE = "t10k-labels.idx1-ubyte"

train_num = 60000
test_num = 10000
img_size = 28

train_img, train_label, test_img, test_label = loadData(train_num, test_num, img_size)

#%%

for i in range(train_num):
    print("Label:",train_label[i])
    printImg(train_img[i], img_size)
    input()