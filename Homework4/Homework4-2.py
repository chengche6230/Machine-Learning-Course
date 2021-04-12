"""
Machine Learning Homework 4-2
    EM Algorithm
    
"""
import numpy as np
from tqdm import trange

def loadData(train_num, test_num, img_size):
    FILE_PATH = "./4-2 MNIST Dataset/"
    TRAIN_IMAGE_FILE = "train-images.idx3-ubyte"
    TRAIN_LABEL_FILE = "train-labels.idx1-ubyte"
    TEST_IMAGE_FILE = "t10k-images.idx3-ubyte"
    TEST_LABEL_FILE = "t10k-labels.idx1-ubyte"
    
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
    
    # Train label
    with open(FILE_PATH + TRAIN_LABEL_FILE, 'rb') as file:
        for i in range(2):
            file.read(4)
        train_label = []
        for k in trange(train_num):
            int_bytes = file.read(1)
            train_label.append(int.from_bytes(int_bytes, 'big'))
            
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
        
    # Test label
    with open(FILE_PATH + TEST_LABEL_FILE, 'rb') as file:
        for i in range(2):
            file.read(4)
        test_label = []
        for k in trange(test_num):
            int_bytes = file.read(1)
            test_label.append(int.from_bytes(int_bytes, 'big'))
        
    return train_img, train_label, test_img, test_label

if __name__ == "__main__":
    train_num = 60000
    test_num = 10000
    img_size = 28
    
    train_img, train_label, test_img, test_label = loadData(train_num, test_num, img_size)
    print("\nData loaded.")