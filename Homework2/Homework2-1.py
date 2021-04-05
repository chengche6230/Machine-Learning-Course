"""
Machine Learning Homework 2-1

"""
import numpy as np
import math
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

def printImg(img, img_size):
    for i in range(img_size):
        for j in range(img_size):
            print("*", end=" ") if img[i][j] > 128 else print(".", end=" ")
        print()

def printPredImg_DIS(px, img_size):
    for n in range(10):
        print(n, ":")
        for i in range(img_size):
            for j in range(img_size):
                pre = np.argmax(px[n][i*img_size + j])
                print("*", end=" ") if pre > 16 else print(".", end=" ")
            print()
        print()

def printPredImg_CON(mean, img_size):
    for n in range(10):
        print(n, ":")
        for i in range(img_size):
            for j in range(img_size):
               print("*", end=" ") if mean[n][i * img_size + j] > 128 else print(".", end=" ")
            print()
        print()
        
def output(post, pred, label):
    print("Posterior (in log scale):")
    for n in range(10):
        print(n, ": ", post[n], sep="")
    print("Prediction:", pred, ", Ans:", label)
    print()

#%%

FILE_PATH = "./dataset/"
TRAIN_IMAGE_FILE = "train-images.idx3-ubyte"
TRAIN_LABEL_FILE = "train-labels.idx1-ubyte"
TEST_IMAGE_FILE = "t10k-images.idx3-ubyte"
TEST_LABEL_FILE = "t10k-labels.idx1-ubyte"
PI = 3.1415926

train_num = 60000
test_num = 10000
img_size = 28

train_img, train_label, test_img, test_label = loadData(train_num, test_num, img_size)
print("\nData loaded.")

#%%
# Discrete Mode

# Init.
count = np.zeros(10, np.uint32)
px = np.ones((10, img_size*img_size, 32), np.uint32) # [0~9, 0~28x28, 0~32]

# Train classifier
for k in trange(train_num):
    num = train_label[k]
    count[num] += 1
    for i in range(img_size):
        for j in range(img_size):
            px[num][i*img_size + j][train_img[k][i][j]//8] += 1
count = count / train_num

# Test classifier
err = 0
for k in trange(test_num):
    post = np.zeros(10)
    for n in range(10):
        post[n] = math.log(count[n])
        for i in range(img_size):
            for j in range(img_size):
                x = px[n][i*img_size + j][test_img[k][i][j]//8]
                post[n] += math.log(x/count[n])
    post = post / sum(post)
    pred = np.argmax(post)
    output(post, pred, test_label[k])
    if pred != test_label[k]:
        err += 1
    
print("Error rate:", err / test_num)

print("Imagination of numbers in Bayesian classifier:\n")
printPredImg_DIS(px, img_size)

#%%
# Continous Mode

count2 = np.zeros(10, np.uint32)
mean = np.zeros((10, img_size*img_size))
var = np.zeros((10, img_size*img_size))

# Find every mean and var
for k in trange(train_num):
    num = train_label[k]
    count2[num] += 1
    for i in range(img_size):
        for j in range(img_size):
            mean[num][i*img_size + j] += train_img[k][i][j]
for n in range(10):
    mean[n] = mean[n] / count2[n]
    
for k in trange(train_num):
    num = train_label[k]
    for i in range(img_size):
        for j in range(img_size):
            index = i * img_size + j
            var[num][index] += (train_img[k][i][j] - mean[num][index])**2
for n in range(10):
    var[n] = var[n] / count2[n]
# Prior
count2 = count2 / train_num

# Test classifier
err2 = 0
for k in trange(test_num):
    post = np.zeros(10)
    for n in range(10):
        post[n] = math.log(count2[n])
        for i in range(img_size):
            for j in range(img_size):
                index = i * img_size + j
                var[n][index] = var[n][index] if var[n][index]!=0 else 1e-5
                post[n] -= 0.5 * math.log(2 * PI * var[n][index])
                post[n] -= 0.5 * ((test_img[k][i][j] - mean[n][index])**2 / var[n][index])
    #post = -post #negative log
    #post = post / sum(post)
    pred = np.argmax(post)
    #output(post, pred, test_label[k])
    if pred != test_label[k]:
        err2 += 1
    
print("Error rate:", err2 / test_num)

print("Imagination of numbers in Bayesian classifier:\n")
printPredImg_CON(mean, img_size)

#%%
#Testttt
for n in range(10):
    print(n, ":")
    for i in range(img_size):
        for j in range(img_size):
            pre = np.argmax(px[n][i*img_size + j])
            print("*", end=" ") if pre > 16 else print(".", end=" ")
        print()
    print()
    print("=============================================================")
    for i in range(img_size):
        for j in range(img_size):
            print("*", end=" ") if mean[n][i * img_size + j] > 128 else print(".", end=" ")
        print()
    print()
    input()