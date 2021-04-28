"""
Machine Learning Homework 4-2
    EM Algorithm
    
"""
import numpy as np
from numpy import matmul as mul
from tqdm import trange

def loadData(train_num, test_num, img_size):
    FILE_PATH = "./4-2 MNIST Dataset/"
    TRAIN_IMAGE_FILE = "train-images.idx3-ubyte"
    TRAIN_LABEL_FILE = "train-labels.idx1-ubyte"
    #TEST_IMAGE_FILE = "t10k-images.idx3-ubyte"
    #TEST_LABEL_FILE = "t10k-labels.idx1-ubyte"
    
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
    """       
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
    """  
    return train_img, train_label

def dataPreprocess(img, train_num, img_size):
    tmp_img = np.zeros((train_num, img_size, img_size))
    for n in trange(train_num):
        for i in range(img_size):
            for j in range(img_size):
                tmp_img[n][i][j] = 1 if img[n][i][j] > 127 else 0
    return tmp_img
    
def printImg(img, img_size):
    for i in range(img_size):
        for j in range(img_size):
            print("*", end=" ") if img[i][j] > 0 else print(".", end=" ")
        print()

def assignLabel(train_label, train_num, w):
    mapping = np.zeros((10), dtype=np.uint32)
    counting = np.zeros((10, 10), dtype=np.uint32)
    for k in range(train_num):
        counting[train_label[k]][np.argmax(w[k])] += 1
    for n in range(10):
        index = np.argmax(counting) # return a 0~99 value
        label = index // 10
        _class = index % 10
        mapping[label] = _class
        counting[:,_class] = 0
        counting[label,:] = 0
    return mapping

def printImagination(p, img_size, mapping, labeled=False):
    for n in range(10):
        if labeled:
            print("labeled", end=" ")
        print("class %d:" % n)
        real_label = mapping[n]
        for i in range(img_size):
            for j in range(img_size):
                print("*", end=" ") if p[real_label][i][j] > 0.5 else print(".", end=" ")
            print()
        print()
        
def printResult(train_label, train_num, mapping, w, _iter):
    err = train_num
    tb = np.zeros((10, 2, 2), dtype=np.uint32)
    
    mapping_inv = np.zeros((10), dtype=np.int32)# idx->value = cluster->label
    for i in range(10):
        mapping_inv[i] = np.where(mapping == i)[0]
        
    for k in range(train_num):
        pred = mapping_inv[np.argmax(w[k])]
        truth = train_label[k]
        for n in range(10):
            tb[n][0][0] = tb[n][0][0] + 1 if truth==n and pred==n else tb[n][0][0] #TP
            tb[n][0][1] = tb[n][0][1] + 1 if truth==n and pred!=n else tb[n][0][1] #FN
            tb[n][1][0] = tb[n][1][0] + 1 if truth!=n and pred==n else tb[n][1][0] #FP
            tb[n][1][1] = tb[n][1][1] + 1 if truth!=n and pred!=n else tb[n][1][1] #TN
        
    for n in range(10):
        print("--------------------------------------------------------")
        print(f"Confusion Matrix {n}:")
        print(f"\t\t\tPredict {n}\tPredict not {n}")
        print(f"Is {n}\t\t\t{tb[n][0][0]}\t\t\t{tb[n][0][1]}")
        print(f"Isn't {n}\t\t\t{tb[n][1][0]}\t\t{tb[n][1][1]}")
        sens = tb[n][0][0] / (tb[n][0][0] + tb[n][0][1])
        spec = tb[n][1][1] / (tb[n][1][0] + tb[n][1][1])
        print(f"\nSensitivity (Successfully predict number {n})\t: {sens}")
        print(f"Specificity (Successfully predict not number {n}): {spec}")
        err -= tb[n][0][0]
    
    print("--------------------------------------------------------")
    print(f"Total iteration to converge: {_iter}")
    print(f"Total error rate: {err/train_num}")

if __name__ == "__main__":
    train_num = 60000
    test_num = 10000
    img_size = 28
    
    train_img, train_label = loadData(train_num, test_num, img_size)
    print("\nData loaded.")
    
    img = dataPreprocess(train_img, train_num, img_size)
    
    lam = np.full((10), 1/10) # chance to be 0~9
    p = np.zeros((10, img_size, img_size))
    last_p  = np.zeros((10, img_size, img_size))
    for n in range(10):
        for i in range(img_size):
            for j in range(img_size):
                # Initial p within range 0.25~0.75
                p[n][i][j] = np.random.rand()/2 + 0.25
    
    w = np.zeros((train_num, 10))
    mapping = np.array([i for i in range(10)], dtype=np.uint32)
    max_iter = 10
    conv_thres = 30
    _iter = 0
    while _iter < max_iter:
        _iter += 1
        w = np.zeros((train_num, 10))
        
        # E step, find w0~w9 of every image (responsibility)
        for k in trange(train_num):
            for n in range(10):
                w[k][n] = lam[n]
                w[k][n] *= np.prod(p[n] ** img[k] )
                w[k][n] *= np.prod((1-p[n]) ** (1-img[k]))
            w[k] = w[k]/sum(w[k])
            
        # M step
        for n in trange(10):
            # Update lambda
            lam[n] = sum(w[:, n]) / train_num
            
            # Update every p
            wn = w[:, n].reshape(train_num, 1)
            for i in range(img_size):
                for j in range(img_size):
                    xd = img[:, i, j].reshape(train_num, 1) # d means i*28+j
                    p[n][i][j] = mul(wn.T, xd) / sum(w[:, n])
                    p[n][i][j] = 1e-5 if p[n][i][j]==0 else p[n][i][j]
        
        printImagination(p, img_size, mapping)
        delta = sum(sum(sum(abs(p - last_p))))
        print(f"No. of Iteration: {_iter}, Difference: {delta}\n")
        print("--------------------------------------------------------")
        last_p = p.copy()
        if delta < conv_thres:
            break
        
    print("--------------------------------------------------------")
    mapping = assignLabel(train_label, train_num, w)
    printImagination(p, img_size, mapping, labeled=True)
    printResult(train_label, train_num, mapping, w, _iter)
    
    