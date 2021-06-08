"""
Machine Learning Homework 7
    Kernel Eigenfaces

"""
import os
import numpy as np
from matplotlib import pyplot as plt
from tqdm import trange

subject_num = 15
image_num = 11
train_num = 9
test_num = 2
height, width = 231, 195
D = height * width
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
        return img.reshape(-1)

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
    return np.array(train_data), np.array(test_data)

def PCA(data, k=25):
    mean = np.mean(data, axis=0)
    cov = (data - mean) @ (data - mean).T
    eigenvalue, eigenvector = np.linalg.eig(cov)
    eigenvector = data.T @ eigenvector
    
    # Normalize w
    for i in range(len(eigenvector[0])):
        eigenvector[:,i] = eigenvector[:,i] / np.linalg.norm(eigenvector[:,i])
        
    # Seclect first k largest eigenvalues
    eigenindex = np.argsort(-eigenvalue)
    eigenvector = eigenvector[:, eigenindex]
    
    W = eigenvector[:, :k].real
    
    return W, mean

def imageCompression(data, S):
    d = np.zeros((len(data), height//S, width//S))
    for n in range(len(data)):
        d[n] = np.full((height//S, width//S), np.mean(data[n]))
        img = data[n].reshape(height, width)
        for i in range(0, height - S + 1, S):
            for j in range(0, width - S + 1, S):
                tmp = 0
                for r in range(S):
                    for c in range(S):
                        tmp += img[i + r][j + c]
                d[n][i//S][j//S] = tmp // (S**2)
    return d.reshape(len(data),-1)

def LDA(data, k=25, S=1):
    mean = np.mean(data, axis=0)
    # Sw, Sb
    Sw = np.zeros((len(data[0]), len(data[0])), dtype=np.float32)
    Sb = np.zeros((len(data[0]), len(data[0])), dtype=np.float32)
    for sub in trange(subject_num):
        xi = data[sub * train_num : (sub + 1) * train_num]
        mj = np.mean(xi, axis=0)
        Sw += (xi - mj).T @ (xi - mj)
        Sb += len(xi) * (mj - mean).reshape(-1, 1) @ (mj - mean).reshape(1, -1)
    
    # Pseudo inv.
    eigenvalue, eigenvector = np.linalg.eig(np.linalg.pinv(Sw) @ Sb)
    
    # Normalize w
    for i in range(len(eigenvector[0])):
        eigenvector[:,i] = eigenvector[:,i] / np.linalg.norm(eigenvector[:,i])
        
    # Seclect first k largest eigenvalues
    eigenindex = np.argsort(-eigenvalue)
    eigenvector = eigenvector[:, eigenindex]
    
    W = eigenvector[:, :k].real
    
    return W

def eigenFace(W, file_path, k=25, S=1):
    fig = plt.figure()
    for i in range(k):
        img = W[:,i].reshape(height//S, width//S)
        plt.imshow(img, cmap='gray')
        fig.savefig(f'{file_path}eigenface_{i:02d}.jpg')

    fig = plt.figure(figsize=(12, 9))
    for i in range(k):
        img = W[:,i].reshape(height//S, width//S)
        row = int(np.sqrt(k))
        ax = fig.add_subplot(row, row, i + 1)
        ax.imshow(img, cmap='gray')
    fig.savefig(f'{file_path}../eigenfaces_{k}.jpg')
    plt.show()

def reconstructFace(W, mean, data, file_path, S=1):
    if mean is None:
        mean = np.zeros(W.shape[0])
    
    sel = np.random.choice(subject_num * train_num, 10, replace=False)
    img = []
    for index in sel:
        x = data[index].reshape(1, -1)
        reconstruct = (x - mean) @ W @ W.T + mean
        img.append(reconstruct.reshape(height//S, width//S))
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        ax.imshow(x.reshape(height//S, width//S), cmap='gray')
        ax = fig.add_subplot(1, 2, 2)
        ax.imshow(reconstruct.reshape(height//S, width//S), cmap='gray')
        fig.savefig(f'{file_path}reconfaces_{len(img)}.jpg')
    
    fig = plt.figure(figsize=(10, 4))
    for i in range(len(img)):
        ax = fig.add_subplot(2, 5, i + 1)
        ax.imshow(img[i], cmap='gray')
    fig.savefig(f'{file_path}../reconfaces.jpg')
    plt.show()

def distance(test, train_data):
    dist = np.zeros(len(train_data), dtype=np.float32)
    for j in range(len(train_data)):
        dist[j] = np.sum((test - train_data[j]) ** 2)
    return dist

def faceRecongnition(W, mean, train_data, test_data, K):
    if mean is None:
        mean = np.zeros(W.shape[0])

    err = 0
    low_train = (train_data - mean) @ W
    low_test = (test_data - mean) @ W
    for i in range(len(low_test)):
        vote = np.zeros(subject_num, dtype=int)
        dist = distance(low_test[i], low_train)
        nearest = np.argsort(dist)[:K]
        for n in nearest:
            vote[n // train_num] += 1
        predict = np.argmax(vote) + 1
        #print(i // 2 + 1, predict, vote)
        if predict != i // 2 + 1:
            err += 1
    print(f"Accuracy:{1 - err/len(low_test):.4f}")

#%%
if __name__ == "__main__":
    train_data, test_data = readData()
    
    # PCA
    if False:
        PCA_file = './Experiment Result/PCA/'
        W_PCA, mean_PCA = PCA(train_data)
        eigenFace(W_PCA, PCA_file + 'eigenfaces/')
        reconstructFace(W_PCA, mean_PCA, train_data, PCA_file + 'reconstruct/')
        faceRecongnition(W_PCA, mean_PCA, train_data, test_data, 5)
    
    # LDA
    if True:
        scalar = 3 # 45000 -> 5000
        LDA_file = './Experiment Result/LDA/'
        data = imageCompression(train_data, scalar)
        compress_test = imageCompression(test_data, scalar)
        
        W_LDA = LDA(data, S=scalar)
        print("LDA done.")
        eigenFace(W_LDA, LDA_file + 'fisherfaces/', S=scalar)
        reconstructFace(W_LDA, None, data, LDA_file + 'reconstruct/', S=scalar)
        faceRecongnition(W_LDA, None, data, compress_test, 5)
    
    # Kernel PCA, LDA
    
    # Different kernels