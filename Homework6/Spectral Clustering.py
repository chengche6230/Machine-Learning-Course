"""
Machine Learning Homework 6
    Spectral Clustering

"""
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from tqdm import trange

IMAGE = 2
img_length = 100
img_size = 10000

K = 2
gamma_s = 1 / img_size
gamma_c = 1 / (256 * 256)

method = "random"

COLOR = [[[153, 102, 51], [0, 153, 255]],
         [[0, 102, 204], [51, 204, 204], [153, 102, 51]],
         [[0, 102, 204], [51, 204, 204], [153, 102, 51], [153, 153, 153]]]

#%%

def readImg(fileName):
    img = cv2.imread('./data/' + fileName)
    return img.reshape(img_size, 3)

def spatialSimi(p1, p2):
    p = (p1 // img_length - p2 // img_length) ** 2
    p += (p1 % img_length - p2 % img_length) ** 2
    return p

def colorSimi(c1, c2):
    c1 = np.array(c1, dtype=np.uint32)
    c2 = np.array(c2, dtype=np.uint32)
    return np.sum((c1 - c2) ** 2)

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

def degreeMatrix(W):
    D = np.zeros((img_size, img_size))
    for n in range(img_size):
        D[n][n] = np.sum(W[n])
    return D

def eigenDecompo(L):
    eigenvalue, eigenvector = np.linalg.eig(L)
    eigenindex = np.argsort(eigenvalue)
    eigenvector = eigenvector[:, eigenindex]
    return eigenvector[:, 1 : 1 + K] # except first eigenvector

def initKMeans(U):
    # Random
    m = []
    cluster = np.zeros(img_size, dtype=np.uint32)
    select = np.random.choice(img_size, K, replace=False)
    print("Select:", select)
    for s in select:
        m.append(U[s])
    for n in range(img_size):
        cluster[n] = np.random.randint(K)
    return m, cluster

def distance(x, m):
    dist = 0
    for k in range(K):
        dist += (x[k] - m[k]) ** 2
    return dist

def E_Step(U, means):
    cluster = np.zeros(img_size, dtype=np.uint32)
    for n in range(img_size):
        min_k, min_dist = None, 1e8
        for k in range(K):
            d = distance(U[n], means[k])
            if d < min_dist:
                min_dist = d
                min_k = k
        if min_k is not None:
            cluster[n] = min_k
    return cluster
    
def M_Step(U, cluster):
    m = np.zeros((K, K), dtype=np.float64)
    for k in range(K):
        size = np.sum(cluster==k)
        for n in range(img_size):
            if cluster[n] == k:
                m[k] += U[n]
        m[k] /= size
    return m

def computeDelta(pre_cluster, cluster):
    delta = 0
    for n in range(img_size):
        if pre_cluster[n] != cluster[n]:
            delta += 1
    return delta

def visualize(img, cluster):
    im = np.zeros((img_length, img_length, 3))
    for n in range(img_size):
        im[n//img_length][n%img_length] = COLOR[K-2][cluster[n]]
    
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(im)
    
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(img.reshape(img_length, img_length, 3))
    plt.show()
    return im

def kMeans(U, img):
    _iter = 0
    
    # Init means
    means, cluster = initKMeans(U)
    visualize(img, cluster)
    pre_cluster = None
    delta = img_size    
    
    result_file_path = f'./sc result/img{IMAGE}_{K} class_{method}'
    try:
        os.mkdir(result_file_path)
    except:
        pass
    
    result = visualize(img, cluster)
    cv2.imwrite(f'{result_file_path}/{_iter}.jpg', result)
    
    while delta > 0:
        _iter += 1
        
        # E Step: clustering
        pre_cluster = cluster.copy()
        cluster = E_Step(U, means)
        print("End E step.")
        
        # M Step: update means
        means = M_Step(U, cluster)
        print("End M step.")
        
        # Validate
        delta = computeDelta(pre_cluster, cluster)
        print(f"Iter:{_iter}, delta:{delta}")
        result = visualize(img, cluster)
        #cv2.imwrite(f'{result_file_path}/{_iter}.jpg', result)
        
        #input()

#%%
if __name__ == "__main__":
    img = readImg(f'image{IMAGE}.png')
    
    #W = computeKernel(img)
    W = np.load(f'kernel_img{IMAGE}.npy')
    D = degreeMatrix(W)
    
    # Ratio Cut
    L = D - W
    U = eigenDecompo(L)
    kMeans(U, img)
    