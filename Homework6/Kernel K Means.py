"""
Machine Learning Homework 6
    Kernel K Means

"""
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt
from tqdm import trange

IMAGE = 1  # {1|2}
img_length = 100
img_size = 10000

K = 2  # {2|3|4}
gamma_s = 1 / img_size
gamma_c = 1 / (256 * 256)
kernel = None

METHOD = "random"  # {random|kmeans++|naive_sharding}

COLOR = [[[153, 102, 51], [0, 153, 255]],
         [[0, 102, 204], [51, 204, 204], [153, 102, 51]],
         [[0, 102, 204], [51, 204, 204], [153, 102, 51], [153, 153, 153]]]

def readImg(fileName):
    img = cv2.imread('./data/' + fileName)
    return img.reshape(img_size, 3)

def minDist(m, pos):
    minDis = 1e8
    x = pos % img_length
    y = pos // img_length
    # Use spatial distance
    for i in range(len(m)):
        m_x = m[i][0] % img_length
        m_y = m[i][0] // img_length
        minDis = min(minDis, (x - m_x) ** 2 + (y - m_y) ** 2)
    return minDis

def nextMean(probSum):
    num = np.random.rand()
    for i in range(1, img_size):
        if num >= probSum[i - 1] and num < probSum[i]:
            return i

def attrSum(img):
    _sum = np.zeros((img_size, 5), dtype=np.float32)
    # Five columns are {index|summation|color R|color G|color B}
    for n in range(img_size):
        # Sum up values of all attribute
        _sum[n][0] = n
        _sum[n][1] = (n % img_length) + (n // img_length)
        for c in range(3):
            _sum[n][c + 2] = img[n][c]
            _sum[n][1] += img[n][c]
    _sum = _sum[np.argsort(_sum[:, 1])]  # Sort by summation value
    return _sum

def sliceMean(_sum):
    mIndex = int(np.average(_sum[:, 0]))
    mR = int(np.average(_sum[:, 2]))
    mG = int(np.average(_sum[:, 3]))
    mB = int(np.average(_sum[:, 4]))
    tmp_img = np.array([mR, mG, mB], dtype=np.uint8)
    return [mIndex, tmp_img]

def init(img):
    m = []
    
    if METHOD == "random":
        select = np.random.choice(img_size, K, replace=False)
        for k in range(K):
            m.append([select[k], img[select[k]]])
            
    if METHOD == "kmeans++":
        t = np.random.randint(img_size)
        m.append([t, img[t]])
        for k in range(K-1):
            distSum = 0
            probSum = np.zeros(img_size, dtype=np.float32)
            for n in range(img_size):
                probSum[n] = minDist(m, n) # shortest distance to mean
                distSum += probSum[n]
            probSum[0] /= distSum # Normalize as probability
            for n in range(1, img_size):
                probSum[n] /= distSum
                probSum[n] += probSum[n - 1] # Compute cumulative probability
            m_index = nextMean(probSum)
            m.append([m_index, img[m_index]])
    
    if METHOD == "naive_sharding":
        sum_attr = attrSum(img)
        _slice = img_size // K
        for k in range(K):
            slice_sum = sum_attr[k*_slice : (k+1)*_slice]
            m.append(sliceMean(slice_sum))

    alpha = np.zeros((img_size, K), dtype=np.uint8)
    for k in range(K):
        alpha[m[k][0]][k] = 1
    c = updateCk(alpha)
    return m, c, alpha

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

def distance(c, alpha):
    dist = np.ones((K, img_size))
    
    for k in range(K):
        a = alpha[:, k].reshape(-1, 1).T
        second_term = a @ kernel
        second_term = second_term * 2 / c[k]
        dist[k] -= second_term.flatten()
        
        indicator = alpha[:, k].reshape(-1,1)
        third_term = np.sum(indicator.T @ kernel @ indicator)
        third_term /= (c[k] ** 2)
        dist[k] += third_term
    
    return dist    

def E_Step(img, c, alpha):
    a = np.zeros((img_size, K), dtype=np.uint8)
    
    # Distance between all data points
    dist = distance(c, alpha)
    
    # Clustering
    for n in range(img_size):
        min_dist = 1e8
        min_k = None
        for k in range(K):
            if dist[k][n] < min_dist:
                min_dist = dist[k][n]
                min_k = k
        if min_k is not None:
            a[n][min_k] = 1
    return a

def updateCk(alpha):
    c = np.zeros(K)
    for k in range(K):
        c[k] = np.sum(alpha[:, k]==1)
    return c

def computeChange(pre_alpha, alpha):
    delta = 0
    for n in range(img_size):
        if np.argmax(pre_alpha[n]) != np.argmax(alpha[n]):
            delta += 1
    return delta

def visualize(img, alpha, _iter, result_file_path):    
    im = np.zeros((img_length, img_length, 3), dtype=np.uint8)
    for n in range(img_size):
        for k in range(K):
            if alpha[n][k] == 1:
                im[n//img_length][n%img_length] = COLOR[K - 2][k]
    
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 2, 1)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    ax.imshow(im)
    plt.title(f"Iteration: {_iter}")
    
    ax = fig.add_subplot(1, 2, 2)
    img = cv2.cvtColor(img.reshape(img_length, img_length, 3), cv2.COLOR_RGB2BGR)
    ax.imshow(img)
    plt.show()
    
    fig.savefig(f'{result_file_path}/{_iter}.jpg')

if __name__ == "__main__":
    img = readImg(f'image{IMAGE}.png')
    pre_alpha = None
    delta = img_size
    _iter = 0
    
    # Init mean and cluster assignment
    means, c, alpha = init(img)
    
    #kernel = computeKernel(img)
    kernel = np.load(f'kernel_img{IMAGE}.npy')
    
    result_file_path = f'./kkm result/img{IMAGE}_{K} class_{METHOD}'
    try:
        os.mkdir(result_file_path)
        1
    except:
        pass
    
    while delta > 0:
        _iter += 1
        
        # E step: Classify all samples
        pre_alpha = alpha.copy()
        alpha = E_Step(img, c, alpha)
        c = updateCk(alpha)
        
        # M step: re-compute means
        # No need to update mean in kernel k-means
        
        delta = computeChange(pre_alpha, alpha)
        print(f"Iter:{_iter}, delta:{delta}")
        
        # Visualize and save clustering result
        visualize(img, alpha, _iter, result_file_path) 