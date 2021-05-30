"""
Machine Learning Homework 6
    Spectral Clustering

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

CUT = "normalized"  # {normalized|ratio}

COLOR = [[[153, 102, 51], [0, 153, 255]],
         [[0, 102, 204], [51, 204, 204], [153, 102, 51]],
         [[0, 102, 204], [51, 204, 204], [153, 102, 51], [153, 153, 153]]]

COLOR_MATPLOT = [['tab:orange', 'tab:blue'], 
                 ['tab:orange', 'tab:blue', 'tab:olive'],
                 ['tab:orange', 'tab:blue', 'tab:olive', 'tab:gray']]

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

def normalizedL(D, W):
    I = np.identity(img_size)
    sqrt_D = np.sqrt(D)
    for i in range(img_size):
        if sqrt_D[i][i] == 0 :
            print(i)
        sqrt_D[i][i] = 1 / sqrt_D[i][i]
    L = I - sqrt_D @ W @ sqrt_D
    return L

def eigenDecompo(L):
    eigenvalue, eigenvector = np.linalg.eig(L)
    eigenindex = np.argsort(eigenvalue)
    eigenvector = eigenvector[:, eigenindex]
    return eigenvector[:, 1:1+K].real # exclude first eigenvector

def minDist(m, Un):
    minDis = 1e8
    for i in range(len(m)):
        d = 0
        for k in range(K):
            d += (m[i][k] - Un[k]) ** 2
        minDis = min(minDis, d)
    return minDis

def nextMean(probSum):
    num = np.random.rand()
    for i in range(1, img_size):
        if num >= probSum[i - 1] and num < probSum[i]:
            return i

def attrSum(U):
    _sum = np.zeros((img_size, K + 2), dtype=np.float32)
    # Columns are {index|summation|U}
    for n in range(img_size):
        # Sum up values of all attribute
        _sum[n][0] = n
        for k in range(K):
            _sum[n][1] += U[n][k]
            _sum[n][k + 2] = U[n][k]
    _sum = _sum[np.argsort(_sum[:, 1])]  # Sort by summation value
    return _sum

def sliceMean(_sum):
    avg = np.zeros(K, dtype=np.float64)
    for k in range(K):
        avg[k] = np.average(_sum[:, k + 2])
    return avg

def initKMeans(U):
    m = []
    cluster = np.zeros(img_size, dtype=np.uint32)
    for n in range(img_size):
        cluster[n] = -1
        
    if METHOD == "random":
        select = np.random.choice(img_size, K, replace=False)
        for s in range(len(select)):
            m.append(U[select[s]])
        
    if METHOD == "kmeans++":
        t = np.random.randint(img_size)
        m.append(U[t])
        for k in range(K-1):
            distSum = 0
            probSum = np.zeros(img_size, dtype=np.float32)
            for n in range(img_size):
                probSum[n] = minDist(m, U[n]) # shortest distance to mean
                distSum += probSum[n]
            probSum[0] /= distSum # Normalize as probability
            for n in range(1, img_size):
                probSum[n] /= distSum
                probSum[n] += probSum[n - 1] # Compute cumulative probability
            m_index = nextMean(probSum)
            m.append(U[m_index])
    
    if METHOD == "naive_sharding":
        sum_attr = attrSum(U)
        _slice = img_size // K
        for k in range(K):
            slice_sum = sum_attr[k*_slice : (k+1)*_slice]
            m.append(sliceMean(slice_sum))
        
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

def visualize(img, cluster, _iter, result_file_path):
    im = np.zeros((img_length, img_length, 3), dtype=np.uint8)
    for n in range(img_size):
        im[n//img_length][n%img_length] = COLOR[K-2][cluster[n]]
    
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

def drawEigenspace(cluster, U, result_file_path):
    pt_x, pt_y, pt_z = [], [], []
    for k in range(K):
        pt_x.append([])
        pt_y.append([])
        pt_z.append([])
    fig = plt.figure()
    if K == 2:
        for n in range(img_size):
            pt_x[cluster[n]].append(U[n][0])
            pt_y[cluster[n]].append(U[n][1])
        for k in range(K):
            plt.scatter(pt_x[k], pt_y[k], c=COLOR_MATPLOT[K - 2][k], s=0.5)
    if K == 3:
        ax = fig.add_subplot(projection='3d')
        for n in range(img_size):
            pt_x[cluster[n]].append(U[n][0])
            pt_y[cluster[n]].append(U[n][1])
            pt_z[cluster[n]].append(U[n][2])
        for k in range(K):
            ax.scatter(pt_x[k], pt_y[k], pt_z[k], c=COLOR_MATPLOT[K - 2][k], s=0.5)
    plt.show()
    
    fig.savefig(f'{result_file_path}/eigen.jpg')

def kMeans(U, img):
    # Init means
    means, cluster = initKMeans(U)
    pre_cluster = None
    delta = img_size 
    _iter = 0   
    
    result_file_path = f'./sc result/img{IMAGE}_{K} class_{METHOD}_{CUT} cut'
    try:
        os.mkdir(result_file_path)
    except:
        pass
   
    while delta > 0:
        _iter += 1
        pre_cluster = cluster.copy()
        
        # E Step: clustering
        cluster = E_Step(U, means)
        
        # M Step: update means
        means = M_Step(U, cluster)
        
        # Validate
        delta = computeDelta(pre_cluster, cluster)
        print(f"Iter:{_iter}, delta:{delta}")
        visualize(img, cluster, _iter, result_file_path)
        
    if K < 4:
        drawEigenspace(cluster, U, result_file_path)

if __name__ == "__main__":
    img = readImg(f'image{IMAGE}.png')
    
    W = computeKernel(img)
    #W = np.load(f'kernel_img{IMAGE}.npy')
    D = degreeMatrix(W)
    
    if CUT == "normalized":
        Ln = normalizedL(D, W)
        Un = eigenDecompo(Ln)
        #Un = np.load(f'Un_img{IMAGE}.npy')
        #Un = Un[:, :K]
        kMeans(Un, img)
    
    if CUT == "ratio":
        L = D - W
        U = eigenDecompo(L)
        #U = np.load(f'U_img{IMAGE}.npy')
        #U = U[:, :K]
        kMeans(U, img)
    