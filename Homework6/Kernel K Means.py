"""
Machine Learning Homework 6
    Kernel K Means

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
#COLOR_3 = 
#COLOR_4 = 

#%%

def readImg(fileName):
    img = cv2.imread('./data/' + fileName)
    return img.reshape(img_size, 3)

def init(img):
    m = []
    select = np.random.choice(img_size, K, replace=False)
    for k in range(K):
        m.append([select[k], img[select[k]]])
    c = np.array([1 for k in range(K)])
    alpha = np.zeros((img_size, K), dtype=np.uint8)
    for k in range(K):
        alpha[m[k][0]][k] = 1
    return m, c, alpha

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

def distance(x, img, k, c, alpha, dist3rd):
    d = kernel[x[0]][x[0]]
    tmp = 0
    for n in range(img_size):
        if alpha[n][k] == 0:
            continue
        tmp += alpha[n][k] * kernel[x[0]][n]
    d -= (2 / c[k]) * tmp
    d += dist3rd[k]
    return d

def dist_third_term(alpha, c):
    v = np.zeros(K)
    for k in range(K):
        tmp = 0
        for p in range(img_size):
            if alpha[p][k] == 0:
                continue
            for q in range(img_size):
                if alpha[q][k] == 0:
                    continue
                tmp += alpha[p][k] * alpha[q][k] * kernel[p][q]
        v[k] = tmp / (c[k] ** 2)
    print("Distance third term calculated.")
    return v

def E_Step(img, c, alpha):
    a = np.zeros((img_size, K), dtype=np.uint8)
    dist3rd = dist_third_term(alpha, c)
    for n in trange(img_size):
        min_dist = 1e8
        min_k = None
        for k in range(K):
            d = distance([n, img[n]], img, k, c, alpha, dist3rd) 
            if d < min_dist:
                min_dist = d
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

def visualize(img, alpha):    
    im = np.zeros((img_length, img_length, 3))
    for n in range(img_size):
        for k in range(K):
            if alpha[n][k] == 1:
                im[n//img_length][n%img_length] = COLOR_2[k]
    
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(im)
    
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(img.reshape(img_length, img_length, 3))
    plt.show()
    return im

#%%
if __name__ == "__main__":
    img = readImg('image2.png')
    pre_alpha = None
    delta = img_size
    _iter = 0
    
    # Init mean
    means, c, alpha = init(img)
    
    kernel = computeKernel(img)
    
    while delta > 100:
        _iter += 1
        
        # E step: Classify all samples
        pre_alpha = alpha.copy()
        print("Before E step")
        alpha = E_Step(img, c, alpha)
        print("After E step")
        c = updateCk(alpha)
        
        # M step: re-compute means
        # No need to update mean in kernel k-means
        
        delta = computeChange(pre_alpha, alpha)
        print(f"Iter:{_iter}, delta:{delta}")
        
        # Visualize cluster result
        result = visualize(img, alpha)
        cv2.imwrite(f'./kkm result/img2_0527_1300/img2_kkm_{_iter}.jpg', result)
        