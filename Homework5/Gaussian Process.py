"""
Machine Learning Homework 5
    Gaussian Process

"""
import numpy as np
from numpy import matmul as mul
from numpy.linalg import inv
import matplotlib.pyplot as plt

def loadData():
    X = []
    Y = []
    with open("./data/input.data", 'r') as file:
        for line in file.readlines():
            li = line.split()
            X.append(float(li[0]))
            Y.append(float(li[1]))
    return X, Y

def k(x1, x2, sigma, alpha, length):    
    d = (x1 - x2)**2
    k = (1 + d / (2 * alpha * length * length)) ** (-alpha)
    return k * (sigma**2)

def Cov(X, beta, sigma, alpha, length):
    C = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(len(X)):
            C[i][j] = k(X[i], X[j], sigma, alpha, length)
            if i == j:
                C[i][j] += 1/beta 
    return C

def predict(X, Y, C, beta, sigma, alpha, length):
    sample = 1000
    mean = np.zeros((sample))
    var = np.zeros((sample))
    
    y = np.array(Y).reshape((len(Y), 1))
    pre_x = np.linspace(-60, 60, sample)
    for n in range(sample):
        K = np.zeros((len(X), 1))
        for i in range(len(X)):
            K[i][0] = k(X[i], pre_x[n], sigma, alpha, length)
        mean[n] = mul(mul(K.T, inv(C)), y)
        k_s = k(pre_x[n], pre_x[n], sigma, alpha, length) + 1 / beta
        var[n] = k_s - mul(mul(K.T, inv(C)), K)
        
    return mean, var

def visualize(data, pre_m, pre_v):
    fig = plt.figure()
    x = np.linspace(-60, 60, len(pre_m))
    interval = 1.96 * (pre_v ** 0.5) # 95% Confidence Interval
    
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(data[0], data[1], "k.")
    ax.plot(x, pre_m, "r-")
    ax.fill_between(x, pre_m + interval, pre_m - interval, color='pink')
    
    plt.show()

def GaussianProcess(X, Y, beta, sigma, alpha, length):
    C = Cov(X, beta, sigma, alpha, length)

    mean, var = predict(X, Y, C, beta, sigma, alpha, length)
    
    visualize([X, Y], mean, var)

if __name__ == "__main__":
    X, Y = loadData()
    beta = 5
    
    sigma = 1
    alpha = 1
    length = 1
    
    GaussianProcess(X, Y, beta, sigma, alpha, length)
    
    # Optimize parameters
    
    