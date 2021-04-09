"""
Machine Learning Homework 3
    -Random Data Generator
        -Univariate Gaussian data generator
        -Polynomial basis linear model data generator
    -Sequential Estimator
    -Bayesian Linear Regression

"""
import numpy as np
import math
import matplotlib.pyplot as plt

def Gaussian(mean, var):
    # Marsaglia polar method
    S, U, V = 0, 0, 0
    while True:
        U = np.random.uniform(-1, 1)
        V = np.random.uniform(-1, 1)
        S = U**2 + V**2
        if S < 1:
            break
    tS = math.sqrt(-2 * math.log(S) / S)
    x = U * tS
    x = x * (s**0.5) + m
    return x

def basisData(n, a, w):
    if n!=len(w):
        print("Error: input size error.")
        return -1
    x = np.random.uniform(-1, 1)
    y = Gaussian(0, a) #eplison
    for i in range(n):
        y += w[i] * (x**i)
    return x, y

def SequentialEstimator(m, s):
    print("Data point source function: N(%.2f, %.2f)" % (m, s))
    
    n = 0
    mean, square_mean = 0.0, 0.0
    last_mean = 0.0
    
    CONVERGE = 25
    con_count = 0
    con_thres = 5e-3
    
    while con_count < CONVERGE:
        d = Gaussian(m, s)
        n += 1
        print("Add data point:", d)
        square_mean = (square_mean * (n-1) + d**2) / n
        mean = (mean * (n-1) + d) / n
        var = square_mean - mean**2 # var(x) = E(x^2) - E(x)^2
        print("Mean =", mean, "Variance =", var)
        
        if abs(last_mean - mean) < con_thres:
            con_count += 1
        else:
            con_count = 0
        last_mean = mean
    print("n =", n)
    
def BayesianLinearRegression():
    
    return
    
if __name__ == "__main__":
    #m = float(input("Enter mean:"))
    #s = float(input("Enter variance:"))
    m = 3
    s = 5
    
    n = 2
    a = 1
    w = [0, 1]
    
    SequentialEstimator(m, s)
    
    BayesianLinearRegression()