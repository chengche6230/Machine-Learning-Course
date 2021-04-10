"""
Machine Learning Homework 3
    -Random Data Generator
        -Univariate Gaussian data generator
        -Polynomial basis linear model data generator
    -Sequential Estimator
    -Bayesian Linear Regression

"""
import math
import numpy as np
from numpy.linalg import inv
from numpy import matmul as mul
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
    x = x * (var**0.5) + mean
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
        
        # Check converge
        if abs(last_mean - mean) < con_thres:
            con_count += 1
        else:
            con_count = 0
        last_mean = mean
    print("n =", n)
    
def designM(x, N): # from hw1
    dMat = np.zeros((1, N))
    for i in range(N):
        dMat[0][i] = x**i
    return dMat

def covarMat(b, n):
    cov = np.zeros((n, n))
    for i in range(n):
        cov[i][i] = 1/b
    return cov

def output(x, y, m, var, post_m, post_var):
    print("Add data point (%.5f, %.5f)" % (x, y))
    print("\nPosterior mean:")
    print(m)
    print("\nPosterior variance:")
    print(var)
    print("\nPredictive distribution ~ N(%.5f, %.5f)" % (post_m, post_var))
    print("--------------------------------------------------")

def figSetting(ax, title, index=-1, LOG=None, sample=500):
    ax.set_title(title)
    ax.set_ylim([-25, 25])
    ax.set_xlim([-2, 2])
    if title == 'Ground truth':
        return
    
    x = np.linspace(-2, 2, sample)
    y_varp = np.zeros(sample)
    y_varn = np.zeros(sample)
    for i in range(sample):
        y_varp[i] = LOG['mean'][index][i] + LOG['var'][index][i]
        y_varn[i] = LOG['mean'][index][i] - LOG['var'][index][i]
    ax.plot(x, LOG['mean'][index], "k")
    ax.plot(x, y_varp, "r")
    ax.plot(x, y_varn, "r")
    
def visualize(X, Y, LOG, N, a, w, sample):
    fig = plt.figure(figsize=(10, 8))
    x = np.linspace(-2, 2, sample)
    y = np.zeros(sample)
    y_varp = np.zeros(sample)
    y_varn = np.zeros(sample)
    
    ax = fig.add_subplot(2, 2, 1)
    figSetting(ax, "Ground truth")
    for i in range(sample):
        for n in range(N):
            y[i] += w[n] * (x[i]**n)
        y_varp[i] = y[i] + math.sqrt(a)
        y_varn[i] = y[i] - math.sqrt(a)
    ax.plot(x, y, "k")
    ax.plot(x, y_varp, "r")
    ax.plot(x, y_varn, "r")
    
    ax = fig.add_subplot(2, 2, 2)
    ax.plot(X, Y, "b.")
    figSetting(ax, "Predict result", 2, LOG, sample)
    
    ax = fig.add_subplot(2, 2, 3)
    ax.plot(X[:10], Y[:10], "b.")
    figSetting(ax, "After 10 data points", 0, LOG, sample)
    
    ax = fig.add_subplot(2, 2, 4)
    ax.plot(X[:50], Y[:50], "b.")
    figSetting(ax, "After 50 data points", 1, LOG, sample)
    
    plt.show()

def logData(LOG, sample, mean, var, n):
    LOG['mean'].append([])
    LOG['var'].append([])
    sample_x = np.linspace(-2, 2, sample)
    for i in range(sample):
        tdMat = designM(sample_x[i], n)
        LOG['mean'][len(LOG['mean'])-1].append(float(mul(mean.T, tdMat.T)))
        LOG['var'][len(LOG['mean'])-1].append(float(1/a + mul(mul(tdMat, inv(var)), tdMat.T)))

def BayesianLinearRegression(b, n, a, w):
    X = []
    Y = []
    LOG = {'mean': [], 'var': []}
    m = np.zeros((n, 1))
    _lambda = inv(covarMat(b, n))
    data_num = 0
    
    sample = 500
    
    CONVERGE = 20
    con_count = 0
    con_thres = 1e-3
    
    while con_count < CONVERGE:
        x, y = basisData(n, a, w)
        X.append(x)
        Y.append(y)
        data_num += 1
        print("# of data:", data_num)
        
        dMat = designM(x, n)
        
        # Update prior
        last_lambda = _lambda
        _lambda = last_lambda + (1/a) * mul(dMat.T, dMat)
        
        last_m = m
        tm = mul(last_lambda, m)
        tm2 = (1/a) * mul(dMat.T, [[y]])
        m = mul(inv(_lambda), tm + tm2)
        
        # Posterior mean, var
        post_m = mul(m.T, dMat.T)
        post_var = 1/a + mul(mul(dMat, inv(_lambda)), dMat.T)
        
        output(x, y, m, inv(_lambda), post_m, post_var)
        
        # Check converge (every weight didn't change too much)
        flag = True
        for t in range(n):
            if abs(last_m[t] - m[t]) >= con_thres:
                flag = False
                break
        if flag:
            con_count += 1
        else:
            con_count = 0
        
        # Log data
        if data_num == 10 or data_num == 50:
            logData(LOG, sample, m, _lambda, n)
            
    logData(LOG, sample, m, _lambda, n)
    visualize(X, Y, LOG, n, a, w, sample)
    
if __name__ == "__main__":
    m = 3
    s = 5
    
    b = 1
    n = 4
    a = 1
    w = [1, 2, 3, 4]
    
    SequentialEstimator(m, s)
    
    BayesianLinearRegression(b, n, a, w)