"""
Machine Learning Homework 4-1
    Logistic Regression
    
"""
import math
import numpy as np
from numpy.linalg import inv
from numpy import matmul as mul
import matplotlib.pyplot as plt

def designM(x1, x2, N): # from hw1
    dMat = np.zeros((len(x1) + len(x2), N))
    for i in range(len(x1)):
        dMat[i][0] = 1 # dummy
        dMat[i][1] = x1[i][0]
        dMat[i][2] = x1[i][1]
    for i in range(len(x1), len(x1) + len(x2)):
        dMat[i][0] = 1 # dummy
        dMat[i][1] = x2[i - len(x1)][0]
        dMat[i][2] = x2[i - len(x1)][1]
    return dMat

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

def gradient(dMat, w, y, N):
    grad = mul(dMat, w)
    for i in range(N):
        grad[i] = y[i] - 1/(1 + math.exp(-1 * grad[i]))
    grad = mul(dMat.T, grad)
    return grad

def Hessian(dMat, w, N):
    D = np.zeros((N, N))
    for i in range(N):
        t = mul(dMat[i], w)
        D[i][i] = math.exp(-t) / ((1 + math.exp(-t))**2)
    #print("D:\n", D)
    H = mul(mul(dMat.T, D), dMat)
    return H

def activate(x):
    return 1/(1 + math.exp(-1 * x))

def validate(N, w, D1, D2):
    pre = []
    for i in range(N):
        x = w[0] * 1 + w[1] * D1[i][0] + w[2] * D1[i][1]
        y = activate(x)
        y = 1 if y>0.5 else 0
        pre.append(y)
    for i in range(N):
        x = w[0] * 1 + w[1] * D2[i][0] + w[2] * D2[i][1]
        y = activate(x)
        y = 1 if y>0.5 else 0
        pre.append(y)
    return pre
    
def confusionMat(y, pre, N):
    TP, FN, FP, TN = 0, 0, 0, 0
    for i in range(2 * N):
        TP = TP + 1 if y[i]==0 and y[i]==pre[i] else TP
        FN = FN + 1 if y[i]==0 and y[i]!=pre[i] else FN
        FP = FP + 1 if y[i]==1 and y[i]!=pre[i] else FP
        TN = TN + 1 if y[i]==1 and y[i]==pre[i] else TN
    return [TP, FN, FP, TN]

def output(w, table, _type=""):
    print(_type)
    print("\nw:")
    for i in w:
        print(i)
    print("\nConfusion Matrix:")
    print("\t\t\tPredict cluster 1\tPredict cluster 2")
    print("In cluster 1\t\t%d\t\t\t\t\t%d" % (table[0], table[1]))
    print("In cluster 2\t\t%d\t\t\t\t\t%d" % (table[2], table[3]))
    print("\nSensitivity (Successfully predict cluster 1):", table[0] / (table[0] + table[1]))
    print("Specificity (Successfully predict cluster 2):", table[3] / (table[2] + table[3]))
    
def visualize(N, D1, D2, pre_gd, pre_nt):
    fig = plt.figure(figsize=(7, 4))
    
    # Ground truth
    ax = fig.add_subplot(1, 3, 1)
    ax.set_title("Ground truth")
    ax.plot([r[0] for r in D1], [r[1] for r in D1], "r.")
    ax.plot([r[0] for r in D2], [r[1] for r in D2], "b.")
    
    # Gradient descent
    ax = fig.add_subplot(1, 3, 2)
    ax.set_title("Gradient descent")
    for i in range(N): # D1
        if pre_gd[i] == 0:
            ax.plot(D1[i][0], D1[i][1], "r.")
        else:
            ax.plot(D1[i][0], D1[i][1], "b.")
    for i in range(N, 2*N): # D2
        j = i - N
        if pre_gd[i] == 0:
            ax.plot(D2[j][0], D2[j][1], "r.")
        else:
            ax.plot(D2[j][0], D2[j][1], "b.")
    
    # Newton's method
    ax = fig.add_subplot(1, 3, 3)
    ax.set_title("Newton's method")
    for i in range(N): # D1
        if pre_nt[i] == 0:
            ax.plot(D1[i][0], D1[i][1], "r.")
        else:
            ax.plot(D1[i][0], D1[i][1], "b.")
    for i in range(N, 2*N): # D2
        j = i - N
        if pre_nt[i] == 0:
            ax.plot(D2[j][0], D2[j][1], "r.")
        else:
            ax.plot(D2[j][0], D2[j][1], "b.")
    
    plt.show()

if __name__ == "__main__":
    lr = 1e-2
    N = 50
    mx1 = my1 = 1
    mx2 = my2 = 3
    vx1 = vy1 = 2
    vx2 = vy2 = 4
    
    # Data pre-process
    D1 = []
    D2 = []
    for i in range(N):
        x = Gaussian(mx1, vx1)
        y = Gaussian(my1, vy1)
        D1.append([x, y])
        x = Gaussian(mx2, vx2)
        y = Gaussian(my2, vy2)
        D2.append([x, y])
    dMat = designM(D1, D2, 3)
    y = np.zeros((len(dMat), 1))
    for i in range(len(D1), len(D1) + len(D2)):
        y[i] = 1
    
    # Converge parameter
    conv_thres_gd = 18e-3
    conv_thres_nt = 18e-3
    converge = 5
    
    # Gradient descent
    w = np.zeros((3, 1))
    last_w = np.zeros((3, 1))
    conv_count = 0
    _iter = 0
    while conv_count < converge:
        _iter += 1
        
        last_w = w.copy()
        w += lr * gradient(dMat, w, y, 2 * N)
        
        _last = sum(abs(last_w))
        _curr = sum(abs(w))
        conv_count  = conv_count + 1 if abs(_last - _curr) < conv_thres_gd else 0
        
    print("Converge iteration:", _iter)
        
    predict_gd = validate(N, w, D1, D2)
    table = confusionMat(y, predict_gd, N)

    output(w, table, "\nGradient descent:")
    print("-----------------------------------------------")
    
    # Newton's method
    w2 = np.zeros((3, 1))
    last_w2 = np.zeros((3, 1))
    conv_count = 0
    _iter = 0
    while conv_count < converge:
        _iter += 1
        grad = gradient(dMat, w2, y, 2 * N)
        H = Hessian(dMat, w2, 2 * N)
        if np.linalg.det(H) == 0:
            print("Hessian isn't invertible.")
        H_inv = inv(H)
        
        last_w2 = w2.copy()
        w2 += lr * mul(H_inv, grad)
        
        _last = sum(abs(last_w2))
        _curr = sum(abs(w2))
        conv_count  = conv_count + 1 if abs(_last - _curr) < conv_thres_nt else 0
    
    print("Converge iteration:", _iter)
    
    predict_nt = validate(N, w2, D1, D2)
    table = confusionMat(y, predict_nt, N)

    output(w2, table, "\nNewton's method':")


    visualize(N, D1, D2, predict_gd, predict_nt)
