"""
Machine Learning Homework 3-1
    Random Data Generator

"""
import numpy as np
import math
import matplotlib.pyplot as plt

def dataGenerator():
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
    y = V * tS    
    return x, y
    
if __name__ == "__main__":
    #m = float(input("Enter mean:"))
    #s = float(input("Enter variance:"))
    m = 0
    s = 1
    
    X = []
    for i in range(100000):
        x, y = dataGenerator()
        x = x * (s**0.5) + m
        X.append(x)
    
    
    b = np.linspace(m-15, m+15, 100)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(X, bins=b)
    plt.show()
    