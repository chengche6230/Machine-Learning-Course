"""
Machine Learning Homework 2-2

"""
import numpy as np
import matplotlib.pyplot as plt
from math import factorial as f

def binomial(m, N, p):
    c = f(N) / (f(m) * f(N - m))
    return c * (p**m) * ((1-p)**(N-m))

def r(x):
    return f(x-1)

def beta(p, a, b):
    c = r(a+b) / (r(a) * r(b))
    return c * (p**(a-1)) * ((1-p)**(b-1))

def output(i, data, likeli, pri_a, pri_b, post_a, post_b):
    print("case %d: " % i, end="")
    print(data, end="")
    print("Likelihood:", likeli)
    print("Beta prior:     a = %d b = %d" % (pri_a, pri_b))
    print("Beta posterior: a = %d b = %d\n" % (post_a, post_b))

def visualize(ta, tb, a, b, m, N):
    fig = plt.figure(figsize=(12, 4))
    x_line = np.linspace(0, 1, 100)
    y_line = np.zeros(len(x_line))
    
    # Prior
    if ta>0 and tb>0:
        ax = fig.add_subplot(1, 3, 1)
        ax.set_title("Prior")
        for i in range(len(x_line)):
            y_line[i] = beta(x_line[i], ta, tb)
        ax.plot(x_line, y_line, 'r')
        #ax.set_ylim([0, y_lim])
        ax.set_xlim([0, 1])
    
    # Likelihood
    ax = fig.add_subplot(1, 3, 2)
    ax.set_title("Likelihood")
    for i in range(len(x_line)):
        y_line[i] = binomial(m, N, x_line[i])
    ax.plot(x_line, y_line, 'b')
    ax.set_ylim([0, 1])
    ax.set_xlim([0, 1])
    
    # Posterior
    ax = fig.add_subplot(1, 3, 3)
    ax.set_title("Posterior")
    for i in range(len(x_line)):
        y_line[i] = beta(x_line[i], a, b)
    ax.plot(x_line, y_line, 'r')
    #ax.set_ylim([0, y_lim])
    ax.set_xlim([0, 1])
    
    plt.show()
    
FILE = "./2-2input.txt"


#a = float(input("Input a: "))
#b = float(input("Input b: "))
a = 0
b = 0

with open(FILE, 'r') as file:
    index = 1
    prior, posterior = 0, 0
    for line in file:
        
        # Data pre-processing
        data = list(line)
        data = data[:-1]
        for t in range(len(data)):
            data[t] = int(data[t])
        
        # Calculate prior/like./post.
        m = data.count(1)
        N = len(data)
        p_mle = m / N
        if index>1:
            prior = posterior
        likelihood = binomial(m, N, p_mle)
        ta, tb = a, b
        a += m
        b += (N-m)
        posterior = beta(p_mle, a, b)
        
        # Output
        output(index, line, likelihood, ta, tb, a, b)
        
        visualize(ta, tb, a, b, m, N)
        
        index += 1
