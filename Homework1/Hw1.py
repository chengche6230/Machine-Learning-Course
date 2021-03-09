# -*- coding: utf-8 -*-
"""
Machine Learning Homework 1

"""

import numpy as np
import matplotlib.pyplot as plt


def inputData(file_path, file_name):
    file = open(file_path+file_name,'r')
    data = []
    for line in file.readlines():
        tmp = line.split(',')
        tmp = [float(t) for t in tmp]
        data.append(tmp)
    A = [[ x[0] for x in data]]
    b = [[ y[1] for y in data]]
    A = np.array(A)
    b = np.array(b)
    return data, A, b

def transpo(ori):
    tm = np.zeros((len(ori[0]),len(ori)))
    for r in range(len(ori)):
        for c in range(len(ori[r])):
            tm[c][r] = ori[r][c]
    return tm

def add(a,b):
    if len(a)!=len(b) or len(a[0])!=len(b[0]):
        print('Error: matrix addition, wrong size of matrix')
        return
    re = np.zeros((len(a),len(a[0])))
    for i in range(len(a)):
        for j in range(len(a[i])):
            re[i][j] = a[i][j] + b[i][j]
    return re
    
def multi(a,b): #a*b
    if len(a[0])!=len(b):
        print('Error: matrix multiple, wrong size of matrix')
        return
    re = np.zeros((len(a),len(b[0])))
    for i in range(len(re)):
        for j in range(len(re[i])):
            for k in range(len(a[0])):
                re[i][j] += a[i][k]*b[k][j]
    return re

def unitM(length, _lambda=1):
    m = np.zeros((length,length))
    for i in range(length):
        m[i][i] = 1*_lambda
    return m

def elementaryM(A,i,j):
    E = unitM(len(A))
    Einv = unitM(len(A))
    E[i][j] = -1 * (A[i][j]/A[j][j])
    Einv[i][j] = A[i][j]/A[j][j]
    return E, Einv

def LUdecompo(A):
    tmp_A = A.copy()
    #find L, U (in fact, just need to find L-1)
    L = unitM(len(A))
    Linv = unitM(len(A))
    U = np.zeros((len(A),len(A)))
    for i in range(1,len(A)):
        for j in range(i):
            E,Einv = elementaryM(tmp_A,i,j)
            tmp_A = multi(E, tmp_A)
            L = multi(L, Einv)
            Linv = multi(E,Linv)
    U = tmp_A
    
    #find L-1, U-1 to get A-1
    Uinv = unitM(len(A))
    for i in range(len(A)):
        Uinv[i][i] = 1/U[i][i] #對角取倒數
        for j in range(i-1,-1,-1): #j from (i-1) to 0
            s = 0
            for k in range(j+1,i+1): #k from (j+1) to i
                s += U[j][k] * Uinv[k][i]
            Uinv[j][i] = -s/U[j][j] #pivot
    
    Ainv = multi(Uinv,Linv)
    return Ainv

def designM(A,N):
    tmp_A = np.zeros((len(A[0]),N))
    for r in range(len(tmp_A)):
        for i in range(N):
            tmp_A[r][N-1-i] = A[0][r]**i
    return tmp_A

#input file and parameter
file_path = "./"
file_name = 'test.txt'
#file_name = input('Enter file name:')
#N = int(input('Enter polynomial bases N:'))
N = 3
_lambda = 1
#_lambda = float(input('Enter lambda:'))

data, A, b = inputData(file_path, file_name)
data_length = len(A[0])
b = transpo(b) #1xn -> nx1
print('Origin data:\n',A)
print('=======================================')

#RLSE
err = 0.0

# X = (ATA + l*I)^-1 * AT * b
Ad = designM(A,N)
AdT = transpo(Ad)
X = add(multi(AdT, Ad),unitM(N,_lambda))
X = LUdecompo(X)
X = multi(multi(X, AdT), b)
print('X:\n',X)

    #compute err by using w
    #output result

#Newton's Method
w2 = np.zeros(N)
err2 = 0.0

    #use formula repeatly compute w
    #implement matrix minus, Gradient, Hession matrix
    #determine when to converge
    
    #compute err by using w
    #output result


#visualize