# -*- coding: utf-8 -*-
"""
Machine Learning Homework 1

"""

import numpy as np
import matplotlib.pyplot as plt
#%%

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

def add(a, b, minus=False):
    if len(a)!=len(b) or len(a[0])!=len(b[0]):
        print('Error: matrix addition, wrong size of matrix')
        return
    re = np.zeros((len(a),len(a[0])))
    for i in range(len(a)):
        for j in range(len(a[i])):
            re[i][j] = a[i][j] + b[i][j] if not minus else a[i][j] - b[i][j]
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

def conMulti(c, ma):
    tmp = ma.copy()
    tmp[:][:] = ma[:][:] * c
    return tmp    

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

def calErr(A, b, X, N, _lambda):
    err = 0.0
    for i in range(len(A)):
        pre = 0.0
        for n in range(N):
            pre += A[i][n] * X[n]
        err += (pre - b[i])**2
    for n in range(N):
        err += _lambda * (X[n]**2)
        #different with sample output here
        #seems that TA forgot to multiple lambda
    return err

def output(err,X,N):
    print('\tFitting line:\n\t\t',end='')
    for n in range(N):
        print('%.10f' % (X[n]), end='')
        if n!=N-1:
            print(' X^%d + ' % (N-1-n), end='')
    print('\n\tTotal error: ',err[0])

#%%
#Input file and parameter

#file_path = input('Enter file path:')
#file_name = input('Enter file name:')
#N = int(input('Enter polynomial bases N:'))
#_lambda = float(input('Enter lambda:'))
file_path = "./"
file_name = 'test.txt'
N = 3
_lambda = 10000

data, A, b = inputData(file_path, file_name)
data_length = len(A[0])
b = transpo(b) #1xn -> nx1
#print('Origin data:\n',A)
#print('=====================')

#%%
#RLSE

# X = (ATA + l*I)^-1 * AT * b
Ad = designM(A,N)
AdT = transpo(Ad)
X = add(multi(AdT, Ad),unitM(N,_lambda))
X = LUdecompo(X)
X = multi(multi(X, AdT), b)

print('LSE:')
output(calErr(Ad, b, X, N, _lambda), X, N)

#%%
#Newton's Method

# L = ||AX-b||^2
X2 = np.zeros((N,1))
'''
tmp1 = conMulti(2, multi(multi(AdT, Ad), X2))
tmp2 = conMulti(2, multi(AdT, b))
delf = add(tmp1, tmp2, True)
Hfinv = LUdecompo(conMulti(2, multi(AdT, Ad)))
#維度錯誤，delf(Nx1)不能和Hf(NxN)相乘
#X2 = add(X2, multi(delf, Hfinv), True)
print(X2)
'''
X2 = LUdecompo(multi(AdT, Ad))
X2 = multi(multi(X2, AdT), b)
print("Newton's Method:")
output(calErr(Ad, b, X2, N, 0), X2, N)

#%%
#Visualize

#RLSE
fig = plt.figure()
ax = fig.add_subplot(2, 1, 1)
ax.scatter(A[0],b)
x_line = np.linspace(-6,6,30000) #sample points of the regression line
y_line = np.zeros(len(x_line))
for i in range(len(x_line)):
    for n in  range(N):
        y_line[i] += X[n] * (x_line[i]**(N-n-1))
ax.plot(x_line, y_line, 'r')

#Newton's Method
ax = fig.add_subplot(2, 1, 2)
ax.scatter(A[0],b)
x_line = np.linspace(-6,6,30000)
y_line = np.zeros(len(x_line))
for i in range(len(x_line)):
    for n in  range(N):
        y_line[i] += X2[n] * (x_line[i]**(N-n-1))
ax.plot(x_line, y_line, 'r')

plt.show()