"""
Machine Learning Homework 5
    SVM

"""
import time
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from svmutil import *

train_num = 5000
test_num = 2500
img_size = 28
img_length = img_size * img_size

kernel = {'linear':0, 'polynomial':1, 'RBF':2}

def loadData(filepath):
    with open(filepath + 'X_train.csv', 'r') as file:
        train_img = np.zeros((train_num, img_length))
        for k in range(train_num):
            train_img[k] = np.array([float(i) for i in file.readline().strip().split(',')])
    with open(filepath + 'Y_train.csv', 'r') as file:
        train_label = np.zeros((train_num))
        for k in range(train_num):
            train_label[k] = float(file.readline().strip())
    with open(filepath + 'X_test.csv', 'r') as file:
        test_img = np.zeros((test_num, img_length))
        for k in range(test_num):
            test_img[k] = np.array([float(i) for i in file.readline().strip().split(',')])
    with open(filepath + 'Y_test.csv', 'r') as file:
        test_label = np.zeros((test_num))
        for k in range(test_num):
            test_label[k] = float(file.readline().strip())
    return train_img, train_label, test_img, test_label

def compare(_iter, X, Y, opt_option, opt_acc, cur_opt):
    acc = svm_train(Y, X, cur_opt)
    if acc > opt_acc:
        return _iter+1, cur_opt, acc
    return _iter+1, opt_option, opt_acc

def gridSearch(X, Y):
    cost = [0.001, 0.01, 0.1, 1, 10]
    gamma = [1e-4, 1/img_length, 0.1, 1]
    degree = [2, 3, 4]
    coef = [0, 1, 2]
    
    opt_option = '-t 0 -v 4'
    opt_acc = 0
    _iter = 0
    
    for k, v in kernel.items():
        for c in cost:
            if k == 'linear':
                opt = f'-t {v} -v 4 -c {c}'
                _iter, opt_option, opt_acc = compare(_iter, X, Y, opt_option, opt_acc, opt)
                print(f'Iter:{_iter}, Kernel:{k}, Acc:{opt_acc:.4f}, Opt:{opt_option}\n')
            elif k == 'polynomial':
                for g in gamma:
                    for d in degree:
                        for coe in coef:
                            opt = f'-t {v} -v 4 -c {c} -g {g} -d {d} -r {coe}'
                            _iter, opt_option, opt_acc = compare(_iter, X, Y, opt_option, opt_acc, opt)
                            print(f'Iter:{_iter}, Kernel:{k}, Acc:{opt_acc}, Opt:{opt_option}\n')
            elif k == 'RBF':
                for g in gamma:
                    opt = f'-t {v} -v 4 -c {c} -g {g}'
                    _iter, opt_option, opt_acc = compare(_iter, X, Y, opt_option, opt_acc, opt)
                    print(f'Iter:{_iter}, Kernel:{k}, Acc(optimal):{opt_acc}, Opt:{opt}\n')
    return opt_option

def output(option):
    inv_ker = {v: k for k, v in kernel.items()}
    print('Optimal Kernel:', inv_ker[int(option[3])])
    print('param:', option[5:])

def linearKernel(X1, X2):
    return X1 @ X2.T

def RBFKernel(X1, X2, gamma):
    dist = np.sum(X1 ** 2, axis=1).reshape(-1, 1) + np.sum(X2 ** 2, axis=1) - 2 * X1 @ X2.T
    return np.exp(-gamma * dist)

if __name__ == "__main__":
    train_img, train_label, test_img, test_label = loadData('./data/')
    
    _type = int(input('Run which part?(1, 2, 3):'))
    
    start = time.time()
    
    # Part 1.
    if _type == 1:
        model = svm_train(train_label, train_img, '-t 0')
        result = svm_predict(test_label, test_img, model)
    
    # Part 2.
    if _type == 2:
        option = gridSearch(train_img, train_label)
        option = option.replace(option[4:9], '') # rm '-v'
        output(option)
        model = svm_train(train_label, train_img, option)
        result = svm_predict(test_label, test_img, model)
    
    # Part 3.
    if _type == 3:
        gamma = 1 / img_length
        train_kernel = linearKernel(train_img, train_img) + RBFKernel(train_img, train_img, gamma)
        test_kernel = linearKernel(test_img, train_img) + RBFKernel(test_img, train_img, gamma)
        
        # Add index in front of kernel
        train_kernel = np.hstack((np.arange(1, train_num+1).reshape(-1, 1), train_kernel))
        test_kernel = np.hstack((np.arange(1, test_num+1).reshape(-1, 1), test_kernel))
        
        model = svm_train(train_label, train_kernel, '-t 4')
        result = svm_predict(test_label, test_kernel, model)
    
    print(f"Cost:{time.time() - start} s")