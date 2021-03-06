# -*- coding: utf-8 -*-
"""
Machine Learning Homework 1

"""

import numpy as np
import matplotlib.pyplot as plt

#input
file_path = "./"
file_name = 'test.txt'
#file_name = input('Enter file name:')
#N = int(input('Enter polynomial bases N:'))
#_lambda = float(input('Enter lambda:'))

#file data
file = open(file_path+file_name,'r')
data = []
for line in file.readlines():
    tmp = line.split(',')
    tmp = [float(t) for t in tmp]
    data.append(tmp)


