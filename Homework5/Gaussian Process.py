"""
Machine Learning Homework 5
    Gaussian Process

"""
import numpy as np
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

if __name__ == "__main__":
    X, Y = loadData()