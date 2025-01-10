import numpy as np


def add(x, y):
    return np.add(x, y)

def neg(x):
    return np.negative(x)

def multiply(x, y):
    return np.matmul(x, y)

def div(x, y):
    return np.divide(x, y)

def exp(x):
    return np.exp(x)

def sigmoid(x):
    return div(1,1 + np.exp(-x))