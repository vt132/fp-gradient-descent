import numpy as np

def add(x, y):
    value = np.add(x, y)
    def grad(grad_output):
        grad_x = grad_output
        grad_y = grad_output
        return [grad_x, grad_y]
    return value, grad

def mul(x, y):
    value = x * y
    def grad(grad_output):
        grad_x = grad_output * y
        grad_y = grad_output * x
        return [grad_x, grad_y]
    return value, grad

def linear(x, weight, bias):
    value = np.dot(x, weight) + bias
    def grad(grad_output):
        grad_output = grad_output.reshape(-1, 1)
        grad_x = np.dot(grad_output, weight.reshape(1, -1))  # Shape (batch_size, input_dim)
        grad_weight = np.dot(x.T, grad_output).flatten()  # Shape (input_dim,)
        grad_bias = np.sum(grad_output, axis=0)  # Shape (1,)
        return [grad_x, grad_weight, grad_bias]
    return value, grad

def exp(x):
    value = np.exp(x)
    def grad(grad_output):
        grad_x = grad_output / value
        return [grad_x]
    return value, grad

def inv(x):
    value = 1 / x.value
    def grad(grad_output):
        grad_x = grad_output * (-1 / (x.value ** 2))
        return [grad_x]
    return value, grad

def sigmoid(x):
    value = 1 / (1 + np.exp(-x))
    def grad(grad_output):
        grad_x = grad_output * value * (1 - value)
        return [grad_x]
    return value, grad

def tanh(x):
    value = np.tanh(x)
    def grad(grad_output):
        grad_x = grad_output * (1 - value ** 2)
        return [grad_x]
    return value, grad

def relu(x):
    value = np.maximum(0, x)
    def grad(grad_output):
        grad_x = grad_output * (x > 0)
        return [grad_x]
    return value, grad
