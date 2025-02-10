import numpy as np

def mse(x, y):
    value = np.mean((x - y) ** 2)
    def grad(grad_output):
        grad_x = 2 * grad_output * (x - y) / len(x) if not np.isscalar(x) else 2 * grad_output * (x - y)
        grad_y = -2 * grad_output * (x - y) / len(x) if not np.isscalar(x) else -2 * grad_output * (x - y)
        return [grad_x, grad_y]
    return value, grad

def cross_entropy(x, y):
    value = -np.mean(y * np.log(x) + (1 - y) * np.log(1 - x))
    def grad(grad_output):
        grad_x = grad_output * (x - y) / len(x) if not np.isscalar(x) else grad_output * (x - y)
        grad_y = -grad_output * (x - y) / len(x) if not np.isscalar(x) else -grad_output * (x - y)
        return [grad_x, grad_y]
    return value, grad

def l1loss(x, y):
    value = np.mean(np.abs(x - y))
    def grad(grad_output):
        grad_x = grad_output * np.sign(x - y) / len(x) if not np.isscalar(x) else grad_output * np.sign(x - y)
        grad_y = -grad_output * np.sign(x - y) / len(x) if not np.isscalar(x) else -grad_output * np.sign(x - y)
        return [grad_x, grad_y]
    return value, grad