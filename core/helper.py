import numpy as np

def forward(computational_graph):
    values = []
    for i, item in enumerate(computational_graph):
        if item[0].startswith("truth"):
            return values[-1]
        input_values = [values[i] for i in item[1]]
        value, _ = item[2](*input_values)
        values.append(value)
    return values[-1]

def backward(computational_graph):
    values = []
    grads = [None] * len(computational_graph)  # Initialize grads as a list to hold arrays of gradients
    for i, item in enumerate(computational_graph):
        input_values = [values[i] for i in item[1]]
        value, grad_fn = item[2](*input_values)
        computational_graph[i][3] = grad_fn
        values.append(value)
    grads[-1] = np.ones_like(values[-1], dtype=np.float32)  # Gradient of the loss with respect to itself is 1
    for i in range(len(computational_graph)):
        op, inputs, _, grad_fn = computational_graph[len(computational_graph)-i-1]
        grad_output = grads[len(computational_graph)-i-1]
        if grad_fn is None:
            continue
        grad_inputs = grad_fn(grad_output)
        for j, grad in zip(inputs, grad_inputs):
            if grads[j] is None:
                grads[j] = grad
            else:
                grads[j] = grads[j] + grad
    return grads

def gradient_step(computational_graph, learning_rate, gradient):
    """SGD update rule"""
    for i, item in enumerate(computational_graph):
        if not item[0].startswith("weight"):  # item[0] is the name of the node
            continue
        weight, _ = item[2]()  # Extract the current weight
        gradient_value = gradient[i]
        # Perform the gradient update
        weight -= learning_rate * gradient_value
        # Ensure the updated weight retains its shape
        computational_graph[i][2] = lambda w=weight: (w, None)