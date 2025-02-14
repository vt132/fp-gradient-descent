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
    # Run a forward pass to compute outputs and store each node’s grad function.
    values = [None] * len(computational_graph)
    for i, node in enumerate(computational_graph):
        input_vals = [values[j] for j in node[1]]
        value, grad_fn = node[2](*input_vals)
        computational_graph[i][3] = grad_fn  # Save grad function
        values[i] = value
    memo = {}
    processed = set()

    def recursive_backward(idx, grad):
        if idx in memo:
            memo[idx] += grad
        else:
            memo[idx] = grad

        grad_fn = computational_graph[idx][3]
        if grad_fn is None or idx in processed:
            return
        grad_inputs = grad_fn(memo[idx])
        processed.add(idx)
        for input_idx, grad_input in zip(computational_graph[idx][1], grad_inputs):
            recursive_backward(input_idx, grad_input)

    # Start the recursion at the output (assumed to be the last node) with a gradient of ones.
    recursive_backward(len(computational_graph) - 1, 
                       np.ones_like(values[-1], dtype=np.float32))

    # Return gradients in the order of nodes in the computational graph.
    return [memo[i] if i in memo else None for i in range(len(computational_graph))]


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