import numpy as np
from core.helper import forward, backward, gradient_step
from core.function import add, mul, sigmoid, linear
from core.loss import cross_entropy, mse, l1loss

# Define the computational graph ()
computational_graph = (
    ["input_data", [], lambda: (np.array([[1.0, 2.0, 3.0],[1.2, 3.5, 2.0]], dtype=np.float32), None), None],  # x
    ["weight_1", [], lambda: (np.array([0.1, 0.2, 0.3], dtype=np.float32), None), None],  # weight
    ["weight_bias", [], lambda: (np.array([-0.2], dtype=np.float32), None), None],  # bias
    ["linear_1", [0, 1, 2], linear, None], # X = x * weight + bias
    ["truth", [], lambda: (np.array([4.8, 2.4], dtype=np.float32), None), None],  # y
    ["loss", [3, 4], l1loss, None], # loss = (x, y)
)

# Run forward pass
output_value = forward(computational_graph)
print("Forward pass result:", output_value)

# Run backward pass
grads = backward(computational_graph)
print(f"Gradient of node: {computational_graph[1][0]}: {grads[1]}")

# Update the weights
learning_rate = 0.002
for i in range(10000):
    grads = backward(computational_graph)
    gradient_step(computational_graph, learning_rate, grads)

# Run forward pass after weight update
output_value = forward(computational_graph)
print("Forward pass result after weight update:", output_value)