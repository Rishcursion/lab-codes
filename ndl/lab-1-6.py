import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# 1. Implementing a basic neuron with different activation functions
class Neuron:
    def __init__(self, weights, bias=0):
        self.weights = np.array(weights)
        self.bias = bias
    
    def activate(self, x, activation='sigmoid'):
        net_input = np.dot(self.weights, x) + self.bias
        if activation == 'binary_step':
            return 1 if net_input >= 0 else 0
        elif activation == 'linear':
            return net_input
        elif activation == 'sigmoid':
            return 1 / (1 + np.exp(-net_input))
        elif activation == 'tanh':
            return np.tanh(net_input)
        elif activation == 'relu':
            return max(0, net_input)
        elif activation == 'leaky_relu':
            return net_input if net_input > 0 else 0.01 * net_input
        elif activation == 'softmax':
            return np.exp(net_input) / np.sum(np.exp(net_input))
        else:
            raise ValueError("Unsupported activation function")

# 2. Hebbian Learning vs. PCA Visualization
def hebbian_vs_pca():
    np.random.seed(42)
    data = np.random.randn(100, 2)
    
    # Hebbian Learning (Simplified PCA)
    hebbian_weights = np.random.rand(2)
    learning_rate = 0.01
    hebbian_trajectory = []
    
    for x in data:
        hebbian_weights += learning_rate * np.dot(x, hebbian_weights) * x
        hebbian_trajectory.append(hebbian_weights.copy())
    
    hebbian_trajectory = np.array(hebbian_trajectory)
    
    # PCA (Principal Component Analysis)
    cov_matrix = np.cov(data.T)
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    pca_direction = eigenvectors[:, np.argmax(eigenvalues)]
    
    # Plot
    plt.figure(figsize=(6, 6))
    plt.scatter(data[:, 0], data[:, 1], alpha=0.3, label='Data')
    plt.quiver(0, 0, hebbian_trajectory[-1][0], hebbian_trajectory[-1][1], color='r', angles='xy', scale_units='xy', scale=1, label='Hebbian Learning')
    plt.quiver(0, 0, pca_direction[0], pca_direction[1], color='b', angles='xy', scale_units='xy', scale=1, label='PCA Direction')
    plt.legend()
    plt.title("Hebbian Learning vs. PCA")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid()
    plt.show()

hebbian_vs_pca()

# 3. Self-Organizing Map (SOM) implementation
class SOM:
    def __init__(self, grid_size, dim):
        self.grid_size = grid_size
        self.dim = dim
        self.weights = np.random.randn(grid_size, grid_size, dim)
    
    def train(self, data, epochs=100, lr=0.1):
        for _ in range(epochs):
            for x in data:
                distances = np.linalg.norm(self.weights - x, axis=2)
                bmu = np.unravel_index(np.argmin(distances), (self.grid_size, self.grid_size))
                self.weights[bmu] += lr * (x - self.weights[bmu])
    
    def plot_weights(self):
        plt.figure(figsize=(6,6))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                plt.scatter(self.weights[i, j, 0], self.weights[i, j, 1], c='blue')
        plt.title("SOM Weights Visualization")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.grid()
        plt.show()

# Example usage of SOM
if __name__ == "__main__":
    data = np.random.randn(100, 2)  # Sample data
    som = SOM(grid_size=5, dim=2)
    som.train(data)
    som.plot_weights()

# 4. Implementing gate operations using perceptron
class Perceptron:
    def __init__(self, input_size):
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn(1)
    
    def forward(self, x):
        return 1 if np.dot(x, self.weights) + self.bias >= 0 else 0
    
    def train(self, x, y, lr=0.1, epochs=100):
        for _ in range(epochs):
            for i in range(len(x)):
                y_pred = self.forward(x[i])
                error = y[i] - y_pred
                self.weights += lr * error * x[i]
                self.bias += lr * error

# 5. XOR using MLP
class MLP_XOR(nn.Module):
    def __init__(self):
        super(MLP_XOR, self).__init__()
        self.hidden = nn.Linear(2, 2)
        self.output = nn.Linear(2, 1)
        self.activation = nn.Sigmoid()
    
    def forward(self, x):
        x = torch.tanh(self.hidden(x))
        return self.activation(self.output(x))

# 6. Radial Basis Function Network for XOR
class RBF_Network:
    def __init__(self, centers, sigma=1.0):
        self.centers = np.array(centers)
        self.sigma = sigma
        self.weights = np.random.randn(len(centers))
    
    def rbf(self, x, c):
        return np.exp(-np.linalg.norm(x - c)**2 / (2 * self.sigma**2))
    
    def forward(self, x):
        return np.sum([w * self.rbf(x, c) for w, c in zip(self.weights, self.centers)])
