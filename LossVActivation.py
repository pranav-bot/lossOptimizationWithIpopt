import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the loss functions
def mse_loss(pred, target):
    return ((pred - target) ** 2).mean()

def cross_entropy_loss(pred, target):
    return -torch.mean(target * torch.log(pred + 1e-10))  # Adding small epsilon to avoid log(0)

def hinge_loss(pred, target):
    return torch.mean(torch.clamp(1 - pred * target, min=0))

# Activation functions
activations = {
    "ReLU": nn.ReLU(),
    "Sigmoid": nn.Sigmoid(),
    "Tanh": nn.Tanh()
}

# Create a grid for visualization
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
Z_mse = np.zeros_like(X)
Z_ce = np.zeros_like(X)
Z_hinge = np.zeros_like(X)

# Target values
target = torch.tensor([[1.0]], dtype=torch.float32)  # Example target for MSE and CE
target_hinge = torch.tensor([[1.0]], dtype=torch.float32)  # Example target for Hinge

# Calculate loss landscape for each activation function and loss type
for activation_name, activation in activations.items():
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            inputs = torch.tensor([[X[i, j], Y[i, j]]], dtype=torch.float32)

            # MSE Loss
            pred_mse = activation(inputs)
            Z_mse[i, j] = mse_loss(pred_mse, target).item()

            # Cross-Entropy Loss (Assuming binary classification)
            pred_ce = activation(inputs)
            pred_ce = pred_ce / pred_ce.sum()  # Softmax-like normalization
            Z_ce[i, j] = cross_entropy_loss(pred_ce, target).item()

            # Hinge Loss (Assuming binary classification with outputs -1 or 1)
            pred_hinge = 2 * activation(inputs) - 1  # Scale ReLU output to [-1, 1]
            Z_hinge[i, j] = hinge_loss(pred_hinge, target_hinge).item()

    # Plotting the results in 3D
    fig = plt.figure(figsize=(18, 5))

    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot_surface(X, Y, Z_mse, cmap='viridis')
    ax1.set_title(f'MSE Loss with {activation_name} Activation')
    ax1.set_xlabel('Input 1')
    ax1.set_ylabel('Input 2')
    ax1.set_zlabel('Loss')

    ax2 = fig.add_subplot(132, projection='3d')
    ax2.plot_surface(X, Y, Z_ce, cmap='viridis')
    ax2.set_title(f'Cross-Entropy Loss with {activation_name} Activation')
    ax2.set_xlabel('Input 1')
    ax2.set_ylabel('Input 2')
    ax2.set_zlabel('Loss')

    ax3 = fig.add_subplot(133, projection='3d')
    ax3.plot_surface(X, Y, Z_hinge, cmap='viridis')
    ax3.set_title(f'Hinge Loss with {activation_name} Activation')
    ax3.set_xlabel('Input 1')
    ax3.set_ylabel('Input 2')
    ax3.set_zlabel('Loss')

    plt.tight_layout()
    plt.show()
