import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
import multiprocessing as mp
from tqdm import tqdm  # Import tqdm for progress bars

# Set random seed for reproducibility
torch.manual_seed(42)

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 28 * 28  # 784
hidden_size = 128
num_classes = 10
num_epochs = 5
batch_size = 32
learning_rate = 0.001

# Load the MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Define the neural network model
class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(-1, input_size)  # Flatten the input
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Function to initialize weights
def initialize_weights(model, zero_weights=False):
    if zero_weights:
        for param in model.parameters():
            nn.init.constant_(param, 0.0)
    else:
        for param in model.parameters():
            nn.init.normal_(param, mean=0.0, std=0.01)

# Evaluating the model
def evaluate_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating Model", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_accuracy = correct / total
    return test_accuracy

# Objective function for optimization
def objective(params, model_class):
    model = model_class().to(device)  # Move model to GPU
    idx = 0
    for param in model.parameters():
        numel = param.numel()
        param.data = torch.tensor(params[idx:idx + numel], dtype=torch.float32, device=device).view(param.size())
        idx += numel

    # Calculate loss on a small subset for optimization
    loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss += nn.CrossEntropyLoss()(outputs, labels).item()
    return loss / len(train_loader)

# Worker function for parallel optimization
def worker(initial_params):
    model_class = NeuralNet
    result = minimize(lambda p: objective(p, model_class), initial_params, method='trust-constr')
    return result.fun, result.x

# Random Search with IPOPT optimization in parallel
def random_search_with_ipopt(num_iterations=10):
    best_loss = float('inf')
    best_model = None

    for _ in tqdm(range(num_iterations), desc="Optimizing with IPOPT"):
        model = NeuralNet().to(device)
        # Random initialization
        initial_params = torch.cat([param.data.view(-1) for param in model.parameters()]).cpu().numpy()

        # Use multiprocessing to optimize in parallel
        with mp.Pool(processes=mp.cpu_count()) as pool:
            results = pool.map(worker, [initial_params] * mp.cpu_count())

        # Evaluate results
        for loss, params in results:
            if loss < best_loss:
                best_loss = loss
                best_model = NeuralNet().to(device)
                idx = 0
                for param in best_model.parameters():
                    numel = param.numel()
                    param.data = torch.tensor(params[idx:idx + numel], dtype=torch.float32, device=device).view(param.size())
                    idx += numel

    return best_model

# Plot training history with progress bar
def plot_training_history(losses, accuracies):
    plt.figure(figsize=(12, 4))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label='Train Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

# Training function with progress bar
def train_model(zero_weights=False):
    model = NeuralNet().to(device)
    initialize_weights(model, zero_weights)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_losses = []
    train_accuracies = []

    for epoch in tqdm(range(num_epochs), desc="Training Model"):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, leave=False):
            optimizer.zero_grad()  # Zero the gradients
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')

    return train_losses, train_accuracies, model

# Benchmarking all three methods
if __name__ == "__main__":
    print("Training with Zero Weights:")
    losses_zero, accuracies_zero, model_zero = train_model(zero_weights=True)
    evaluate_model(model_zero, test_loader)
    plot_training_history(losses_zero, accuracies_zero)

    print("Training with Random Weights:")
    losses_random, accuracies_random, model_random = train_model(zero_weights=False)
    evaluate_model(model_random, test_loader)
    plot_training_history(losses_random, accuracies_random)

    print("Training with Random Search and IPOPT Optimization:")
    model_ipopt = random_search_with_ipopt()
    test_accuracy = evaluate_model(model_ipopt, test_loader)
    print(f'Test Accuracy (Random Search + IPOPT): {test_accuracy * 100:.2f}%')
