import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from torchvision import datasets, transforms

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_units, activation):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(input_dim, hidden_units)
        self.output = nn.Linear(hidden_units, output_dim)
        self.activation = activation

    def forward(self, x):
        x = self.activation(self.hidden(x))
        x = self.output(x)
        return x

# Activation functions to test
activation_functions = {
    'tanh': nn.Tanh(),
    'relu': nn.ReLU(),
    'elu': nn.ELU()
}

# XOR Dataset
def xor_dataset():
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
    return X, y

# Iris Dataset
def iris_dataset():
    data = load_iris()
    X = torch.tensor(data.data, dtype=torch.float32)
    y = data.target.reshape(-1, 1)
    encoder = OneHotEncoder(sparse=False)
    y = torch.tensor(encoder.fit_transform(y), dtype=torch.float32)
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Heart Dataset (using breast cancer dataset as a proxy)
def heart_dataset():
    data = load_breast_cancer()
    X = torch.tensor(data.data, dtype=torch.float32)
    y = data.target.reshape(-1, 1)
    return train_test_split(X, y, test_size=0.2, random_state=42)

# MNIST Dataset
def mnist_dataset():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return train_dataset, test_dataset

# Function to train and evaluate the model
def train_and_evaluate(model, X_train, y_train, X_test, y_test, epochs=1000):
    criterion = nn.BCEWithLogitsLoss() if y_train.shape[1] == 1 else nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        accuracy = ((torch.sigmoid(test_outputs) > 0.5).float() == y_test).float().mean() if y_train.shape[1] == 1 else (
            test_outputs.argmax(dim=1) == y_test.argmax(dim=1)).float().mean()

    return test_loss.item(), accuracy.item()

# XOR Dataset
print("XOR Dataset")
X, y = xor_dataset()
for activation_name, activation in activation_functions.items():
    model = SimpleNN(input_dim=2, output_dim=1, hidden_units=2, activation=activation).to(device)
    loss, accuracy = train_and_evaluate(model, X.to(device), y.to(device), X.to(device), y.to(device))
    print(f"Activation: {activation_name}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

# Iris Dataset
print("\nIris Dataset")
X_train, X_test, y_train, y_test = iris_dataset()
for activation_name, activation in activation_functions.items():
    model = SimpleNN(input_dim=4, output_dim=3, hidden_units=4, activation=activation).to(device)
    loss, accuracy = train_and_evaluate(model, X_train.to(device), y_train.to(device), X_test.to(device), y_test.to(device))
    print(f"Activation: {activation_name}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

# Heart Dataset
print("\nHeart Dataset")
X_train, X_test, y_train, y_test = heart_dataset()
for activation_name, activation in activation_functions.items():
    model = SimpleNN(input_dim=X_train.shape[1], output_dim=1, hidden_units=10, activation=activation).to(device)
    loss, accuracy = train_and_evaluate(model, X_train.to(device), y_train.to(device), X_test.to(device), y_test.to(device))
    print(f"Activation: {activation_name}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

# MNIST Dataset
print("\nMNIST Dataset")
train_dataset, test_dataset = mnist_dataset()
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

for activation_name, activation in activation_functions.items():
    model = SimpleNN(input_dim=784, output_dim=10, hidden_units=128, activation=activation).to(device)
    for epoch in range(10):
        model.train()
        for X_batch, y_batch in train_loader:
            y_batch = torch.nn.functional.one_hot(y_batch, num_classes=10).float()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            optimizer.zero_grad()
            outputs = model(X_batch.view(-1, 784).to(device))
            loss = nn.CrossEntropyLoss()(outputs, y_batch.to(device))
            loss.backward()
            optimizer.step()
    
    # Evaluate on test set
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            y_batch = torch.nn.functional.one_hot(y_batch, num_classes=10).float()
            outputs = model(X_batch.view(-1, 784).to(device))
            test_loss += nn.CrossEntropyLoss()(outputs, y_batch.to(device)).item()
            pred = outputs.argmax(dim=1)
            correct += (pred == y_batch.argmax(dim=1).to(device)).sum().item()

    accuracy = correct / len(test_loader.dataset)
    print(f"Activation: {activation_name}, Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.4f}")
