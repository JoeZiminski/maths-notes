import numpy as np
import matplotlib.pyplot as plt

import torch
from torch import nn
# DataLoader is an iterable around DataSet, which stores samples and their corresponding labels.
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.FashionMNIST(
    root="data",  # root directory
    train=True,
    download=True,
    transform=ToTensor()
)
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

class MyStupidNetwork:
    def __init__(self, learning_rate=0.02):

        self.a = learning_rate

        # Define weight matrix (output dim, input dim) by convention
        self.W1 = np.random.uniform(0, 0.005, (28*28, 512))
        self.W2 = np.random.uniform(0, 0.005, (512, 512))
        self.W3 = np.random.uniform(0, 0.005, (512, 10))

    def loss(self, l3, y):
        p = self.softmax(l3)[0][y]
        return -np.log( p + 1e-15)

    def softmax(self, vec):
        C = np.max(vec)
        return np.exp(vec - C) / np.sum(np.exp(vec - C))

    def predict(self, x):
        x = x.reshape(1, x.size)
        l3 = x @ self.W1 @ self.W2 @ self.W3
        return np.argmax(self.softmax(l3))

    def update_weights(self, x, y, verbose=False):

        x = x.reshape(1, x.size)

        # Forward pass
        l1 = x @ self.W1
        l2 = l1 @ self.W2
        l3 = l2 @ self.W3

        loss = self.loss(l3, y)

        if verbose:
            print(f"Loss: {loss}")

        # Compute the derivatives
        dloss_dl3 = self.softmax(l3)  # double check this
        dloss_dl3[0][y] -= 1

        dloss_dW3 = l2.T @ dloss_dl3       # (512, 10) = (512, 1) x (1, 10)

        dloss_dl2 = dloss_dl3 @ self.W3.T  # (1, 512) = (1, 10) x (10, 512)
        dloss_dW2 = l1.T @ dloss_dl2       # (512, 512) = (512, 1) x (1, 512)

        dloss_dl1 = dloss_dl2 @ self.W2.T  # (1, 512) = (1, 512) x (512, 512)
        dloss_dW1 = x.T @ dloss_dl1        # (784, 512) = (781, 1) x (1, 512)

        self.W3 -= self.a * dloss_dW3
        self.W2 -= self.a * dloss_dW2
        self.W1 -= self.a * dloss_dW1

# Initialise and train the model (no batching)
model = MyStupidNetwork()

for i, (X, y) in enumerate(training_data):

    x = np.asarray(X[0, :, :])
    y = int(y)

    model.update_weights(x, y)

    if i % 1000 == 0:
        print(f"Training iteration: {i}")

# Check the model accuracy
results = np.empty(len(test_data))

for i, (X, y) in enumerate(test_data):

    x = np.asarray(X[0, :, :])

    results[i] = model.predict(x) == y

print(f"Percent Correct: {np.mean(results) * 100}%")