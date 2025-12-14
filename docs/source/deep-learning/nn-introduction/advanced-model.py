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

class MyBetterNetwork:
    def __init__(self, learning_rate=0.02):

        self.a = learning_rate

        # Define weight matrix (output dim, input dim) by convention
        # Use zero-mean Xavier init (good for sigmoid)
        # This makes a huge differences vs uniform.
        self.W1 = np.random.randn(784, 512) * np.sqrt(1 / 784)
        self.W2 = np.random.randn(512, 512) * np.sqrt(1 / 512)
        self.W3 = np.random.randn(512, 10) * np.sqrt(1 / 512)

        self.b1 = np.zeros((1, 512))
        self.b2 = np.zeros((1, 512))
        self.b3 = np.zeros((1, 10))

    def loss(self, l3, y):
        p = self.softmax(l3)[0][y]
        return -np.log( p + 1e-15)

    def softmax(self, vec):
        C = np.max(vec)
        return np.exp(vec - C) / np.sum(np.exp(vec - C))

    def predict(self, x):
        # forward pass through the network
        x = x.reshape(1, x.size)

        l1_hat = x @ self.W1 + self.b1
        l1 = self.phi(l1_hat)

        l2_hat = l1 @ self.W2 + self.b2
        l2 = self.phi(l2_hat)

        l3 = l2 @ self.W3 + self.b3

        pred = np.argmax(self.softmax(l3))

        return pred, l1_hat, l1, l2_hat, l2, l3

    def phi(self, vec):
        return 1 / (1 + np.exp(-vec))

    def dphi_dvec(self, vec):
        return np.exp(-vec) / (1 + np.exp(-vec))**2

    def update_weights(self, x, y, verbose=False):

        x = x.reshape(1, x.size)

        _, l1_hat, l1, l2_hat, l2, l3 = self.predict(x)

        loss = self.loss(l3, y)

        if verbose:
            print(f"Loss: {loss}")

        # Compute the derivatives
        dloss_dl3 = self.softmax(l3)  # double check this
        dloss_dl3[0][y] -= 1

        dloss_dW3 = l2.T @ dloss_dl3
        dloss_db3 = dloss_dl3

        dloss_dl2 = dloss_dl3 @ self.W3.T                               # (1, 512) = (1, 10) x (10, 512)
        dloss_dW2 = l1.T @ (self.dphi_dvec(l2_hat) * dloss_dl2)         # (512, 512) = (512, 1) x (1, 512) * (1, 512)
        dloss_db2 = self.dphi_dvec(l2_hat) * dloss_dl2                  # (1, 512) = (512, 1) x (1, 512)

        dloss_dl1 = (dloss_dl2 * self.dphi_dvec(l2_hat)) @ self.W2.T    # (1, 512) = (1, 512) * (1, 512) x (512, 512)
        dloss_dW1 = x.T @ (self.dphi_dvec(l1_hat) * dloss_dl1)          # (784, 512) = (784, 1) x (1, 512) * (1, 512)
        dloss_db1 = self.dphi_dvec(l1_hat) * dloss_dl1                  # (1, 512) = (1, 512) * (1, 512)

        self.W3 -= self.a * dloss_dW3
        self.W2 -= self.a * dloss_dW2
        self.W1 -= self.a * dloss_dW1

        self.b3 -= self.a * dloss_db3
        self.b2 -= self.a * dloss_db2
        self.b1 -= self.a * dloss_db1

# Initialise and train the model (no batching)
model = MyBetterNetwork()

for i, (X, y) in enumerate(training_data):

    x = np.asarray(X[0, :, :])
    y = int(y)

    model.update_weights(x, y, verbose=False)

    if i % 1000 == 0:
        print(f"Training iteration: sample: {i}")

# Check the model accuracy
results = np.empty(len(test_data))

for i, (X, y) in enumerate(test_data):

    x = np.asarray(X[0, :, :])

    results[i] = model.predict(x)[0] == y

print(f"Percent Correct: {np.mean(results) * 100}%")
# 82.45%
# 82.26
# with 3 epochs:
# 84.89
