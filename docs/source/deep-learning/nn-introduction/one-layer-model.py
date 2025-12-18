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

class MyBasicNetwork:
    def __init__(self, learning_rate=0.02):

        self.a = learning_rate

        # Define weight matrix (output dim, input dim) by convention
        # Use zero-mean Xavier init (good for sigmoid, it has little
        # effect here as we don't use activation functions,
        # but useful for comparison.)
        self.W = np.random.randn(784, 10) * np.sqrt(1 / 10)

    def loss(self, l, y):
        p = self.softmax(l)[0][y]
        return -np.log( p + 1e-15)

    def softmax(self, vec):
        C = np.max(vec)
        return np.exp(vec - C) / np.sum(np.exp(vec - C))

    def predict(self, x):
        # forward pass through the network
        x = x.reshape(1, x.size)

        l = x @ self.W

        pred = np.argmax(self.softmax(l))

        return pred, x, l

    def update_weights(self, x, y, verbose=False):

        _, x, l = self.predict(x)

        loss = self.loss(l, y)

        if verbose:
            print(f"Loss: {loss}")

        # Compute the derivatives
        dloss_dl = self.softmax(l)
        dloss_dl[0][y] -= 1

        dloss_dW = x.T @ dloss_dl       # (512, 10) = (512, 1) x (1, 10)

        self.W -= self.a * dloss_dW

# Initialise and train the model (no batching)
model = MyBasicNetwork()



for i, (X, y) in enumerate(training_data):

    x = np.asarray(X[0, :, :])
    y = int(y)

    model.update_weights(x, y)

    if i % 1000 == 0:
        print(f"Training iteration: sample: {i}")

# Check the model accuracy
results = np.empty(len(test_data))

for i, (X, y) in enumerate(test_data):

    x = np.asarray(X[0, :, :])

    results[i] = model.predict(x)[0] == y

print(f"Percent Correct: {np.mean(results) * 100}%")
