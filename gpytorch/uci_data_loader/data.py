"""
Data loader class for predefined data-sets in ./uci
"""
import torch
from math import floor

class UCI_Dataset():

    def __init__(self, dataset, seed, device, dtype=torch.float32):

        self.dataset = dataset(dtype=dtype)
        self.seed = seed
        self.device = device
        self.preprocess()

    def preprocess(self):
        X, y = self.dataset.tensors
        X_args = [X.min(0)[0], X.max(0)[0]]
        Y_args = [y.mean(), y.std()]
        X = X - X.min(0)[0]
        X = 2.0 * (X / X.max(0)[0]) - 1.0
        y -= y.mean()
        y /= y.std()

        shuffled_indices = torch.randperm(X.size(0))
        X = X[shuffled_indices, :]
        y = y[shuffled_indices]

        train_n = int(floor(0.8 * X.size(0)))

        self.train_x = X[:train_n, :].contiguous().to(self.device)
        self.train_y = y[:train_n].contiguous().to(self.device)


        self.test_x = X[train_n:, :].contiguous().to(self.device)
        self.test_y = y[train_n:].contiguous().to(self.device)
        

