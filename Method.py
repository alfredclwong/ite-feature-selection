from abc import ABC, abstractmethod
import numpy as np


class Method(ABC):
    def __init__(self, n_features, n_treatments):
        self.n_features = n_features
        self.n_treatments = n_treatments

    @abstractmethod
    def train(self, X, T, Yf, n_iters=None):
        pass

    @abstractmethod
    def predict(self, X, T):
        pass
