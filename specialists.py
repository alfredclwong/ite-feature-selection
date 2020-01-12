import numpy as np
from keras.layers import Input, Dense, Multiply, Lambda
from keras.models import Model
from keras import backend as K
from tqdm import tqdm
import matplotlib.pyplot as plt

from Method import Method
#from invase import INVASE
from minvase import MINVASE

class Specialists(Method):
    def __init__(self, n_features, n_treatments, n_specialisms, relevant_features=None):
        super().__init__(n_features, n_treatments)
        self.n_specialisms = n_specialisms
        self.relevant_features = relevant_features
        # Learn n_specialisms feature sets for each treatment
        self.selectors = [MINVASE(self.n_features, self.n_specialisms, relevant_features=self.relevant_features)
            for t in range(self.n_treatments)]

    def train(self, X, T, Yf, n_iters):
        for t in range(self.n_treatments):
            X_split = X[T==t]
            Y_split = Yf[T==t]
            self.selectors[t].train(X_split, Y_split, n_iters)

    def predict(self, X):
        N = X.shape[0]
        Y = np.zeros((N, self.n_treatments))
        for t in range(self.n_treatments):
            Y[:, t] = self.selectors[t].predict(X, threshold=None).flatten()
        return Y
