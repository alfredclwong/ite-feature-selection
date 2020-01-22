import numpy as np
from keras.layers import Input, Dense, Multiply, Lambda
from keras.models import Model
from keras import backend as K
from tqdm import tqdm
import matplotlib.pyplot as plt
from functools import reduce

from Method import Method
#from invase import INVASE
from minvase import MINVASE

use_predictors = False

class Specialists(Method):
    def __init__(self, n_features, n_treatments, n_specialisms, relevant_features=None):
        super().__init__(n_features, n_treatments)
        self.n_specialisms = n_specialisms
        self.relevant_features = relevant_features
        # Learn n_specialisms feature sets for each treatment
        self.specialists = [MINVASE(self.n_features, self.n_specialisms, relevant_features=self.relevant_features)
            for t in range(self.n_treatments)]
        self.ss = []
        self.models = []

    def train(self, X, T, Yf, n_iters):
        for t in range(self.n_treatments):
            X_split = X[T==t]
            Y_split = Yf[T==t]
            self.specialists[t].train(X_split, Y_split, n_iters)
            if not use_predictors:
                ss = [np.mean(s, axis=0).astype(np.float32) for s in self.specialists[t].predict_features(X, threshold=None)]
                s = reduce(lambda s1, s2: np.logical_or(s1, s2), [s>0.3 for s in ss])
                self.ss.append(s)
                print(s.astype(np.float32))
                X_split = X_split * s
                Xs = Input((np.sum(s),))
                H = Dense(self.n_features, activation='relu')(Xs)
                H = Dense(self.n_features, activation='relu')(H)
                Yt = Dense(1)(H)
                self.models.append(Model(Xs, Yt))
                self.models[-1].compile(loss='mse', optimizer='adam')
                self.models[-1].fit(X_split[:, s], Y_split, epochs=1000, batch_size=16, verbose=0)

    def predict(self, X):
        N = X.shape[0]
        Y = np.zeros((N, self.n_treatments))
        for t in range(self.n_treatments):
            if use_predictors:
                Y[:, t] = self.specialists[t].predict(X).flatten()
            else:
                Y[:, t] = self.models[t].predict(X[:, self.ss[t]]).flatten()
        return Y
