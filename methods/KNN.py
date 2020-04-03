from methods.Method import Method
import numpy as np


class KNN(Method):
    def __init__(self, n_features, n_treatments, k):
        super().__init__(n_features, n_treatments)
        self.k = k

    def train(self, X, T, Yf, num_iters=None):
        self.XT = np.concatenate([X, T.reshape(-1, 1)], axis=1)
        self.Yf = Yf

    def predict(self, X):
        n_test = X.shape[0]
        Y_pred = np.zeros((n_test, self.n_treatments))
        for t in range(self.n_treatments):
            Xt = np.concatenate([X, t * np.ones((n_test, 1))], axis=1)
            for i in range(n_test):
                distances = np.sum(np.square(self.XT - Xt[i]), axis=1)
                k_nearest_neighbours = np.argsort(distances)[:self.k]
                Y_pred[i, t] = np.mean(self.Yf[k_nearest_neighbours])
        return Y_pred
