from methods.Method import Method
import numpy as np


def prepend_ones_col(X):
    return np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)

class OLS(Method):
    def train(self, X, T, Yf, num_iters=None):
        n_train = X.shape[0]
        X = prepend_ones_col(X)
        XT = np.concatenate([X, T.reshape(-1, 1)], axis=1)
        self.beta_pred = np.linalg.inv(XT.T @ XT) @ XT.T @ Yf
        print(self.beta_pred)

    def predict(self, X):
        n_test = X.shape[0]
        Y_pred = np.zeros((n_test, self.n_treatments))
        X = prepend_ones_col(X)
        for t in range(self.n_treatments):
            Xt = np.concatenate([X, t * np.ones((n_test, 1))], axis=1)
            Y_pred[:, t] = Xt @ self.beta_pred
        return Y_pred
