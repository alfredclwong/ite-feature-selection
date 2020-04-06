from methods.method import Method
import numpy as np
from keras.models import Sequential
from keras.layers import Dense


class NN(Method):
    def __init__(self, n_features, n_treatments):
        super().__init__(n_features, n_treatments)
        self.models = [Sequential() for t in range(self.n_treatments)]
        for t in range(self.n_treatments):
            self.models[t].add(Dense(16, activation='relu', input_shape=(self.n_features,)))
            self.models[t].add(Dense(16, activation='relu'))
            self.models[t].add(Dense(1))
            self.models[t].compile(optimizer='adam', loss='mse', metrics=['mse'])

    def train(self, X, T, Yf, num_iters):
        for t in range(self.n_treatments):
            history = self.models[t].fit(X[T==t], Yf[T==t], epochs=num_iters, batch_size=16, verbose=0)

    def predict(self, X):
        n_test = X.shape[0]
        Y_pred = np.zeros((n_test, self.n_treatments))
        for t in range(self.n_treatments):
            Y_pred[:, t] = self.models[t].predict(X).flatten()
        return Y_pred
