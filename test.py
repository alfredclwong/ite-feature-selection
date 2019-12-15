import pandas as pd
import numpy as np
from keras.layers import Input, Dense, Activation
from keras.models import Sequential
import keras.backend as K
import tensorflow as tf
from matplotlib import pyplot as plt
import os
from tqdm import tqdm

from invase import INVASE
from metrics import PEHE

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

data = pd.read_csv('ihdp.csv').values
X = data[:, 2:-3]
X = (X - np.mean(X)) / np.std(X)
T = data[:, 1].astype(np.int)

not_white = data[:, -3] == 0
treated = T == 1
keep = np.invert(not_white & treated)  # remove non-whites who were treated
X = X[keep]
T = T[keep]
control = T == 0
treated = T == 1

n, n_features = X.shape
n_treatments = int(np.max(T)) + 1
assert n_treatments == 2

Y = np.zeros((n, 2))
setting = 'B'
if setting == 'A':
    beta = np.random.choice(5, size=n_features, p=[0.5, 0.2, 0.15, 0.1, 0.05]).astype(np.float32)
    Y[:, 0] = X @ beta
    Y[:, 1] = X @ beta + 4
elif setting == 'B':
    beta = np.random.choice(5, size=n_features, p=[0.6, 0.1, 0.1, 0.1, 0.1]) / 10
    Y[:, 0] = np.exp((X + 0.5) @ beta)
    Y[:, 1] = X @ beta
Y[:, 0] = np.random.multivariate_normal(Y[:, 0], np.eye(n))
Y[:, 1] = np.random.multivariate_normal(Y[:, 1], np.eye(n))
print(beta)

n_train = int(n * 0.8)
n_test = n - n_train
X_train = X[:n_train]
T_train = T[:n_train]
Yf_train = np.choose(T_train, Y[:n_train].T)
Y_train = Y[:n_train]
X_test = X[n_train:]
Y_test = Y[n_train:]

PEHEs = {}

XT_train = np.concatenate([X_train, T_train.reshape((n_train, 1))], axis=1)
XT_test_control = np.concatenate([X_test, np.zeros((n_test, 1))], axis=1)
XT_test_treated = np.concatenate([X_test, np.ones((n_test, 1))], axis=1)

np.set_printoptions(formatter={"float_kind": lambda x: f"{x:.2f}"})

print('ols')
XT_train_ols = np.concatenate([np.ones((n_train, 1)), XT_train], axis=1)
beta_pred = np.linalg.inv(XT_train_ols.T @ XT_train_ols) @ XT_train_ols.T @ Yf_train
Y_pred = np.zeros(Y_test.shape)
Y_pred[:, 0] = np.concatenate([np.ones((n_test, 1)), XT_test_control], axis=1) @ beta_pred
Y_pred[:, 1] = np.concatenate([np.ones((n_test, 1)), XT_test_treated], axis=1) @ beta_pred
PEHEs['ols'] = PEHE(Y_pred, Y_test)

print('knn')
k = 5
for i in range(n_test):
    distances = np.sum(np.square(XT_train - XT_test_control[i]), axis=1)
    nearest_neighbours = np.argsort(distances)[:k]
    Y_pred[i, 0] = np.mean(Yf_train[nearest_neighbours])
    distances = np.sum(np.square(XT_train - XT_test_treated[i]), axis=1)
    nearest_neighbours = np.argsort(distances)[:k]
    Y_pred[i, 1] = np.mean(Yf_train[nearest_neighbours])
PEHEs['knn'] = PEHE(Y_pred, Y_test)

print('nn')
models = [Sequential() for i in range(n_treatments)]
for i in range(n_treatments):
    models[i].add(Dense(16, activation='relu', input_shape=(n_features,)))
    models[i].add(Dense(16, activation='relu'))
    models[i].add(Dense(1))
    models[i].compile(optimizer='adam', loss='mse', metrics=['mse'])
    history = models[i].fit(X_train[T_train==i], Yf_train[T_train==i], epochs=200, validation_data=(X_test, Y_test[:, i]), verbose=0)
    #plt.plot(history.history['mse'])
    #plt.show()
    print(history.history['mse'][-1])
    Y_pred[:, i] = models[i].predict(X_test).flatten()
PEHEs['nn'] = PEHE(Y_pred, Y_test)

print('invase')
print(X_train)
print(Y_train)
print(beta)
invase = INVASE(n_features, n_treatments, gamma=0.01)
invase.train([2000, 10000], X_train, Y_train, X_test, Y_test)
Y_pred, ss = invase.predict(X_test)
PEHEs['invase'] = PEHE(Y_pred, Y_test)

print(PEHEs)
