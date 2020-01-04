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
from OLS import OLS
from KNN import KNN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress warnings
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # disable GPU
np.set_printoptions(linewidth=160)

# Read X and T, standardise X
data = pd.read_csv('ihdp.csv').values
X = data[:, 2:-3]
X = (X - np.mean(X)) / np.std(X)
T = data[:, 1].astype(np.int)

# Introduce selection bias
not_white = data[:, -3] == 0
treated = T == 1
keep = np.invert(not_white & treated)  # remove non-whites who were treated
X = X[keep]
T = T[keep]
control = T == 0
treated = T == 1

# Extract dims
n, n_features = X.shape
n_treatments = int(np.max(T)) + 1
assert n_treatments == 2

# Construct synthetic outcomes
Y = np.zeros((n, 2))
setting = 'A'
if setting == 'A':
    # Generate linear response surfaces with non-heterogenous treatment effects
    beta = np.random.choice(5, size=n_features, p=[0.5, 0.2, 0.15, 0.1, 0.05]).astype(np.float32)
    Y[:, 0] = X @ beta
    Y[:, 1] = X @ beta + 4
elif setting == 'B':
    # Generate non-linear response surfaces with heterogenous treatment effects
    beta = np.random.choice(5, size=n_features, p=[0.6, 0.1, 0.1, 0.1, 0.1]) / 10
    Y[:, 0] = np.exp((X + 0.5) @ beta)
    Y[:, 1] = X @ beta

# Add N(0, 1) noise
Y[:, 0] = np.random.multivariate_normal(Y[:, 0], np.eye(n))
Y[:, 1] = np.random.multivariate_normal(Y[:, 1], np.eye(n))
print(f'Setting {setting}')
print(beta)
np.set_printoptions(formatter={'float_kind': lambda x: f'{x:.1f}'})

# test:train split
n_train = int(n * 0.8)
n_test = n - n_train
X_train = X[:n_train]
T_train = T[:n_train]
Yf_train = np.choose(T_train, Y[:n_train].T)
Y_train = Y[:n_train]
X_test = X[n_train:]
Y_test = Y[n_train:]

PEHEs = {}
methods = {
        'ols':      (OLS(n_features, n_treatments),     None            ),
        'knn':      (KNN(n_features, n_treatments, 5),  None            ),
#        'invase':   (INVASE(n_features, n_treatments),  [10000, 10000]  ),
}

for name, (method, n_iters) in methods.items():
    print(name)
    method.train(X_train, T_train, Yf_train, n_iters)
    Y_pred = method.predict(X_test)
    PEHEs[name] = PEHE(Y_pred, Y_test)
    print(f'R^2 = {1 - np.sum(np.square(Y_pred-Y_test)) / np.sum(np.square(Y_test - np.mean(Y_test)))}')

#'''
# remove irrelevant features
X = X[:, beta!=0]
n_features = X.shape[1]
X_train = X[:n_train]
T_train = T[:n_train]
Yf_train = np.choose(T_train, Y[:n_train].T)
Y_train = Y[:n_train]
X_test = X[n_train:]
Y_test = Y[n_train:]
#'''

for name, (method, n_iters) in methods.items():
    method.train(X_train, T_train, Yf_train, n_iters)
    Y_pred = method.predict(X_test)
    PEHEs[name] = PEHE(Y_pred, Y_test)
    print(f'R^2 = {1 - np.sum(np.square(Y_pred-Y_test)) / np.sum(np.square(Y_test - np.mean(Y_test)))}')

'''
print('lasso')
#'''

'''
print('nn')
models = [Sequential() for i in range(n_treatments)]
for i in range(n_treatments):
    #models[i].add(Dense(16, activation='relu', input_shape=(n_features,)))
    #models[i].add(Dense(16, activation='relu'))
    models[i].add(Dense(1, input_shape=(n_features,)))
    models[i].compile(optimizer='sgd', loss='mse', metrics=['mse'])
    history = models[i].fit(X_train[T_train==i], Yf_train[T_train==i], epochs=20, validation_data=(X_test, Y_test[:, i]), verbose=0)
    #plt.plot(history.history['mse'])
    #plt.show()
    print(history.history['mse'][-1])
    Y_pred[:, i] = models[i].predict(X_test).flatten()
PEHEs['nn'] = PEHE(Y_pred, Y_test)
#'''

'''
print('invase')
print(X_train)
print(Y_train)
print(beta.reshape(1,-1))
invase = INVASE(n_features, n_treatments, gamma=0.01)
invase.train([5000, 20000], X_train, Y_train, X_test, Y_test)
Y_pred, ss = invase.predict(X_test)
PEHEs['invase'] = PEHE(Y_pred, Y_test)
#'''

print(PEHEs)
