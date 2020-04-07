## try using invase to predict treatment assignment

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from methods.invase import INVASE

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress warnings
np.set_printoptions(linewidth=160)

data = pd.read_csv('data/ihdp.csv').values
X = data[:, 2:-3]
X = (X - np.mean(X,axis=0)) / np.std(X, axis=0)
T = data[:, 1].astype(np.int)

not_white = data[:, -3] == 0
treated = T == 1
keep = np.invert(not_white & treated)
X = X[keep]
T = T[keep]
control = T == 0
treated = T == 1

n, n_features = X.shape
n_treatments = int(np.max(T)) + 1

n_train = int(n * 0.8)
n_test = n - n_train

# try a few random train/test splits
for i in range(20):
    train_idxs = np.random.choice(n, size=n_train, replace=False)
    test_idxs = np.ones(n)
    test_idxs[train_idxs] = 0
    test_idxs = np.where(test_idxs)[0]
    X_train = X[train_idxs]
    T_train = T[train_idxs]
    X_test = X[test_idxs]
    T_test = T[test_idxs]

    invase = INVASE(n_features, n_classes=n_treatments, lam=0.01)
    history = invase.train(X_train, T_train, 10000, verbose=False)
    T_pred = invase.predict(X_test)
    T_pred = np.argmax(T_pred, axis=1)
    T_base = invase.predict(X_test, use_baseline=True)
    T_base = np.argmax(T_base, axis=1)

    s = invase.predict_features(X_test)
    #print(T_pred)
    print(f'pred ({min(np.sum(s, axis=1))}-{max(np.sum(s, axis=1))}/{n_features}): {np.sum(T_pred==T_test)}/{n_test}')
    #print(T_base)
    print(f'base ({n_features}/{n_features}): {np.sum(T_base==T_test)}/{n_test}')
    #print(T_test)

    history['loss'] /= np.max(np.abs(history['loss']), axis=0)
    plt.plot(history['loss'])
    plt.show()
