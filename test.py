import pandas as pd
import numpy as np
from keras.layers import Input, Dense, Activation
from keras.models import Sequential
import keras.backend as K
import tensorflow as tf
from matplotlib import pyplot as plt
import os
from tqdm import tqdm
from scipy.linalg import null_space

from metrics import PEHE, r2
from methods.OLS import OLS
from methods.KNN import KNN
from methods.NN import NN
from methods.invase import INVASE
from methods.specialists import Specialists
from methods.ganite import GANITE

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress warnings
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # disable GPU
np.set_printoptions(linewidth=160)

# Read X and T, standardise X
data = pd.read_csv('data/ihdp.csv').values
X = data[:, 2:-3]
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
print(np.mean(X, axis=0))
print(np.std(X, axis=0))
T = data[:, 1].astype(np.int)

# Introduce selection bias
not_white = data[:, -3] == 0
treated = T == 1
keep = np.invert(not_white & treated)  # remove non-whites who were treated
X = X[keep]
T = T[keep]
control = T == 0
treated = T == 1

print(null_space(X))

# Pad X with noise
padding = 0
if padding:
    X = np.concatenate([X, np.random.randn(X.shape[0], padding)], axis=1)

# Extract dims
n, n_features = X.shape
n_treatments = int(np.max(T)) + 1
assert n_treatments == 2

# TODO generate multiple beta corresponding to different feature sets for extraction
# Construct synthetic outcomes
Y = np.zeros((n, 2))
for setting in ['A', 'B']:
    if setting == 'A':
        # Generate linear response surfaces with non-heterogenous treatment effects
        beta = np.random.choice(5, size=n_features, p=[0.5, 0.2, 0.15, 0.1, 0.05]).astype(np.float32)
        if padding:
            beta[-padding:] = 0
        Y[:, 0] = X @ beta
        Y[:, 1] = X @ beta + 4
    elif setting == 'B':
        # Generate non-linear response surfaces with heterogenous treatment effects
        beta = np.random.choice(5, size=n_features, p=[0.6, 0.1, 0.1, 0.1, 0.1]) / 10
        if padding:
            beta[-padding:] = 0
        Y[:, 0] = np.exp((X + 0.5) @ beta)
        Y[:, 1] = X @ beta
    relevant_features = (beta > 0).astype(np.float32)

    # Add N(0, 1) noise
    Y[:, 0] = np.random.multivariate_normal(Y[:, 0], np.eye(n))
    Y[:, 1] = np.random.multivariate_normal(Y[:, 1], np.eye(n))
    print(f'Setting {setting}')
    print(np.array2string(np.concatenate([[0], beta]), formatter={'float': lambda x: "{0:0.1f}".format(x)}))
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

    n_specialisms = 1
    PEHEs = [{}, {}]
    r2s = [{}, {}]
    methods = {
            'ols':      (OLS(n_features, n_treatments),     None            ),
            'knn':      (KNN(n_features, n_treatments, 5),  None            ),
            #'nn':       (NN(n_features, n_treatments),      1000            ),
            #'spec':     (Specialists(n_features, n_treatments, n_specialisms, relevant_features=relevant_features), 5000),
            'ganite':   (GANITE(n_features, n_treatments),  [5000, 2000]    ),
    }

    for name, (method, n_iters) in methods.items():
        print(name)
        method.train(X_train, T_train, Yf_train, n_iters)
        Y_pred_in = method.predict(X_train)
        Y_pred_out = method.predict(X_test)
        PEHEs[0][name] = [PEHE(Y_train, Y_pred_in), PEHE(Y_test, Y_pred_out)]
        r2s[0][name] = [r2(Y_train, Y_pred_in), r2(Y_test, Y_pred_out)]
    print('PEHE (in/out)')
    print(PEHEs[0])
    #print('r squared (in/out)')
    #print(r2s[0])

    perfect = False

    if perfect:
        #'''
        # remove irrelevant features
        print('with perfect features')
        print(np.concatenate([[0], beta[beta!=0]]))
        X = X[:, beta!=0]
        n_features = X.shape[1]
        X_train = X[:n_train]
        T_train = T[:n_train]
        Yf_train = np.choose(T_train, Y[:n_train].T)
        Y_train = Y[:n_train]
        X_test = X[n_train:]
        Y_test = Y[n_train:]
        #'''

        methods = {
                'ols':      (OLS(n_features, n_treatments),     None            ),
                'knn':      (KNN(n_features, n_treatments, 5),  None            ),
                #'nn':       (NN(n_features, n_treatments),      1000            ),
                #'spec':     (Specialists(n_features, n_treatments, n_specialisms, relevant_features=relevant_features), 5000),
                'ganite':   (GANITE(n_features, n_treatments),  [5000, 2000]    ),
        }

        for name, (method, n_iters) in methods.items():
            method.train(X_train, T_train, Yf_train, n_iters)
            Y_pred_in = method.predict(X_train)
            Y_pred_out = method.predict(X_test)
            PEHEs[1][name] = [PEHE(Y_train, Y_pred_in), PEHE(Y_test, Y_pred_out)]
            r2s[1][name] = [r2(Y_train, Y_pred_in), r2(Y_test, Y_pred_out)]

        print('PEHE (in/out)')
        print(PEHEs[1])
        #print('r squared (in/out)')
        #print(r2s[1])

        '''
        n_methods = len(methods.keys())
        x = np.arange(n_methods)
        y_out = [[PEHEs[i][name][1] for name in methods.keys()] for i in range(len(PEHEs))]
        y_in = [[PEHEs[i][name][0] for name in methods.keys()] for i in range(len(PEHEs))]
        ax = plt.subplot(121)
        ax.bar(x, y_out[0], width=0.4, align='edge')
        ax.bar(x+0.4, y_out[1], width=0.4, align='edge')
        ax = plt.subplot(122)
        ax.bar(x, y_in[0], width=0.4, align='edge')
        ax.bar(x+0.4, y_in[1], width=0.4, align='edge')
        #plt.show()
        '''

    '''
    print('lasso')
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
