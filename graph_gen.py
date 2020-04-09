import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from graphviz import Digraph
from methods.invase import INVASE

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
np.set_printoptions(formatter={'float_kind': lambda x: f'{x:.1f}'})
ord_A = ord('A')

'''
N = 10000
n_features = 6
biases = np.random.uniform(size=n_features)
stds = np.random.uniform(size=n_features)

# Generate causal graph
E_arr = np.random.choice(2, size=n_features*(n_features-1)//2)
#E_arr = E_arr * np.around(np.random.uniform(size=E_arr.size), 2)
E = np.zeros((n_features, n_features))
E[np.tril_indices(n_features, k=-1, m=n_features)] = E_arr
dot = Digraph()
for i in range(n_features):
    dot.node(chr(ord_A+i), xlabel=f'{stds[i]:.2f}')
    for j in range(i):
        if E[i, j]:
            dot.edge(chr(ord_A+j), chr(ord_A+i), label=f'{E[i, j]}')
dot.view()

# Estimate predictive power scores
scores = np.zeros(n_features)
for i in range(n_features-1, -1, -1):
    scores += E[i] + 0.5 * scores[i] * (E[i]>0)  # predictions obscured by noise and other edges as path gets longer
print(E)
print(scores)

# Generate data
roots, = np.where(np.sum(E, axis=1)==0)
print(roots)
X = np.zeros((N, 1+n_features))
X[:, roots+1] = np.random.choice(2, size=(N, roots.size))
#X[:, 0] = 1
E = np.c_[biases, E]
for i in range(n_features):
    #X[:, i+1] = np.random.normal(np.exp(np.abs(X @ E[i].T - 0.5)), stds[i])  # non-invertible, non-linear
    if i in roots:
        continue
    X[:, i+1] = (X @ E[i].T) % 2
X = X[:, 1:]; E = E[:, 1:]  # get rid of bias columns
'''

import pandas as pd
# Read X and T, standardise X
df = pd.read_csv('data/ihdp.csv')
data = df.values; headers = df.columns[2:-3]
X = data[:, 2:-3]
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
T = data[:, 1].astype(np.int)

# Introduce selection bias
not_white = data[:, -3] == 0
treated = T == 1
keep = np.invert(not_white & treated)  # remove non-whites who were treated
X = X[keep]
T = T[keep]

# Extract dims
n, n_features = X.shape
n_treatments = int(np.max(T)) + 1
assert n_treatments == 2

print(X)

# Predict predictive power scores
corr_matrix = np.corrcoef(X, rowvar=False)
invase_matrix = np.zeros((n_features, n_features))
for i in range(n_features):
    invase = INVASE(n_features=n_features-1, lam=0.05)
    other_features = [j for j in range(n_features) if i!=j]
    history = invase.train(X[:, other_features], X[:, i], 8000, verbose=True)
    S = invase.predict_features(X[:, other_features], threshold=None)
    sns.heatmap(S[:100].T, center=0.5, vmin=0, vmax=1, cmap='gray', square=True, cbar=False, yticklabels=headers[other_features])
    plt.title(headers[i])
    plt.show()
    invase_matrix[i, other_features] = np.mean(invase.predict_features(X[:, other_features]), axis=0)
print(corr_matrix)
print(invase_matrix)
#print(E)
