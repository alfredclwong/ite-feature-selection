## try using invase to predict treatment assignment
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from fsite.invase import Invase
from utils.utils import process
from utils.synthetic_data import get_YS, synthetic_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress warnings
os.environ['CUDA_VISIBLE_DEVICES'] = ''   # disable gpu
np.set_printoptions(linewidth=160)

n_iters = 10000

df = pd.read_csv('data/ihdp.csv')
data = df.values; headers = df.columns[2:-3]
X = data[:, 2:-3]
X = (X - np.mean(X,axis=0)) / np.std(X, axis=0)
#sns.heatmap(np.corrcoef(X.T), vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200), square=True)
#plt.show()
#T = data[:, 1].astype(np.int)
#T = X[:, 0] > np.mean(X[:, 0])
T, S = get_YS(X, models=[1])  # INVASE syn1
T = T.flatten()

'''
not_white = data[:, -3] == 0
treated = T == 1
keep = np.invert(not_white & treated)
X = X[keep]
T = T[keep]
control = T == 0
treated = T == 1
'''

n, n_features = X.shape
n_treatments = int(np.max(T)) + 1

n_train = int(.9 * n)
n_test = n - n_train

train_idxs = np.random.choice(n, size=n_train, replace=False)
test_idxs = np.ones(n)
test_idxs[train_idxs] = 0
test_idxs = np.where(test_idxs)[0]
X_train = X[train_idxs]
T_train = T[train_idxs]
X_test = X[test_idxs]
T_test = T[test_idxs]

invase = Invase(n_features, n_classes=n_treatments, lam=.1)
history = invase.train(X_train, T_train, n_iters, X_test, T_test, verbose=True)
T_pred = invase.predict(X_test)
T_pred = np.argmax(T_pred, axis=1)
T_base = invase.predict(X_test, use_baseline=True)
T_base = np.argmax(T_base, axis=1)

S_pred = invase.predict_features(X_test)
print(np.argmax(invase.predict(X), axis=1))
print(T_pred); print(T_base); print(T_test)
print(f'pred ({min(np.sum(S_pred, axis=1))}-{max(np.sum(S_pred, axis=1))}/{n_features}): {np.sum(T_pred==T_test)}/{n_test}')
print(f'base ({n_features}/{n_features}): {np.sum(T_base==T_test)}/{n_test}')

loss, acc, acc_test = map(lambda x: process(x, n=10), [history[s] for s in ['loss', 'acc', 'acc-test']])
loss = loss / np.max(np.abs(loss), axis=0)
plt.figure()
plt.plot(loss); plt.plot(acc); plt.plot(acc_test)
plt.axhline(np.sum(1-T)/n, ls=':')
plt.xlim([0, loss.shape[0]]); plt.ylim([0, 1])
plt.legend(['pred loss', 'base loss', 'sele loss', 'pred acc', 'base acc', 'pred acc (test)', 'base acc (test)'])

S_soft_pred = invase.predict_features(X_test, threshold=None)
plt.figure()
sns.heatmap(S_soft_pred.T, center=0.5, vmin=0, vmax=1, cmap='gray', yticklabels=headers, square=True, cbar=False, linewidth=.5)
plt.show()