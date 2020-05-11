import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from fsite.invase import Invase
from utils.utils import process, default_env
from data.synthetic_data import get_YS, synthetic_data


default_env()

X, _, T, S_T = synthetic_data(models=[6], corr=False)
Y, S_Y = get_YS(X, models='A', binary=False)

n, n_features = X.shape
n_treatments = int(np.max(T)) + 1

n_train = int(.8 * n)
n_test = n - n_train

train_idxs = np.random.choice(n, size=n_train, replace=False)
test_idxs = np.ones(n)
test_idxs[train_idxs] = 0
test_idxs = np.where(test_idxs)[0]
X_train = X[train_idxs]
T_train = T[train_idxs]
X_test = X[test_idxs]
T_test = T[test_idxs]

invase = Invase(n_features, n_classes=n_treatments)
history = invase.train(X_train, T_train, 5000, X_test, T_test, verbose=True)
T_pred = invase.predict(X_test).T
T_pred = np.argmax(T_pred, axis=1)
T_base = invase.predict(X_test, use_baseline=True)
T_base = np.argmax(T_base, axis=1)

S_pred = invase.predict_features(X_test)
print(np.argmax(invase.predict(X), axis=1))
print(T_pred)
print(T_base)
print(T_test)
print(f'pred ({min(np.sum(S_pred, axis=1))}-{max(np.sum(S_pred, axis=1))}/{n_features}): {np.sum(T_pred==T_test)}/{n_test}')
print(f'base ({n_features}/{n_features}): {np.sum(T_base==T_test)}/{n_test}')

loss, acc, acc_test = map(lambda x: process(x, n=10), [history[s] for s in ['loss', 'acc', 'acc-test']])
loss = loss / np.max(np.abs(loss), axis=0)
plt.figure()
plt.plot(loss)
plt.plot(acc)
plt.plot(acc_test)
plt.axhline(np.sum(1-T)/n, ls=':')
plt.xlim([0, loss.shape[0]])
plt.ylim([0, 1])
plt.legend(['pred loss', 'base loss', 'sele loss', 'pred acc', 'base acc', 'pred acc (test)', 'base acc (test)'])

S_soft_pred = invase.predict_features(X_test, threshold=None)
plt.figure()
sns.heatmap(S_soft_pred[:100].T, center=0.5, vmin=0, vmax=1, cmap='gray', square=True, cbar=False)#, yticklabels=headers, linewidth=.5)
plt.show()
