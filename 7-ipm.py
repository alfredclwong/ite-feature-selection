import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt
import keras.backend as K
from utils.utils import default_env, corr_cmap
from utils.loss import mmd2
import seaborn as sns
default_env()


n = 10000
n_features = 11
n_treatments = 2

X = np.random.standard_normal((n, n_features))
lambdas = [0, 1, 2, 4]
selections = list(range(2, n_features+1))

# T = np.random.binomial(1, p=expit(X[:, 0] * X[:, 1]))
# mmds = np.zeros((len(selections), len(selections)))
# for i in range(len(selections)):
#     for j in range(len(selections)):
#         mmds[i, j] = mmd2(X[:, [selections[i], selections[j]]], T, tensor=False)
# print(mmds)
# sns.heatmap(mmds, cmap=corr_cmap(), vmin=0)

mmds = np.zeros((len(lambdas), len(selections)))
for i, lam in enumerate(lambdas):
    for j, fs in enumerate(selections):
        T = np.random.binomial(1, p=expit(lam * X[:, 0] * X[:, 1]))
        mmds[i, j] = K.eval(mmd2(K.constant(X[:, :fs+1]), K.constant(T)))

plt.figure(figsize=(4, 2.5))
for i in range(len(lambdas)):
    plt.plot([i for i in selections], mmds[i])
# for i, lam in enumerate([0, 1, 2, 4]):
#     for j, fs in enumerate([3, 4, 5, 6, 7, 8, 9, 10, 11][::-1]):
#         T = np.random.binomial(1, p=expit(lam * X[:, 0] * X[:, 1] + X[:, 2]))
#         mmds[i, j] = K.eval(mmd2(K.constant(X[:, :fs]), K.constant(T)))
# plt.plot(mmds.T, ls=':')
plt.xlim([min(selections), max(selections)])
plt.ylim([-1e-4, 6e-2])
plt.xticks(selections)
plt.yticks([0, .02, .04, .06])
plt.xlabel('Number of features')
plt.ylabel('Maximum mean discrepancy')
plt.legend([f'$\lambda = {lam}$' for lam in lambdas])
plt.savefig('../iib-diss/ipm.pdf', bbox_inches='tight')

# for t in range(n_treatments):
#     plt.scatter(X[T == t, 0], X[T == t, 1])
plt.show()
