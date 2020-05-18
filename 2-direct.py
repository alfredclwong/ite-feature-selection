import pandas as pd
import numpy as np
from data.synthetic_data import get_ihdp
import matplotlib.pyplot as plt
from fsite.invase import Invase
from utils.utils import default_env

save_stuff = False

df = pd.read_csv('data/ihdp.csv', index_col=0)
features = list(df.columns)[1:]
X, T, Y, beta = get_ihdp('B1')
n, n_features = X.shape
n_treatments = int(np.max(T)) + 1

means = np.mean(Y, axis=0)
catt = np.mean(Y[T == 1, 1]) - np.mean(Y[T == 1, 0])
catc = np.mean(Y[T == 0, 1]) - np.mean(Y[T == 0, 0])
print('beta\t', list(beta))
print('means\t', means)
print('ATE\t', means[1] - means[0])
print('CATT\t', catt)
print('CATC\t', catc)
print('vars\t', np.var(Y, axis=0))

if save_stuff:
    binary_from = 6
    xlim_cont = np.max(np.abs(X[:, :binary_from])) * .8
    xlim_binary = np.max(np.abs(X[:, binary_from:])) * 1.2
    plt.figure(figsize=(11, 9))
    for i in range(n_features):
        ax = plt.subplot(5, 5, i+1)
        ax.set_title('$\\bf{' + f'{features[i]}}}$' if beta[i] else features[i])
        ax.scatter(X[T == 0, i], Y[T == 0, 0], s=.2)
        ax.scatter(X[T == 1, i], Y[T == 1, 1], s=.2)
        ax.set_ylim([0, 15])
        if i < binary_from:
            ax.set_xticks([-3, 0, 3])
            ax.set_xlim([-xlim_cont, xlim_cont])
        else:
            ax.set_xticks([-5, 0, 5])
            ax.set_xlim([-xlim_binary, xlim_binary])
    plt.subplots_adjust(wspace=.3, hspace=.8)
    plt.savefig('../iib-diss/ihdp-b1-sample.pdf', bbox_inches='tight')
    plt.show()

default_env()
invase = Invase(n_features, n_classes=0, lam=0.1)
invase.train(X[T == 0], Y[T == 0, 0], 5000)
print(np.array2string(beta, formatter={'float_kind': '{0:.2f}'.format}))
