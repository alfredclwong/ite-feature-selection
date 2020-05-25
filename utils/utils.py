import numpy as np
from itertools import product
import os
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity
import tensorflow as tf
from tensorflow.python.client import device_lib
import re
import pandas as pd
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# Chinese restaurant process
def crp(n, alpha=1):
    tables = []
    n_at_tables = []
    n_of_tables = 0
    for i in range(n):
        p = [x / (i + alpha) for x in n_at_tables + [alpha]]
        # p.append(1 - sum(p))
        table_choice = np.random.choice(n_of_tables + 1, p=p)
        if table_choice == n_of_tables:
            tables.append([i])
            n_at_tables.append(1)
            n_of_tables += 1
        else:
            tables[table_choice].append(i)
            n_at_tables[table_choice] += 1
    return tables


# Use KDE method to estimate a pdf with bandwidth determined by CV20 grid search
def est_pdf(x, x_grid, bws=np.linspace(0.1, 1.0, 30), cv=20):
    grid = GridSearchCV(KernelDensity(), {'bandwidth': bws}, cv=cv)
    grid.fit(x[:, None])
    # print(grid.best_params_)
    kde = grid.best_estimator_
    pdf = np.exp(kde.score_samples(x_grid[:, None]))
    return pdf


# Train-test split for {X, T, Y} observational datasets, stratified by treatment assignment
def XTY_split(X, T, Y, test_size=.2):
    n = X.shape[0]
    n_test = int(n * test_size)
    n_treatments = np.max(T) + 1

    # number of samples and number of desired test samples for each t-value
    nt = np.array([np.sum(T == t) for t in range(n_treatments)])
    nt_test = (nt * test_size).astype(int)
    nt_test[-1] = n_test - np.sum(nt_test[:-1])  # to avoid rounding errors when n_treatments > 2
    nt_test_cs = np.cumsum(nt_test)

    # idxs for each t-value
    t_idxs = [np.where(T == t)[0] for t in range(n_treatments)]
    test_idxs = np.zeros(n_test, dtype=int)
    for t in range(n_treatments):
        idxs = np.random.choice(t_idxs[t], size=nt_test[t], replace=False)
        if t == 0:
            test_idxs[:nt_test_cs[t]] = idxs
        else:
            test_idxs[nt_test_cs[t-1]: nt_test_cs[t]] = idxs
    train_idxs = np.ones(n, dtype=bool)
    train_idxs[test_idxs] = 0

    return X[train_idxs], X[test_idxs], T[train_idxs], T[test_idxs], Y[train_idxs], Y[test_idxs]


def pad_lr(X):
    n = X.shape[0]
    a = np.zeros((n, 1))
    return np.hstack([a, X, a])


# Piece together T, Yf and Ycf to get Y
def make_Y(T, Yf, Ycf):
    assert T.max() == 1
    n = Yf.shape[0]
    Y = np.zeros((n, 2))
    Y[np.arange(n), T] = Yf
    Y[np.arange(n), 1-T] = Ycf
    return Y


def high_dim_vis(X, T):
    n_treatments = T.max() + 1
    X_embedded = TSNE().fit_transform(X)
    for t in range(n_treatments):
        plt.scatter(X_embedded[T == t, 0], X_embedded[T == t, 1])
    plt.show()


# Progress saving and loading for long experiments
def continue_experiment(results_path, n_trials, trial_fn, columns=None):
    # Check progress. Number of rows (0-indexed) = number of trials already done, because of header
    progress = 0
    try:
        with open(results_path) as csv:
            for (progress, _) in enumerate(csv):
                pass
    except IOError:
        df = pd.DataFrame(columns=columns)
        df.to_csv(results_path)
    for i in tqdm(range(progress, n_trials), initial=progress, total=n_trials):
        df = pd.DataFrame(trial_fn().reshape(1, -1))
        df.to_csv(results_path, mode='a', header=False)


# n-width mean filter
def process(x, n=3):
    cs = np.cumsum(x, axis=0)
    return (cs[n:]-cs[:-n])/n


def param_search(params):
    for idxs in product(*(range(len(l)) for l in params.values())):
        yield {k: v[idxs[i]] for i, (k, v) in enumerate(params.items())}


def default_env(gpu=False):
    np.set_printoptions(
            linewidth=160,
            formatter={'float_kind': lambda x: f'{x:.4f}'},
    )
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'      # suppress warnings
    if gpu:
        gpu_device_name = tf.test.gpu_device_name()
        if gpu_device_name:
            for device_attributes in device_lib.list_local_devices():
                if device_attributes.name == gpu_device_name:
                    d = device_attributes.physical_device_desc
                    name = re.search('(?<=name: )[\w\s]+', d).group()
                    print(f'Loaded GPU: {name}')
                    break

        else:
            print('Please install GPU version of TF')
    else:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''   # disable GPU


# Nice colormap for sns correlation heatmaps
def corr_cmap():
    return sns.diverging_palette(220, 10, as_cmap=True)
