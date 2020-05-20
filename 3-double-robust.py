import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
import matplotlib.pyplot as plt

from fsite.invase import Invase
from data.synthetic_data import get_ihdp_XT, get_ihdp_Yb
from utils.utils import XTY_split, default_env, est_pdf

hyperparams_shallow = {
    'h_layers_pred':    1,
    'h_dim_pred':       lambda x: 100,  # noqa 272
    'h_layers_base':    1,
    'h_dim_base':       lambda x: 100,  # noqa 272
    'h_layers_sel':     1,
    'h_dim_sel':        lambda x: 100,  # noqa 272
    'optimizer':        'adam'
}
hyperparams_deep = {
    'h_layers_pred':    2,
    'h_dim_pred':       lambda x: 100,  # noqa 272
    'h_layers_base':    2,
    'h_dim_base':       lambda x: 100,  # noqa 272
    'h_layers_sel':     1,
    'h_dim_sel':        lambda x: 100,  # noqa 272
    'optimizer':        'adam'
}
save_stuff = True
headers = 'true naive regr ipw aipw'.split()
subheaders = 'E[Y0] E[Y1]'.split()
columns = [f'{a} {b}' for a, b in product(headers, subheaders)]
n_trials = 1000
results_path = 'results/3-double-robust.csv'


def trial(X, T):
    Y, beta = get_ihdp_Yb(X, T, 'B1')
    Yf = Y[np.arange(n), T]

    # Train the three estimators
    X_train, X_test, T_train, T_test, Y_train, Y_test = XTY_split(X, T, Yf)
    invases = {'prop': Invase(n_features, n_treatments, 0.1, hyperparams_shallow)}
    for t in range(n_treatments):
        invases[t] = Invase(n_features, 0, 2, hyperparams_deep)
    invases['prop'].train(X_train, T_train, 500, X_test, T_test, verbose=False)
    prop_scores = invases['prop'].predict(X)
    Y_pred = np.zeros(Y.shape)
    for t in range(n_treatments):
        train_idxs = T_train == t
        test_idxs = T_test == t
        invases[t].train(X_train[train_idxs], Y_train[train_idxs], 1000,
                         X_test[test_idxs], Y_test[test_idxs], verbose=False)
        Y_pred[:, t] = invases[t].predict(X).flatten()

    # Calculate various ATE estimates
    Y_means = np.zeros((len(headers), 2))
    Y_means[0] = np.mean(Y, axis=0)
    Y_means[1] = np.array([np.mean(Yf[T == t]) for t in range(n_treatments)])
    Y_means[2] = np.mean(Y_pred, axis=0)
    for t in range(n_treatments):
        idxs = T == t
        Y_means[3, t] = np.mean(Yf[idxs] / prop_scores[idxs, t])
        Y_means[4, t] = np.mean((Yf - Y_pred[:, t]) * idxs / prop_scores[np.arange(n), T] + Y_pred[:, t])
    return Y_means


default_env(gpu=True)
X, T = get_ihdp_XT()
n, n_features = X.shape
n_treatments = np.max(T) + 1

# TODO make a generic, less memory-intensive implementation of progress saving already
progress = 0
results = np.zeros((n_trials, len(headers) * len(subheaders)))
try:
    _results = pd.read_csv(results_path, index_col=0).values
    assert(_results.shape[1] == results.shape[1])
    progress = _results.shape[0]
    if progress < n_trials:
        results[:progress] = _results
        raise IndexError
    results = _results
except (IOError, IndexError):
    for i in tqdm(range(progress, n_trials)):
        results[i] = trial(X, T).flatten()
        df = pd.DataFrame(results[:i+1], columns=columns)
        if save_stuff:
            df.to_csv(results_path)

print('\t'.join(columns))
print('\t'.join(map(str, np.mean(results, axis=0).round(8))))
mae = np.array(results)
mae[:, ::2] -= results[:, 0, None]
mae[:, 1::2] -= results[:, 1, None]
mae = np.mean(np.abs(mae), axis=0)
print('\t'.join(map(lambda x: f'{str(x)}\t' if x == 0 else str(x), mae.round(8))))

plt.figure(figsize=(4, 2.5))
x_grid = np.linspace(0, 20, 100)
for i in range(len(columns)):
    if 'naive' in columns[i]:
        continue
    pdf = est_pdf(results[:, i], x_grid)
    plt.plot(x_grid, pdf)
plt.legend(list(filter(lambda s: 'naive' not in s, columns)))
plt.xlabel('Estimate')
plt.ylabel('Probability')
plt.xlim([np.min(x_grid), np.max(x_grid)])
plt.ylim(bottom=0)
if save_stuff:
    plt.savefig('../iib-diss/3-doubly-robust.pdf', bbox_inches='tight')
plt.show()
