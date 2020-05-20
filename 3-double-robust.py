import numpy as np
import pandas as pd
from itertools import product
import matplotlib.pyplot as plt
import seaborn as sns

from fsite.invase import Invase
from data.synthetic_data import get_ihdp_XT, get_ihdp_Yb
from utils.utils import XTY_split, default_env, est_pdf, continue_experiment

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
results_path = 'results/3-double-robust.csv'
n_trials = 1000


def trial(X, T):
    Y, beta = get_ihdp_Yb(X, T, 'B1')
    Yf = Y[np.arange(n), T]

    # Train the three estimators and make predictions
    X_train, X_test, T_train, T_test, Y_train, Y_test = XTY_split(X, T, Yf)
    invases = []
    for t in range(n_treatments):
        invases.append(Invase(n_features, 0, 2, hyperparams_deep))
    invases.append(Invase(n_features, n_treatments, .1, hyperparams_shallow))
    Y_pred = np.zeros(Y.shape)
    for t in range(n_treatments):
        train_idxs = T_train == t
        test_idxs = T_test == t
        invases[t].train(X_train[train_idxs], Y_train[train_idxs], 1000,
                         X_test[test_idxs], Y_test[test_idxs], verbose=False)
        Y_pred[:, t] = invases[t].predict(X).flatten()
    invases[-1].train(X_train, T_train, 1000, X_test, T_test, verbose=False, imbalanced=True, batch_size=128)
    prop_scores = invases[-1].predict(X)

    # Plot heatmaps for each selector
    # n_sample = 20
    # s_pred = np.array([invase.predict_features(X_test, threshold=False) for invase in invases])
    # for i in range(s_pred.shape[0]):
    #     plt.subplot(s_pred.shape[0], 1, i+1)
    #     sns.heatmap(s_pred[i, :n_sample].T, vmin=0, vmax=1, cmap='gray', square=True, cbar=False, linewidth=.5)
    # plt.show()

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
# trial(X, T)
if save_stuff:
    continue_experiment(results_path, n_trials, lambda: trial(X, T), columns)
results = pd.read_csv(results_path, index_col=0).values

print(f'{results.shape[0]} trials')
print('\t'.join(columns))
print('\t'.join(map(str, np.mean(results, axis=0).round(8))))
mae = np.array(results)
mae[:, ::2] -= results[:, 0, None]
mae[:, 1::2] -= results[:, 1, None]
mae = np.mean(np.abs(mae), axis=0)
print('\t'.join(map(lambda x: f'{str(x)}\t' if x == 0 else str(x), mae.round(8))))
print()

print('\t'.join([f'{s} ATE\t' if s == 'ipw' else f'{s} ATE' for s in headers]))
ate = results[:, 1::2] - results[:, ::2]
print('\t'.join(map(str, np.mean(ate, axis=0).round(8))))
mae = np.array(ate)
mae -= ate[:, 0, None]
mae = np.mean(np.abs(mae), axis=0)
print('\t'.join(map(lambda x: f'{str(x)}\t' if x == 0 else str(x), mae.round(8))))

# plt.figure(figsize=(4, 2.5))
# x_grid = np.linspace(0, 20, 100)
# for i in range(len(columns)):
#     if 'naive' in columns[i]:
#         continue
#     pdf = est_pdf(results[:, i], x_grid)
#     plt.plot(x_grid, pdf)
# plt.legend(list(filter(lambda s: 'naive' not in s, columns)))
# plt.xlabel('Estimate')
# plt.ylabel('Probability')
# plt.xlim([np.min(x_grid), np.max(x_grid)])
# plt.ylim(bottom=0)
# if save_stuff:
#     plt.savefig('../iib-diss/3-doubly-robust.pdf', bbox_inches='tight')
# plt.show()
