import numpy as np

from fsite.invase import Invase
from data.synthetic_data import get_ihdp_XT, get_ihdp_Yb
from utils.utils import XTY_split, default_env

hyperparams = {
    'h_layers_pred':    2,
    'h_dim_pred':       lambda x: 100,  # noqa 272
    'h_layers_base':    2,
    'h_dim_base':       lambda x: 100,  # noqa 272
    'h_layers_sel':     1,
    'h_dim_sel':        lambda x: 2*x,  # noqa 272
    'optimizer':        'adam'
}

X, T = get_ihdp_XT()
n, n_features = X.shape
n_treatments = np.max(T) + 1
Y, beta = get_ihdp_Yb(X, T, 'B1')
Yf = Y[np.arange(n), T]

default_env()

X_train, X_test, T_train, T_test, Y_train, Y_test = XTY_split(X, T, Yf)
invases = {'prop': Invase(n_features, n_treatments, 0.1, hyperparams)}
for t in range(n_treatments):
    invases[t] = Invase(n_features, 0, 2, hyperparams)
invases['prop'].train(X_train, T_train, 500, X_test, T_test, verbose=True)
prop_scores = invases['prop'].predict(X)
Y_pred = np.zeros(Y.shape)
for t in range(n_treatments):
    train_idxs = T_train == t
    test_idxs = T_test == t
    invases[t].train(X_train[train_idxs], Y_train[train_idxs], 1000, X_test[test_idxs], Y_test[test_idxs])
    Y_pred[:, t] = invases[t].predict(X).flatten()

# lack of overlap in treated group means that our prop score function may violate positivity - fix here
eps = 1e-5
not_pos = prop_scores < eps
prop_scores[not_pos] = eps
prop_scores[not_pos[:, np.arange(n_treatments-1, -1, -1)]] = 1 - eps

Y_mean = np.mean(Y, axis=0)
Y_mean_pred = np.mean(Y_pred, axis=0)
Y_mean_pred_aipw = np.zeros(n_treatments)
for t in range(n_treatments):
    idxs = T == t
    aipw = (Yf * idxs - Y_pred[:, t] * (idxs - prop_scores[:, t])) / prop_scores[:, t]
    print(np.partition(prop_scores[:, t], 5)[:5])
    print(np.partition(aipw, -5)[-5:])
    Y_mean_pred_aipw[t] = np.mean(aipw)
ate = Y_mean_pred[1] - Y_mean_pred[0]
print(Y_mean)
print(Y_mean_pred)
print(Y_mean_pred_aipw)
print(ate)
