import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import dirichlet

from fsite.invase import Invase
from data.synthetic_data import synthetic_data, get_YS
from utils.metrics import tpr_fdr, roc, auc
from utils.utils import default_env


def run_invase(X, Y, S):
    n, n_features = X.shape
    n_classes = int(np.max(Y)) + 1
    YS = np.concatenate([Y, S], axis=-1)
    X_train, X_test, YS_train, YS_test = train_test_split(X, YS, test_size=.2)
    (Y_train, S_train), (Y_test, S_test) = map(lambda arr: np.hsplit(arr, [1]), [YS_train, YS_test])

    invase = Invase(n_features, n_classes)
    invase.train(X_train, Y_train, 2000, X_test, Y_test, S_test)

    Y_pred = invase.predict(X_test)
    Y_base = invase.predict(X_test, use_baseline=True)
    S_pred = invase.predict_features(X_test)

    X_str, Y_str, Y_pred_str, S_pred_str = map(np.array2string, [X, Y, Y_pred, S_pred.astype(int)])
    print('\n'.join(['X', X_str, 'Y', Y_str, 'Y_pred', Y_pred_str, 'S_pred', S_pred_str]))

    # Test: tpr and fdr within acceptable ranges (replicate the paper)
    tpr, fdr = tpr_fdr(S_test, S_pred)
    tpr10, fdr10 = tpr_fdr(S_test[:, :-1], S_pred[:, :-1])
    print(f'TPR: {tpr*100:.1f}% ({tpr10*100:.1f}%)\nFDR: {fdr*100:.1f}% ({fdr10*100:.1f}%)')

    # Test: AUC-ROC improved in predictor
    r_base = roc(Y_test, Y_base[:, 1])
    r_pred = roc(Y_test, Y_pred[:, 1])
    a_base = auc(Y_test, Y_base[:, 1])
    a_pred = auc(Y_test, Y_pred[:, 1])
    plt.plot(r_base[:, 0], r_base[:, 1])
    plt.plot(r_pred[:, 0], r_pred[:, 1])
    plt.legend([f'base ({a_base:.2f})', f'pred ({a_pred:.2f})'])
    plt.show()

    # Plot feature selection heatmap for diagnostics
    s_pred = invase.predict_features(X_test, threshold=None)
    sns.heatmap(s_pred[:100].T, center=.5, vmin=0, vmax=1, cmap='gray', square=True, cbar=False, linewidth=.5)
    plt.show()


# 1. Models with overlapping feature sets
def overlap_test():
    # Start with Syn 6: model 2 on X3-6, model 3 on X7-10, switch on X11
    X, _, Y, S = synthetic_data(models=[6])
    S = S[0]
    # Apply model 1 on X10-11, switch on X3
    idxs, = np.where(X[:, 2] > 0)
    S[idxs] = 0
    S[idxs, 2] = 1
    Y[idxs], S[idxs, -2:] = get_YS(X[idxs, -2:], models=[1])
    # Now X3 and X11 are both switches and models, X10 is in two models

    run_invase(X, Y, S)


# 2. Measurement noise + high correlation clones for each feature
def clones():
    X, _, Y, S = synthetic_data(models=[1], noiseY=True)
    S = S[0]
    n, n_features = X.shape

    rho = .9
    Z = np.random.multivariate_normal(np.zeros(n_features), np.eye(n_features), size=n)
    X_perp = Z - np.einsum('ij,ij->j', X, Z) / np.einsum('ij,ij->j', X, X) * X
    X_clone = rho * np.std(X_perp) * X + np.sqrt(1 - rho**2) * np.std(X) * X_perp
    X = np.concatenate([X, X_clone], axis=-1)
    S = np.concatenate([S, np.zeros(S.shape)], axis=-1)
    Z = np.random.multivariate_normal(np.zeros(n_features), np.eye(n_features), size=n)
    X_perp = Z - np.einsum('ij,ij->j', X, Z) / np.einsum('ij,ij->j', X, X) * X
    X_clone = rho * np.std(X_perp) * X + np.sqrt(1 - rho**2) * np.std(X) * X_perp

    run_invase(X, Y, S)


# 3. Measurement noise + perfect split predictors for each feature
def split(n_splits=10):
    X, _, Y, S = synthetic_data(models=[1], noiseY=True)
    S = S[0]
    n, n_features = X.shape

    d = dirichlet.rvs(np.ones(n_splits)/n_splits, size=n)
    X_splits = np.concatenate([X * d[:, i, np.newaxis] for i in range(n_splits)][:-1], axis=-1)
    X = np.concatenate([X, X_splits], axis=-1)
    S = np.concatenate([S, np.zeros(X_splits.shape)], axis=-1)

    noiseX = np.random.standard_normal(X.shape)
    noiseX[:, 2:] = 0
    noiseX += np.random.standard_normal(X.shape) / n_splits
    X += noiseX

    run_invase(X, Y, S)


# 4. Graded effect modifiers
def graded():
    pass


# 5. Long causal paths


if __name__ == '__main__':
    default_env()
    # overlap_test()
    split()
