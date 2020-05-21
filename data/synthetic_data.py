import pandas as pd
import numpy as np
from scipy.special import expit


def synthetic_data(csv_path=None, N=20000, n_features=11, n_treatments=2, models=None, corr=False, noiseX=False, noiseY=False):
    assert n_features >= 10
    if models:
        n_treatments = len(models)
    X = get_X(csv_path, N, n_features, corr)
    T = get_T(X, n_treatments)
    Y, S = get_YS(X, n_treatments, models, noise=noiseY)
    if noiseX:
        X += .5 * np.random.standard_normal(X.shape)
    return X, T, Y, S


def get_ihdp_npci(small=False, val=True):
    n_trials = 100 if small else 1000
    files = 'train test'.split()
    arrays = 'x t yf ycf'.split()
    data = {f: np.load(f'data/ihdp/ihdp_npci_1-{n_trials}.{f}.npz') for f in files}
    for i in range(n_trials):
        row = {f: [data[f][a].take(i, axis=-1) for a in arrays] for f in files}
        if val:
            split = [471, 201]
            idx = np.ones(sum(split), dtype=bool)
            idx[np.random.choice(sum(split), split[0], replace=False)] = 0
            row['val'] = [a[idx] for a in row['train']]
            row['train'] = [a[~idx] for a in row['train']]
        yield row


def get_ihdp_XT():
    data = pd.read_csv('data/ihdp.csv', index_col=0).values
    assert data.shape == (985, 29)
    X = data[:, 1:-3]
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    T = data[:, 0].astype(int)
    # remove non-whites who were treated
    not_white = data[:, -3] == 0
    treated = T == 1
    keep = np.invert(not_white & treated)
    X = X[keep]
    T = T[keep]
    assert X.shape == (747, 25) and T.shape == (747,)
    return X, T


def get_ihdp_Yb(X, T, model):
    n, n_features = X.shape
    Y = np.zeros((n, 2))
    if model == 'A':
        beta = np.random.choice(5, size=n_features, p=[.5, .2, .15, .1, .05]).astype(np.float)
        Y[:, 0] = X @ beta
        Y[:, 1] = X @ beta + 4
    if model.startswith('B'):
        beta = np.random.choice(5, size=n_features, p=[.6, .1, .1, .1, .1]) / 10
        Y[:, 0] = np.exp((X + .5) @ beta)
        if model == 'B1':
            # overlap: E[tau(x_i)|T_i=1] = 4
            E_Y0_T1 = np.mean(Y[T == 1, 0])
            Xb = X @ beta
            Y[:, 1] = Xb - np.mean(Xb[T == 1]) + E_Y0_T1 + 4
        elif model == 'B2':
            # incomplete: E[tau(x_i)|T_i=0] = 4
            E_Y0_T0 = np.mean(Y[T == 0, 0])
            Xb = X @ beta
            Y[:, 1] = Xb - np.mean(Xb[T == 0]) + E_Y0_T0 + 4
    Y = np.random.normal(Y)
    return Y, beta


def get_X(csv_path=None, N=10000, n_features=10, corr=False):
    assert csv_path or (N and n_features)
    if csv_path:
        data = pd.read_csv(csv_path)
        features = data.keys()[:-2]
        X = data[features].values
    else:
        # x[i] ~ N(mean, cov)
        # where mean = 0
        #        cov = S.T @ S or I
        #    S[j][k] ~ U[-1, 1]
        mean = np.zeros(n_features)
        if corr:
            S = np.random.uniform(-1, 1, size=(n_features, n_features))
            cov = S.T @ S  # +ve semi def
        else:
            cov = np.diag(np.ones(n_features))
        X = np.random.multivariate_normal(mean, cov, size=N)
    return X


def sample_multinomial_sigmoid(x):
    assert len(x.shape) == 2
    U = np.random.uniform(0, 1, size=x.shape)
    Y = (x - np.log(-np.log(U))).argmax(axis=1)
    return Y


def get_T(X, n_treatments=2):
    # T[i]|x ~ Multinomial(Sigmoid(w'x + n))
    # where w[j] ~ U[-0.1, 0.1]
    #       n[k] ~ N(0, 0.1)
    N, n_features = X.shape
    w = np.random.uniform(-0.1, 0.1, size=(n_features, n_treatments))
    n = np.random.normal(0, np.sqrt(0.1), size=(N, n_treatments))
    P = X@w + n
    T = sample_multinomial_sigmoid(P)
    return T


def get_model(i):
    if i == 1:
        # soft XOR
        def model(X):
            Y = X[:, 0] * X[:, 1] #*10
            S = np.zeros(X.shape)
            S[:, :2] = 1
            return Y, S
    elif i == 2:
        def model(X):
            Y = np.sum(np.square(X[:, 2:6]), axis=-1) - 4
            S = np.zeros(X.shape)
            S[:, 2:6] = 1
            return Y, S
    elif i == 3:
        # X_7 can dominate if the coefficient is too high (100 in L2X, 10 in INVASE, 5 => 100%)
        def model(X):
            Y = - 5 * np.sin(2 * X[:, 6]) + 2 * np.abs(X[:, 7]) + X[:, 8] + np.exp(-X[:, 9])
            S = np.zeros(X.shape)
            S[:, 6:10] = 1
            return Y, S
    elif i == 4:
        def model(X):
            Y = np.zeros(X.shape[0])
            S = np.zeros(X.shape)
            idx = X[:, -1] < 0
            Y[idx], S[idx] = get_model(1)(X[idx])
            idx = np.invert(idx)
            Y[idx], S[idx] = get_model(2)(X[idx])
            S[:, -1] = 1
            return Y, S
    elif i == 5:
        def model(X):
            Y = np.zeros(X.shape[0])
            S = np.zeros(X.shape)
            idx = X[:, -1] < 0
            Y[idx], S[idx] = get_model(1)(X[idx])
            idx = np.invert(idx)
            Y[idx], S[idx] = get_model(3)(X[idx])
            S[:, -1] = 1
            return Y, S
    elif i == 6:
        def model(X):
            Y = np.zeros(X.shape[0])
            S = np.zeros(X.shape)
            idx = X[:, -1] < 0
            Y[idx], S[idx] = get_model(2)(X[idx])
            idx = np.invert(idx)
            Y[idx], S[idx] = get_model(3)(X[idx])
            S[:, -1] = 1
            return Y, S
    elif i == 7:
        # merged models
        def model(X):
            Y = np.zeros(X.shape[0])
            S = np.zeros(X.shape)
            idx = X[:, -1] < 0
            Y[idx], S[idx, 3:] = get_model(1)(X[idx, 3:])
            idx = np.invert(idx)
            Y[idx], S[idx] = get_model(2)(X[idx])
            return Y, S
    else:
        model = get_model(np.random.randint(1, 7))
    return model


def get_YS(X, n_treatments=2, models=None, binary=True, noise=False):
    N, n_features = X.shape
    if not models:
        models = [t % 6 + 1 for t in range(n_treatments)]

    n_treatments = len(models)
    Y = np.zeros((N, n_treatments))
    S = np.zeros((n_treatments, N, n_features))
    for t, model in enumerate(models):
        Y[:, t], S[t, :, :] = get_model(model)(X)

    if noise:
        Y += np.random.standard_normal(size=Y.shape)
    if binary:
        Y = np.random.binomial(1, p=expit(Y))
    return Y, S
