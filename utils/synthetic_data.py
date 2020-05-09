import pandas as pd
import numpy as np
from scipy.special import logit, expit


def synthetic_data(csv_path=None, N=20000, n_features=11, n_treatments=2, models=None, corr=False):
    assert n_features >= 10
    if models:
        n_treatments = len(models)
    X = get_X(csv_path, N, n_features, corr)
    T = get_T(X, n_treatments)
    Y, S = get_YS(X, n_treatments, models)
    return X, T, Y, S


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
    if i == 'A':
        def model(X):
            n, n_features = X.shape
            beta = np.random.choice(5, size=n_features, p=[.5, .2, .15, .1, .05]).astype(np.float32)
            Y = np.zeros((n, 2))
            Y[:, 0] = X @ beta
            Y[:, 1] = X @ beta + 4
            return Y, beta
    elif i == 'B':
        def model(X):
            n, n_features = X.shape
            beta = np.random.choice(5, size=n_features, p=[.6, .1, .1, .1, .1]) / 10
            Y[:, 0] = np.exp((X + .5) @ beta)
            Y[:, 1] = X @ beta
            return Y, beta
    elif i == 1:
        # XOR
        def model(X):
            Y = X[:, 0] * X[:, 1]
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
        # X_7 can dominate if the coeffecient is too high (100 in L2X, 10 in INVASE, 5 => 100%)
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
    else:
        model = get_model(np.random.randint(1, 7))
    return model


def get_YS(X, n_treatments=2, models=None, binary=True):
    N, n_features = X.shape
    if not models:
        models = [t%6+1 for t in range(n_treatments)]
    n_treatments = len(models)

    Y = np.zeros((N, n_treatments))
    S = np.zeros((n_treatments, N, n_features))
    for t, model in enumerate(models):
        Y[:, t], S[t, :, :] = get_model(model)(X)
    #Y = expit(Y)
    #Y = Y - Y.min()
    #Y = Y / Y.max()
    #Y = 1 / (1. + np.exp(Y))
    if binary:
        Y = np.random.binomial(1, p=expit(Y))
    return Y, S


if __name__ == '__main__':
    X, T, Y, S = synthetic_data(models=[1, 2, 3, 4, 5, 6])
    print(X)
    print(T)
    print(Y)
    print(S)
