import pandas as pd
import numpy as np
from scipy.special import logit, expit


def synthetic_data(csv_path=None, N=10000, n_features=10, n_treatments=2, models=None):
    assert n_features >= 10
    if models:
        n_treatments = len(models)
    X = get_X(csv_path, N, n_features)
    T = get_T(X, n_treatments)
    Y = get_Y(X, n_treatments, models)
    return X, T, Y


def get_X(csv_path=None, N=10000, n_features=10):
    assert csv_path or (N and n_features)
    if csv_path:
        data = pd.read_csv(csv_path)
        features = data.keys()[:-2]
        X = data[features].values
    else:
        # x[i] ~ N(mean, cov)
        # where mean = 0
        #        cov = S.T @ S
        #    S[j][k] ~ U[-1, 1]
        mean = np.zeros(n_features)
        #S = np.random.uniform(-1, 1, size=(n_features, n_features))
        #cov = S.T @ S  # +ve semi def

        # actually for feature selection we want independence
        cov = np.diag(np.random.uniform(0, 2, size=n_features))
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
        def model(X):
            return np.exp(X[:, 0] * X[:, 1])
    elif i == 2:
        def model(X):
            return np.exp(np.sum(np.square(X[:, 2:5]), axis=-1)/4)
    elif i == 3:
        def model(X):
            return -np.sin(2 * X[:, 6]) + 2 * np.abs(X[:, 7]) + X[:, 8] + np.exp(-X[:, 9])
    elif i == 4:
        def model(X):
            Y = np.zeros(X.shape[0])
            idx = X[:, 0] < 0
            Y[idx] = get_model(1)(X[idx])
            idx = np.invert(idx)
            Y[idx] = get_model(2)(X[idx])
            return Y
    elif i == 5:
        def model(X):
            Y = np.zeros(X.shape[0])
            idx = X[:, 0] < 0
            Y[idx] = get_model(1)(X[idx])
            idx = np.invert(idx)
            Y[idx] = get_model(3)(X[idx])
            return Y
    elif i == 6:
        def model(X):
            Y = np.zeros(X.shape[0])
            idx = X[:, 0] < 0
            Y[idx] = get_model(2)(X[idx])
            idx = np.invert(idx)
            Y[idx] = get_model(3)(X[idx])
            return Y
    else:
        model = get_model(np.random.randint(1, 7))
    return model


def get_Y(X, n_treatments=2, models=None):
    N, n_features = X.shape
    if not models:
        models = [t%6+1 for t in range(n_treatments)]

    Y = np.zeros((N, n_treatments))
    for t, model in enumerate(models):
        Y[:, t] = get_model(model)(X)
    Y = expit(Y)
    Y = Y - Y.min()
    Y = Y / Y.max()
    return Y


if __name__ == '__main__':
    X, T, Y = synthetic_data(models=[1, 2, 3])
    print(X)
    print(T)
    print(Y)
