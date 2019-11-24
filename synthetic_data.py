from pandas import read_csv
import numpy as np
from scipy.special import logit

def get_X(csv_path=None, N=10000, n_features=10):
    assert csv_path or (N and n_features)
    if csv_path:
        data = read_csv(csv_path)
        features = data.keys()[:-2]
        X = twin_data[features].values
    else:
        # x[i] ~ N(mean, cov)
        # where mean = 0
        #        cov = S.T @ S
        #    S[j][k] ~ U[-1, 1]
        mean = np.zeros(n_features)
        S = np.random.uniform(-1, 1, size=(n_features, n_features))
        cov = S.T @ S  # +ve semi def
        X = np.random.multivariate_normal(mean, cov, size=N)
    return X

def sample_multinomial_sigmoid(x):
    assert len(x.shape) == 2
    U = np.random.uniform(0, 1, size=x.shape)
    Y = (x - np.log(-np.log(U))).argmax(axis=1)
    return Y

def get_T(X, n_treatments=2):
    # T[i]|x ~ Binomial(Sigmoid(w'x + n))
    # where w[j] ~ U[-0.1, 0.1]
    #       n[k] ~ N(0, 0.1)
    N, n_features = X.shape
    w = np.random.uniform(-0.1, 0.1, size=(n_features, n_treatments))
    n = np.random.normal(0, np.sqrt(0.1), size=(N, n_treatments))
    P = X@w + n
    T = sample_multinomial_sigmoid(P)
    return T

def get_Y(X, n_treatments=2):
    # Y[i]|x ~ (w'x + n)
    # where w[j][k] ~ U[-1, 1]
    #       n[p][q] ~ N(0, 0.1)
    N, n_features = X.shape
    w = np.random.uniform(-1, 1, size=(n_features, n_treatments))
    n = np.random.normal(0, np.sqrt(0.1), size=(n_treatments,))
    Y = X@w + n
    Y = Y - Y.min()
    Y = Y / Y.max()
    return Y

def synthetic_data(csv_path=None, N=10000, n_features=10, n_treatments=2):
    X = get_X(csv_path, N, n_features)
    T = get_T(X, n_treatments)
    Y = get_Y(X, n_treatments)
    return X, T, Y

if __name__ == '__main__':
    X, T, Y = synthetic_data()
    print(X)
    print(T)
    print(Y)
