import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from scipy.special import expit

from fsite.invase import Invase
from utils.utils import default_env
from data.synthetic_data import synthetic_data, get_YS

default_env()

hyperparams = {
    'h_layers_pred':    1,
    'h_dim_pred':       lambda x: 100,  # noqa 272
    'h_layers_base':    1,
    'h_dim_base':       lambda x: 100,  # noqa 272
    'h_layers_sel':     1,
    'h_dim_sel':        lambda x: 2*x,  # noqa 272
#    'lam':              0.1,
    'optimizer':        'adam'
}


def predict(X, T, Y, Y_true=None):
    T = T.astype(int)
    n_treatments = len(set(T))
    n, n_features = X.shape

    X_train, X_test, T_train, T_test = train_test_split(X, T, test_size=.2)
    invase = Invase(n_features, n_treatments, hyperparams=hyperparams)
    invase.train(X_train, T_train, 1000, X_test, T_test, save_history=False)

    '''
    loss, acc, acc_test = map(lambda x: process(x, n=3), [history[s] for s in ['loss', 'acc', 'acc-test']])
    loss = loss / np.max(np.abs(loss), axis=0)
    plt.figure()
    plt.plot(loss)
    plt.plot(acc)
    plt.plot(acc_test)
    plt.axhline(np.mean(1-T), ls=':')
    plt.xlim([0, loss.shape[0]])
    plt.ylim([0, 1])
    plt.legend(['pred loss', 'base loss', 'sele loss', 'pred acc', 'base acc', 'pred acc (test)', 'base acc (test)'])
    plt.show()
    '''

    '''
    S_pred = invase.predict_features(X_test, threshold=None)
    plt.figure()
    sns.heatmap(S_pred[:100].T, center=0.5, vmin=0, vmax=1, cmap='gray', square=True, cbar=False)#, yticklabels=headers, linewidth=.5)
    plt.show()
    '''

    propensity_scores = invase.predict(X, threshold=None)
    #propensity_scores = np.zeros((n, 2))
    #propensity_scores[:, 1] = expit(X[:, 0])
    #propensity_scores[:, 0] = 1 - propensity_scores[:, 1]
    weights = np.zeros(n)
    pT = [np.mean(T == 0)]
    pT.append(1 - pT[0])
    for i in range(n):
        weights[i] = pT[T[i]] / propensity_scores[i, T[i]]
    print(max(weights))

    ate_naive = np.mean(Y[T == 1]) - np.mean(Y[T == 0])
    Yw = Y * weights
    ate_ipw = np.mean(Yw[T == 1]) - np.mean(Yw[T == 0])
    ate_true = np.mean(Y_true[:, 1] - Y_true[:, 0])
    print('naive', ate_naive)
    print('ipw', ate_ipw)
    print('true', ate_true)

    predictions = np.zeros((n, n_treatments))
    predictions[np.arange(n), T] = Y
    predictions[np.arange(n), 1 - T] = Y + (1 - 2 * T) * ate_ipw
    return predictions


X = np.random.standard_normal(size=(10000, 11))
n, n_features = X.shape
T = np.random.binomial(1, p=expit(X[:, 0] * X[:, 1] + X[:, 2]))
#T = np.random.binomial(1, p=expit(X[:, 0]))
n_treatments = int(np.max(T)) + 1

beta = np.zeros((n_features, 2))
beta[:, 0] = 1 + np.random.standard_normal(size=n_features)
beta[:, 1] = beta[:, 0]
Y = X @ beta
Y[:, 1] += 1
Yf = Y[np.arange(n), T]
print(np.mean(Y[:, 1] - Y[:, 0]))
print(np.mean(Yf[T == 1]) - np.mean(Yf[T == 0]))
predict(X, T, Yf, Y_true=Y)
