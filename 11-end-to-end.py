from fsite.ournet import OurNet
from fsite.invase import Invase
from data.synthetic_data import partitioned_X, get_YS, get_ihdp_npci
from data.ibm import IBM
import numpy as np
from utils.utils import default_env
from utils.metrics import PEHE
from itertools import chain
import sys
import time
import statsmodels.api as sm
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import os
default_env(gpu=True)
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(f'results/{time.strftime("%Y%m%d-%H%M%S")}.txt', 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
# sys.stdout = Logger()
n_treatments = 2


def full_trial(X_train, T_train, Yf_train, Ycf_train,
               X_val, T_val, Yf_val, Ycf_val,
               X_test, T_test, Yf_test, Ycf_test):
    print('invase')
    # Disentanglement stage
    invases = [
        Invase(n_features, n_treatments, .1),
        Invase(n_features, 0, .1),
        Invase(n_features, 0, .1),
    ]
    selection = []
    for invase, target in zip(invases, [T_train, Yf_train[:, None], Yf_train[:, None]]):
        invase.train(X_train, target, 1000, silent=False)
        S_pred = invase.predict_features(X_train, .5).mean(axis=0) > .5
        selection.append(np.where(S_pred)[0].tolist())
    selection = [
        list(set(selection[0]) - set(selection[1]) - set(selection[2])),
        list(set(selection[0])),
        list(set(selection[1] + selection[2]) - set(selection[0]))
    ]
    selection[1] = list(set(selection[1]) - set(selection[0]))
    print(selection)
    if len(selection[0]) + len(selection[1]) == 0 or len(selection[1]) + len(selection[2]) == 0:
        print('skipping')
        return
    ournet = OurNet([len(a) for a in selection])
    Xs_sorted = [np.hstack([X[:, sel] for sel in selection]) for X in [X_train, X_val, X_test]]
    ournet.train(Xs_sorted[0], T_train, Yf_train, 1000, Ycf=Ycf_train,
                 val_data=(Xs_sorted[1], T_val, Yf_val, Ycf_val),
                 test_data=(Xs_sorted[2], T_test, Yf_test, Ycf_test), verbose=True)


def oracle_trial(X_train, T_train, Yf_train, Ycf_train,
                 X_val, T_val, Yf_val, Ycf_val,
                 X_test, T_test, Yf_test, Ycf_test,
                 markov):
    print('oracle')
    selection = [list(set(markov[0]) - set(markov[1])), [], list(set(markov[1]) - set(markov[0]))]  # TCY
    selection[1] = list(set(markov[0]) - set(selection[0]))
    print(selection)
    ournet = OurNet([len(a) for a in selection])
    Xs_sorted = [np.hstack([X[:, sel] for sel in selection]) for X in [X_train, X_val, X_test]]
    ournet.train(Xs_sorted[0], T_train, Yf_train, 1000, Ycf=Ycf_train,
                 val_data=(Xs_sorted[1], T_val, Yf_val, Ycf_val),
                 test_data=(Xs_sorted[2], T_test, Yf_test, Ycf_test), verbose=True)


def xc_trial(X_train, T_train, Yf_train, Ycf_train,
             X_val, T_val, Yf_val, Ycf_val,
             X_test, T_test, Yf_test, Ycf_test):
    print('all xc')
    ournet = OurNet([0, X_train.shape[1], 0])
    ournet.train(X_train, T_train, Yf_train, 1000, Ycf=Ycf_train,
                 val_data=(X_val, T_val, Yf_val, Ycf_val),
                 test_data=(X_test, T_test, Yf_test, Ycf_test), verbose=True)


def nn_trial(X_train, T_train, Yf_train, Ycf_train,
             X_val, T_val, Yf_val, Ycf_val,
             X_test, T_test, Yf_test, Ycf_test):
    print('nn')
    def fit_nn(X, Y):
        model = Sequential()
        model.add(Dense(100, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X, np.zeros(X.shape[0]), epochs=50, batch_size=128, verbose=0)
        return model
    models = [fit_nn(X_train[T_train == t], Yf_train[T_train == t])
              for t in range(n_treatments)]

    pred_train = np.hstack([m.predict(X_train) for m in models])
    Y_train = np.zeros((X_train.shape[0], n_treatments))
    Y_train[np.arange(X_train.shape[0]), T_train] = Yf_train
    Y_train[np.arange(X_train.shape[0]), 1-T_train] = Ycf_train
    print(f'pehe (train) {PEHE(Y_train, pred_train):.4f}')
    pred_test = np.hstack([m.predict(X_test) for m in models])
    Y_test = np.zeros((X_test.shape[0], n_treatments))
    Y_test[np.arange(X_test.shape[0]), T_test] = Yf_test
    Y_test[np.arange(X_test.shape[0]), 1-T_test] = Ycf_test
    print(f'pehe (test) {PEHE(Y_test, pred_test):.4f}')


def ols_trial(X_train, T_train, Yf_train, Ycf_train,
              X_val, T_val, Yf_val, Ycf_val,
              X_test, T_test, Yf_test, Ycf_test):
    print('ols reg')
    Xb_train, Xb_test = map(sm.add_constant, [X_train, X_test])
    models = [sm.OLS(Yf_train[T_train == t], Xb_train[T_train == t]).fit_regularized()
              for t in range(n_treatments)]

    pred_train = np.vstack([m.predict(Xb_train) for m in models]).T
    Y_train = np.zeros((X_train.shape[0], n_treatments))
    Y_train[np.arange(X_train.shape[0]), T_train] = Yf_train
    Y_train[np.arange(X_train.shape[0]), 1-T_train] = Ycf_train
    print(f'pehe (train) {PEHE(Y_train, pred_train):.4f}')
    pred_test = np.vstack([m.predict(Xb_test) for m in models]).T
    Y_test = np.zeros((X_test.shape[0], n_treatments))
    Y_test[np.arange(X_test.shape[0]), T_test] = Yf_test
    Y_test[np.arange(X_test.shape[0]), 1-T_test] = Ycf_test
    print(f'pehe (test) {PEHE(Y_test, pred_test):.4f}')


'''
1. Synthetic data
Sample partitioned X and use random causal models for T and Y. Report disentanglement
accuracy, PEHE, and results from other methods (all XC, neural net, linear regression)
'''
# n_trials = 100
# n = 20000
# n_train = int(.7 * n)
# n_test = int(.2 * n)
# n_features = 15
# for i in range(n_trials):
#     print(f'{i+1}/{n_trials}')
#     X, partitions = partitioned_X(n, n_features, 4)
#     T, ST = get_YS(X, models=np.random.choice(3, 1).tolist(), permute=True)
#     T = T.flatten()
#     Y, SY = get_YS(X, models=np.random.choice(3, 2).tolist(), permute=True, binary=False, noise=True)

#     part_map = {}
#     for j, part in enumerate(partitions):
#         for x in part:
#             part_map[x] = j
#     markov = [
#         np.where(ST[0].mean(axis=0))[0].tolist(),
#         list(set(np.where(SY[0].mean(axis=0))[0].tolist() + np.where(SY[1].mean(axis=0))[0].tolist()))
#     ]
#     correlated = [list(chain(*[partitions[j] for j in set([part_map[x] for x in blanket])])) for blanket in markov]
#     correlated = [list(set(correlated[j]) - set(markov[j])) for j in range(2)]
#     print(markov)
#     print(correlated)

#     Yf = Y[np.arange(n), T]
#     Ycf = Y[np.arange(n), 1-T]
#     full_trial(X[:n_train], T[:n_train], Yf[:n_train], Ycf[:n_train],
#                X[n_train:-n_test], T[n_train:-n_test], Yf[n_train:-n_test], Ycf[n_train:-n_test],
#                X[-n_test:], T[-n_test:], Yf[-n_test:], Ycf[-n_test:])
#     print()
#     oracle_trial(X[:n_train], T[:n_train], Yf[:n_train], Ycf[:n_train],
#                  X[n_train:-n_test], T[n_train:-n_test], Yf[n_train:-n_test], Ycf[n_train:-n_test],
#                  X[-n_test:], T[-n_test:], Yf[-n_test:], Ycf[-n_test:], markov)
#     print()
#     xc_trial(X[:n_train], T[:n_train], Yf[:n_train], Ycf[:n_train],
#              X[n_train:-n_test], T[n_train:-n_test], Yf[n_train:-n_test], Ycf[n_train:-n_test],
#              X[-n_test:], T[-n_test:], Yf[-n_test:], Ycf[-n_test:])
#     print()
#     ols_trial(X[:n_train], T[:n_train], Yf[:n_train], Ycf[:n_train],
#               X[n_train:-n_test], T[n_train:-n_test], Yf[n_train:-n_test], Ycf[n_train:-n_test],
#               X[-n_test:], T[-n_test:], Yf[-n_test:], Ycf[-n_test:])
#     print()
#     nn_trial(X[:n_train], T[:n_train], Yf[:n_train], Ycf[:n_train],
#              X[n_train:-n_test], T[n_train:-n_test], Yf[n_train:-n_test], Ycf[n_train:-n_test],
#              X[-n_test:], T[-n_test:], Yf[-n_test:], Ycf[-n_test:])
#     print()

'''
2. IHDP
Report PEHE and results from other methods
'''
# for i, row in enumerate(get_ihdp_npci()):
#     print(f'{i+1}/1000')
#     X_train, T_train, Yf_train, Ycf_train = row['train']
#     X_val, T_val, Yf_val, Ycf_val = row['val']
#     X_test, T_test, Yf_test, Ycf_test = row['test']
#     n, n_features = X_train.shape

#     full_trial(X_train, T_train, Yf_train, Ycf_train,
#                X_val, T_val, Yf_val, Ycf_val,
#                X_test, T_test, Yf_test, Ycf_test)
#     print()
#     xc_trial(X_train, T_train, Yf_train, Ycf_train,
#              X_val, T_val, Yf_val, Ycf_val,
#              X_test, T_test, Yf_test, Ycf_test)
#     print()

'''
3. IBM
Report PEHE and results from other methods
'''
paths_50k = {
    'covariates':       'data/LBIDD-50k/x.csv',
    'factuals':         'data/LBIDD-50k/scaling/',
    'counterfactuals':  'data/LBIDD-50k/scaling/',
    'predictions':      'data/results-50k/',
}
paths = paths_50k
index_col = 'sample_id'
x = pd.read_csv(paths['covariates'], index_col=index_col)
ufids = [s[:-7] for s in os.listdir(paths['counterfactuals']) if s.endswith('_cf.csv')]
for i, ufid in enumerate(ufids):
    print(f'{i+1}/{len(ufids)}')
    Yf = pd.read_csv(os.path.join(paths['factuals'], f'{ufid}.csv'), index_col=index_col)
    Ycf = pd.read_csv(os.path.join(paths['counterfactuals'], f'{ufid}_cf.csv'), index_col=index_col)
    df = x.join([Yf, Ycf], how='inner')
    X = df.values[:, :-4]
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    n, n_features = X.shape
    n_train = int(.7 * n)
    n_test = int(.2 * n)
    T = df['z'].values
    Yf = df['y'].values
    Y = np.zeros((n, 2))
    Y[:, 0] = df['y0'].values
    Y[:, 1] = df['y1'].values
    Ycf = Y[np.arange(n), 1-T]

    full_trial(X[:n_train], T[:n_train], Yf[:n_train], Ycf[:n_train],
               X[n_train:-n_test], T[n_train:-n_test], Yf[n_train:-n_test], Ycf[n_train:-n_test],
               X[-n_test:], T[-n_test:], Yf[-n_test:], Ycf[-n_test:])
    print()
    xc_trial(X[:n_train], T[:n_train], Yf[:n_train], Ycf[:n_train],
             X[n_train:-n_test], T[n_train:-n_test], Yf[n_train:-n_test], Ycf[n_train:-n_test],
             X[-n_test:], T[-n_test:], Yf[-n_test:], Ycf[-n_test:])
    print()
    ols_trial(X[:n_train], T[:n_train], Yf[:n_train], Ycf[:n_train],
              X[n_train:-n_test], T[n_train:-n_test], Yf[n_train:-n_test], Ycf[n_train:-n_test],
              X[-n_test:], T[-n_test:], Yf[-n_test:], Ycf[-n_test:])
    print()
    nn_trial(X[:n_train], T[:n_train], Yf[:n_train], Ycf[:n_train],
             X[n_train:-n_test], T[n_train:-n_test], Yf[n_train:-n_test], Ycf[n_train:-n_test],
             X[-n_test:], T[-n_test:], Yf[-n_test:], Ycf[-n_test:])
