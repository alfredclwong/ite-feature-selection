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
sys.stdout = Logger()
n_treatments = 2


def full_trial(X_train, T_train, Yf_train, Ycf_train,
               X_val, T_val, Yf_val, Ycf_val,
               X_test, T_test, Yf_test, Ycf_test):
    print('invase')
    # Disentanglement stage
    invases = [
        Invase(n_features, n_treatments, 0.1),
        Invase(n_features, 0, 1.5),
        Invase(n_features, 0, 1.5),
    ]
    selection = []
    for invase, target in zip(invases, [T_train, Yf_train[:, None], Yf_train[:, None]]):
        invase.train(X_train, target, 2000, silent=True)
        S_pred = invase.predict_features(X_train, .2).mean(axis=0) > .2
        selection.append(np.where(S_pred)[0].tolist())
    selection = [
        list(set(selection[0]) - set(selection[1]) - set(selection[2])),
        list(set(selection[0])),
        list(set(selection[1] + selection[2]) - set(selection[0]))
    ]
    selection[1] = list(set(selection[1]) - set(selection[0]))
    print(selection)
    ournet = OurNet([len(a) for a in selection])
    Xs_sorted = [np.hstack([X[:, sel] for sel in selection]) for X in [X_train, X_val, X_test]]
    ournet.train(Xs_sorted[0], T_train, Yf_train, 2000, Ycf=Ycf_train,
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
    ournet.train(Xs_sorted[0], T_train, Yf_train, 2000, Ycf=Ycf_train,
                 val_data=(Xs_sorted[1], T_val, Yf_val, Ycf_val),
                 test_data=(Xs_sorted[2], T_test, Yf_test, Ycf_test), verbose=True)


def xc_trial(X_train, T_train, Yf_train, Ycf_train,
             X_val, T_val, Yf_val, Ycf_val,
             X_test, T_test, Yf_test, Ycf_test):
    print('all xc')
    ournet = OurNet([0, X_train.shape[1], 0])
    ournet.train(X_train, T_train, Yf_train, 2000, Ycf=Ycf_train,
                 val_data=(X_val, T_val, Yf_val, Ycf_val),
                 test_data=(X_test, T_test, Yf_test, Ycf_test), verbose=True)


def nn_trial(X, T, Y):
    pass


def ols_trial(X, T, Y):
    pass


'''
1. Synthetic data
Sample partitioned X and use random causal models for T and Y. Report disentanglement
accuracy, PEHE, and results from other methods (all XC, neural net, linear regression)
'''
# n_trials = 10
# n = 10000
# n_train = int(.6 * n)
# n_test = int(.2 * n)
# n_features = 10
# for i in range(n_trials):
#     print(f'{i+1}/1000')
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
#                  X[n_train:-n_test], T[n_train:-n_test], Yf[n_train:-n_test], Ycf[n_train:-n_test],
#                  X[-n_test:], T[-n_test:], Yf[-n_test:], Ycf[-n_test:])
#     print()
#     oracle_trial(X[:n_train], T[:n_train], Yf[:n_train], Ycf[:n_train],
#                  X[n_train:-n_test], T[n_train:-n_test], Yf[n_train:-n_test], Ycf[n_train:-n_test],
#                  X[-n_test:], T[-n_test:], Yf[-n_test:], Ycf[-n_test:], markov)
#     print()
#     xc_trial(X[:n_train], T[:n_train], Yf[:n_train], Ycf[:n_train],
#              X[n_train:-n_test], T[n_train:-n_test], Yf[n_train:-n_test], Ycf[n_train:-n_test],
#              X[-n_test:], T[-n_test:], Yf[-n_test:], Ycf[-n_test:])
#     print()

'''
2. IHDP
Report PEHE and results from other methods
'''
for i, row in enumerate(get_ihdp_npci()):
    print(f'{i+1}/1000')
    X_train, T_train, Yf_train, Ycf_train = row['train']
    X_val, T_val, Yf_val, Ycf_val = row['val']
    X_test, T_test, Yf_test, Ycf_test = row['test']
    n, n_features = X_train.shape

    full_trial(X_train, T_train, Yf_train, Ycf_train,
               X_val, T_val, Yf_val, Ycf_val,
               X_test, T_test, Yf_test, Ycf_test)
    print()
    xc_trial(X_train, T_train, Yf_train, Ycf_train,
             X_val, T_val, Yf_val, Ycf_val,
             X_test, T_test, Yf_test, Ycf_test)
    print()

'''
3. IBM
Report PEHE and results from other methods
'''
