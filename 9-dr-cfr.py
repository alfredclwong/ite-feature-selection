from fsite.drcfr import DrCfr
from utils.utils import default_env, make_Y, pad_lr
import numpy as np
from data.synthetic_data import get_ihdp_npci
import matplotlib.pyplot as plt

default_env(gpu=True)

# n = 10000
# n_features = [1, 1, 1]
# n_treatments = 2
# nt, nc, ny = n_features
# X = np.random.standard_normal(size=(n, sum(n_features)))
# T = np.random.choice(n_treatments, size=n)
# Y = np.zeros((n, n_treatments))
# Yf = Y[np.arange(n), T]

# drcfr = DrCfr([nt, nc, ny])
# drcfr.train(X, T, Yf, 1000, verbose=True)

for i, row in enumerate(get_ihdp_npci(small=True)):
    X_train, T_train, Yf_train, Ycf_train = row['train']
    X_val, T_val, Yf_val, Ycf_val = row['val']
    X_test, T_test, Yf_test, Ycf_test = row['test']
    n_features = [1, X_train.shape[1], 1]
    n_treatments = T_train.max() + 1
    X_train, X_val, X_test = map(pad_lr, [X_train, X_val, X_test])

    drcfr = DrCfr(n_features)
    history = drcfr.train(X_train, T_train, Yf_train, 2000, Ycf=Ycf_train,
                          val_data=(X_val, T_val, Yf_val, Ycf_val),
                          test_data=(X_test, T_test, Yf_test, Ycf_test),
                          verbose=True, save_history=True)

    Y_train, Y_test = map(lambda a: make_Y(a[0], a[1], a[2]),
                          [[T_train, Yf_train, Ycf_train], [T_test, Yf_test, Ycf_test]])
    Y_train_pred, Y_test_pred = map(drcfr.predict, [X_train, X_test])
    # print(f'PEHE {PEHE(Y_train, Y_train_pred)}')
    # print(f'PEHE_test {PEHE(Y_test, Y_test_pred)}')
    # print(Y_train_pred[:10])
    # print(Y_train[:10])
    print(np.mean(np.square(Y_train_pred[:, 0] - Y_train[:, 0])))
    print(np.mean(np.square(Y_train_pred[:, 1] - Y_train[:, 1])))

    metrics = 'obj h0 h1 imb val'.split()
    metrics = 'h0 h1'.split()
    for m in metrics:
        plt.plot(history[m])
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.legend(metrics)
    plt.show()
    break
