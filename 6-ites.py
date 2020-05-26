import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from scipy.special import expit
import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf

from fsite.invase import Invase
from data.synthetic_data import get_ihdp_XT, get_ihdp_Yb, get_YS, get_ihdp_npci
from utils.utils import default_env, XTY_split, high_dim_vis, make_Y
from utils.loss import mmd2
from utils.metrics import PEHE
from fsite.cfr import CfrNet

default_env(gpu=True)
for i, row in enumerate(get_ihdp_npci(small=True)):
    X_train, T_train, Yf_train, Ycf_train = row['train']
    X_val, T_val, Yf_val, Ycf_val = row['val']
    X_test, T_test, Yf_test, Ycf_test = row['test']
    n_features = X_train.shape[1]
    n_treatments = T_train.max() + 1

    print(mmd2(tf.cast(X_train, 'float'), T_train, tensor=False))
    cfrnet = CfrNet(n_features, n_treatments)
    history = cfrnet.train(X_train, T_train, Yf_train, 2000, Ycf=Ycf_train,
                           val_data=(X_val, T_val, Yf_val, Ycf_val),
                           test_data=(X_test, T_test, Yf_test, Ycf_test),
                           verbose=True, save_history=True)
    R_train = cfrnet.project(X_train)
    # print(R_train)
    # print(T_train)
    print(mmd2(R_train, T_train, tensor=False))
    np.testing.assert_almost_equal(np.sum(np.square(R_train), axis=-1), 1, decimal=4)

    Y_train, Y_test = map(lambda a: make_Y(a[0], a[1], a[2]),
                          [[T_train, Yf_train, Ycf_train], [T_test, Yf_test, Ycf_test]])
    Y_train_pred, Y_test_pred = map(cfrnet.predict, [X_train, X_test])
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

# X, T = get_ihdp_XT()
# n, n_features = X.shape
# T = np.random.binomial(1, expit(X[:, 0] * X[:, 1] - X[:, 2] - 2), n)
# Y, beta = get_ihdp_Yb(X, T, 'B1')
# n_treatments = T.max() + 1