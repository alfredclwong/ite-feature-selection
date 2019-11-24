import numpy as np
from keras.utils import to_categorical
import pandas as pd

from ganite import GANITE
from derp import Derp
from synthetic_data import get_T, synthetic_data

if __name__ == '__main__':
    np.set_printoptions(formatter={'float_kind': lambda x: f'{x:.4f}'})
    
    synthetic = False
    if synthetic:
        N = 10000
        n_features = 30
        n_treatments = 2
        X, t, Y = synthetic_data(N=N, n_features=n_features, n_treatments=n_treatments)
    else:
        # Load data
        data = pd.read_csv('Twin_Data.csv').values
        N = data.shape[0]
        n_features = 30
        n_treatments = 2

        # Preprocess
        X = data[:, :n_features]
        X = X - X.min(axis=0)
        X = X / X.max(axis=0)
        Y = data[:, n_features:]
        Y[Y>365] = 365
        Y = 1 - Y/365.0
        #t = np.random.randint(n_treatments, size=N)
        t = get_T(X, n_treatments=n_treatments)
        t[np.any(Y, axis=1)] = 1
    Yf = np.choose(t, Y.T)
    T = to_categorical(t, num_classes=n_treatments)
   
    N_train = int(0.8 * N)
    X_train = X[:N_train]
    T_train = T[:N_train]
    Yf_train = Yf[:N_train]
    X_test = X[N_train:]
    T_test = T[N_train:]
    Yf_test = Yf[N_train:]
    Y_test = Y[N_train:]

    params = {
        'n_features':   n_features,
        'n_treatments': n_treatments,
    }
    hyperparams = {
        'h_layers':     2,
        'h_dim':        n_features,
        'alpha':        1,
        'beta':         0.5,
        'batch_size':   128,
        'k1':           16,
        'k2':           1,
    }
    ganite = GANITE(params, hyperparams)
    ganite.train([1000, 3000], X_train, T_train, Yf_train, X_test, T_test, Yf_test, Y_test)

    derp = Derp(n_features, n_treatments)
    derp.train(5000, X_train, T_train, Yf_train, X_test, Y_test)

    print('Y_pred (ganite)')
    print(ganite.predict(X_test))
    print('Y_pred (derp)')
    print(derp.predict(X_test))
    print('Y_test')
    print(Y_test)
