import numpy as np
from keras.layers import Dense, Lambda, BatchNormalization, Dropout, Input, Add, Multiply
from keras.models import Sequential, Model
from keras.regularizers import l2
from keras.optimizers import Adam
import keras.backend as K
from tqdm import tqdm
import tensorflow as tf
from utils.metrics import PEHE
from utils.utils import make_Y
# from imblearn.over_sampling import SMOTE

default_hyperparams = {
    'alpha':            0,   # tradeoff between factual loss and IPM (more alpha = more IPM loss)
    'sigma':            0.5,    # for Gaussian RBF kernel
    'rep_h_layers':     1,     #
    'rep_h_dim':        50,    #
    'rep_dim':          50,   #
    'pred_h_layers':    1,     #
    'pred_h_dim':       50,   #
    'batch_size':       100,   #
    'learning_rate':    1e-3,  #
    'reg_penalty':      1e-4,  # TODO
}


class CfrNet():
    def __init__(self, n_features, n_treatments, hyperparams=default_hyperparams):
        assert n_treatments == 2
        self.n_features = n_features
        self.n_treatments = n_treatments
        self.alpha = hyperparams['alpha']
        self.sigma = hyperparams['sigma']
        self.batch_size = hyperparams['batch_size']

        # Build nets
        self.rep = self.build_rep(hyperparams['rep_h_layers'],
                                  hyperparams['rep_h_dim'],
                                  hyperparams['rep_dim'])
        self.preds = []
        for t in range(self.n_treatments):
            self.preds.append(self.build_pred(hyperparams['pred_h_layers'],
                                              hyperparams['pred_h_dim'],
                                              t))
            self.preds[t].compile(Adam(1e-3), 'mse')
            self.preds[t].trainable = False

        # Build graph
        X = Input((self.n_features,))
        R = self.rep(X)
        Y0 = self.preds[0](R)
        Y1 = self.preds[1](R)
        self.cfr = Model(X, [R, Y0, Y1])

    def factual_loss(self, TYt, Yt_pred):
        T = TYt[:, 0]
        Yt = TYt[:, 1:]
        return K.square(Yt - Yt_pred) / K.sum(T)

    def mmd2(self, R, T):
        def k(x, y):
            return K.exp(-(K.sum(K.square(x), axis=-1, keepdims=True)
                         + K.transpose(K.sum(K.square(y), axis=-1, keepdims=True))
                         - 2 * x @ K.transpose(y)) / K.square(self.sigma))
        n0 = K.sum(1-T)
        n1 = K.sum(T)
        kernel = k(R, R)
        pos = K.concatenate([
                K.concatenate([K.ones((n0, n0), dtype='int32'), K.zeros((n0, n1), dtype='int32')], axis=1),
                K.concatenate([K.zeros((n1, n0), dtype='int32'), K.ones((n1, n1), dtype='int32')], axis=1)],
                axis=0) - K.eye(n0 + n1, dtype='int32')
        neg = 1 - pos - K.eye(n0 + n1, dtype='int32')
        mmd2 = K.mean(K.gather(kernel, pos)) - K.mean(K.gather(kernel, neg))
        # mmd2 = K.sum(kernel * pos) / (n0*(n0-1) + n1*(n1-1)) - K.sum(kernel * neg) / n0 / n1
        return mmd2

    def train(self, X, T, Yf, n_iters, Ycf=None, val_data=None, test_data=None, verbose=False, save_history=False):
        n = X.shape[0]
        if Ycf is not None:
            Y = make_Y(T, Yf, Ycf)
        val = val_data is not None
        if val:
            assert len(val_data) == 4
            X_val, T_val, Yf_val, Ycf_val = val_data
            Y_val = make_Y(T_val, Yf_val, Ycf_val)
        test = test_data is not None
        if test:
            assert len(test_data) == 4
            X_test, T_test, Yf_test, Ycf_test = test_data
            Y_test = make_Y(T_test, Yf_test, Ycf_test)

        # Set up class weights
        p1 = np.mean(T)
        weights = np.array([1/(1-p1), 1/p1]) / 2
        # weights = np.ones(self.n_treatments) / 2
        self.cfr.compile(
            Adam(1e-4),
            loss={
                'R': lambda Tb, Rb_pred: self.mmd2(Rb_pred, Tb),
                'Y0': lambda TY0b, Y0b_pred: self.factual_loss(TY0b, Y0b_pred),
                'Y1': lambda TY1b, Y1b_pred: self.factual_loss(TY1b, Y1b_pred),
            },
            loss_weights={
                'R': self.alpha,
                'Y0': weights[0],
                'Y1': weights[1],
            })

        if verbose:
            v_iters = n_iters // 10
        if save_history:
            n_savepoints = 100
            history = {m: np.zeros(n_savepoints) for m in 'obj h0 h1 imb PEHE'.split()}
            if val:
                history['PEHE_val'] = np.zeros(n_savepoints)
            if test:
                history['PEHE_test'] = np.zeros(n_savepoints)
            h_iters = n_iters // n_savepoints

        # for it in tqdm(range(n_iters)):
        for it in range(n_iters):
            idx = np.random.choice(n, size=self.batch_size)
            # idx = np.zeros(n, dtype=bool)
            # for t in range(self.n_treatments):
            #     idx[np.random.choice(np.where(T == t)[0], size=self.batch_size//2)] = 1
            Xb, Tb, Yb = map(lambda x: x[idx], [X, T, Yf])

            ## THE ACTUAL TRAINING HAPPENS HERE ##
            TY0b = np.vstack([Tb, (1-Tb)*Yb]).T
            TY1b = np.vstack([Tb, Tb*Yb]).T
            cfr_loss = self.cfr.train_on_batch(Xb, [Tb, TY0b, TY1b])

            Rb = self.project(Xb)
            h0_loss = self.preds[0].train_on_batch(Rb[Tb == 0], Yb[Tb == 0])
            h1_loss = self.preds[1].train_on_batch(Rb[Tb == 1], Yb[Tb == 1])
            ## THE ACTUAL TRAINING HAPPENED HERE ##

            do_history = save_history and (it+1) % h_iters
            do_verbose = (verbose and (it+1) % v_iters == 0) or it == 0 or (it+1) == n_iters
            if not do_history and not do_verbose:
                continue

            obj, imb, _, _ = cfr_loss
            if self.alpha != 0:
                imb /= self.alpha
            Y_pred = self.predict(X)
            pehe = PEHE(Y, Y_pred)
            if val:
                Y_val_pred = self.predict(X_val)
                pehe_val = PEHE(Y_val, Y_val_pred)
            if test:
                Y_test_pred = self.predict(X_test)
                pehe_test = PEHE(Y_test, Y_test_pred)

            if do_history:
                idx = it//h_iters
                history['obj'][idx] = obj
                history['h0'][idx] = h0_loss
                history['h1'][idx] = h1_loss
                history['imb'][idx] = imb
                history['PEHE'][idx] = pehe
                if val:
                    history['PEHE_val'][idx] = pehe_val
                if test:
                    history['PEHE_test'][idx] = pehe_test

            if do_verbose:
                table_row = '\t'.join(f'{a} {b:.4f}' for a, b in zip('obj h0 h1 imb PEHE'.split(),
                                                                     [obj, h0_loss, h1_loss, imb, pehe]))
                print(f'{it+1}\t{table_row}', end='')
                if val:
                    print(f'\tval {pehe_val:.4f}', end='')
                if test:
                    print(f'\ttest {pehe_test:.4f}', end='')
                print()

        if save_history:
            return history

    def predict(self, X, T=None):
        n = X.shape[0]
        R = self.rep.predict(X)
        Y = np.zeros((n, self.n_treatments))
        for t in range(self.n_treatments):
            Y[:, t] = self.preds[t].predict(R).flatten()
        return Y if T is None else Y[np.arange(n), T]

    def project(self, X):
        R = self.rep.predict(X)
        return R

    def build_rep(self, h_layers, h_dim, rep_dim):
        rep = Sequential(name='R')
        for _ in range(h_layers):
            rep.add(Dense(h_dim, activation='elu'))
            # rep.add(BatchNormalization())
            # rep.add(Dropout(.1))
        rep.add(Dense(rep_dim, activation='elu'))
        rep.add(Lambda(lambda x: K.l2_normalize(1000*x, axis=1)))
        return rep

    def build_pred(self, h_layers, h_dim, t):
        pred = Sequential(name=f'Y{t}')
        for _ in range(h_layers):
            pred.add(Dense(h_dim, activation='elu', kernel_regularizer=l2(1e-4)))
        pred.add(Dense(1))
        return pred
