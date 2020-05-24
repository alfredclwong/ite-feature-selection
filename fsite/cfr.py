import numpy as np
from keras.layers import Dense, Lambda, Input
from keras.models import Sequential, Model
from keras.regularizers import l2
from keras.optimizers import Adam
import keras.backend as K
from utils.metrics import PEHE
from utils.utils import make_Y
from utils.loss import mmd2, factual
# from imblearn.over_sampling import SMOTE

default_batch_size = 100
default_hyperparams = {
    'alpha':            1e-1,      # tradeoff between factual loss and IPM (more alpha = more IPM loss)
    'sigma':            1.0,       # for Gaussian RBF kernel
    'rep_h_layers':     3,         #
    'rep_h_dim':        200,       #
    'rep_dim':          100,       #
    'pred_h_layers':    3,         #
    'pred_h_dim':       100,       #
    'pred_lr':          1e-3,      #
    'rep_lr':           1e-4,      #
    'pred_reg':         l2(1e-4),  #
    'activation':       'elu'      #
}


class CfrNet():
    def __init__(self, n_features, n_treatments, hyperparams=default_hyperparams):
        assert n_treatments == 2
        self.n_features = n_features
        self.n_treatments = n_treatments
        self.alpha = hyperparams['alpha']
        self.sigma = hyperparams['sigma']
        self.rep_lr = hyperparams['rep_lr']

        # Build nets
        self.rep = self.build_rep(hyperparams['rep_h_layers'],
                                  hyperparams['rep_h_dim'],
                                  hyperparams['rep_dim'],
                                  hyperparams['activation'])
        self.preds = []
        for t in range(self.n_treatments):
            self.preds.append(self.build_pred(hyperparams['pred_h_layers'],
                                              hyperparams['pred_h_dim'],
                                              hyperparams['activation'],
                                              hyperparams['pred_reg'],
                                              t))
            self.preds[t].compile(Adam(hyperparams['pred_lr']), 'mse')
            self.preds[t].trainable = False  # so that we can use preds in the cfr model

        # Build graph
        X = Input((self.n_features,))
        R = self.rep(X)
        Y0 = self.preds[0](R)
        Y1 = self.preds[1](R)
        self.cfr = Model(X, [R, Y0, Y1])  # compile later using weights from training set

    def train(self, X, T, Yf, n_iters, Ycf=None, batch_size=default_batch_size,
              val_data=None, test_data=None, verbose=False, save_history=False):
        # Check data and make adjustments if necessary
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

        # Compile cfr model using class weights
        p1 = np.mean(T)
        weights = np.array([1/(1-p1), 1/p1]) / 2
        # weights = np.ones(self.n_treatments) / 2
        self.cfr.compile(
            Adam(self.rep_lr),
            loss={
                'R': lambda Tb, Rb_pred: mmd2(Rb_pred, Tb, sigma=self.sigma),
                'Y0': lambda TY0b, Y0b_pred: factual(TY0b, Y0b_pred, 0),
                'Y1': lambda TY1b, Y1b_pred: factual(TY1b, Y1b_pred, 1),
            },
            loss_weights={
                'R': self.alpha,
                'Y0': weights[0],
                'Y1': weights[1],
            })

        # Prep for training loop
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

        # TRAINING LOOP
        for it in range(n_iters):
            # Random (imbalanced) batch
            idx = np.random.choice(n, size=batch_size)
            Xb, Tb, Yb = map(lambda x: x[idx], [X, T, Yf])

            # THE ACTUAL TRAINING HAPPENS HERE #
            TYb = np.vstack([Tb, Yb]).T
            cfr_loss = self.cfr.train_on_batch(Xb, [Tb, TYb, TYb])

            Rb = self.project(Xb)
            h0_loss = self.preds[0].train_on_batch(Rb[Tb == 0], Yb[Tb == 0])
            h1_loss = self.preds[1].train_on_batch(Rb[Tb == 1], Yb[Tb == 1])
            # THE ACTUAL TRAINING HAPPENED HERE #

            # Report training progress
            do_history = save_history and (it+1) % h_iters
            do_verbose = (verbose and (it+1) % v_iters == 0) or it == 0 or (it+1) == n_iters
            if not do_history and not do_verbose:
                continue

            obj, imb, h0_cfr, h1_cfr = cfr_loss
            # obj = imb = cfr_loss
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
                # print(f'\t{h0_cfr:.4f}\t{h1_cfr:.4f}')

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

    def build_rep(self, h_layers, h_dim, rep_dim, activation):
        rep = Sequential(name='R')
        for _ in range(h_layers):
            rep.add(Dense(h_dim, activation=activation))
        rep.add(Dense(rep_dim, activation=activation))
        rep.add(Lambda(lambda x: K.l2_normalize(1000*x, axis=1)))
        return rep

    def build_pred(self, h_layers, h_dim, activation, reg, t):
        pred = Sequential(name=f'Y{t}')
        for _ in range(h_layers):
            pred.add(Dense(h_dim, activation=activation, kernel_regularizer=reg))
        pred.add(Dense(1))
        return pred
