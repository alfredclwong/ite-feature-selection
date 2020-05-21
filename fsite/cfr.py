import numpy as np
from keras.layers import Dense, Input, Lambda, BatchNormalization
from keras.models import Model
from keras.regularizers import l2
import keras.backend as K
from tqdm import tqdm

default_hyperparams = {
    'alpha':            .1,   # tradeoff between factual loss and IPM (more alpha = more IPM loss)
    'gamma':            .5,   # sigma^2/2 for Gaussian RBF kernel
    'rep_h_layers':     2,    #
    'rep_h_dim':        100,  #
    'pred_h_layers':    2,    #
    'pred_h_dim':       100,  #
    'batch_size':       128,  #
}


class CfrNet():
    def __init__(self, n_features, n_treatments, hyperparams=default_hyperparams):
        assert n_treatments == 2
        self.n_features = n_features
        self.n_treatments = n_treatments

        self.preds = []
        for t in range(self.n_treatments):
            self.preds.append(self.build_pred(hyperparams['pred_h_layers'], hyperparams['pred_h_dim']))
            self.preds[t].compile('adam', 'mse')
            self.preds[t].trainable = False

        alpha = hyperparams['alpha']
        gamma = hyperparams['gamma']
        batch_size = hyperparams['batch_size']

        def k(x, y):
            return K.exp(-gamma * (K.square(x) + K.transpose(K.square(y)) - 2*x@K.transpose(y)))

        def rep_loss_fn(TYwb, R_pred):
            # Calculate h_loss
            Tb, Yb, wb = [TYwb[:, i] for i in range(3)]
            Y0_pred = self.preds[0](R_pred)
            Y1_pred = self.preds[1](R_pred)
            Y_pred = Y0_pred[1] * Tb + Y1_pred[0] * (1 - Tb)
            h_loss = K.mean(K.square(Yb - Y_pred) * wb)

            if alpha == 0:
                return h_loss

            # Estimate MMD squared
            n0 = batch_size - K.sum(Tb)
            n1 = K.sum(Tb)
            y0 = K.reshape(R_pred[Tb == 0], (-1, 1))
            y1 = K.reshape(R_pred[Tb == 1], (-1, 1))
            k00 = k(y0, y0)
            k11 = k(y1, y1)
            k01 = k(y0, y1)
            k10 = k(y1, y0)
            ndiag0 = ~K.eye(n0, dtype=bool)
            ndiag1 = ~K.eye(n1, dtype=bool)
            mmd2 = K.mean(k00[ndiag0]) + K.mean(k11[ndiag1]) - K.mean(k01) - K.mean(k10)
            return alpha * mmd2 + h_loss

        self.rep = self.build_rep(hyperparams['rep_h_layers'], hyperparams['rep_h_dim'])
        self.rep.compile('adam', rep_loss_fn)

    def train(self, X, T, Y, n_iters, val_data=None, test_data=None,
              batch_size=128, verbose=False, save_history=False):
        n = X.shape[0]
        # p1 = np.mean(T)
        # weights = (T/p1 + (1-T)/(1-p1)) / 2
        weights = np.ones(n) / 2

        if verbose:
            v_iters = n_iters // 10
        if save_history:
            history = {
                'h_loss':   np.zeros((n_iters, self.n_treatments)),
                'rep_loss': np.zeros(n_iters),
            }
        for it in tqdm(range(n_iters)):
            idx = np.random.choice(n, size=batch_size)
            Xb, Tb, Yb, wb = map(lambda x: x[idx], [X, T, Y, weights])
            Rb = self.rep.predict(Xb)

            h_loss = np.zeros(self.n_treatments)
            for t in range(self.n_treatments):
                h_loss[t] = self.preds[t].train_on_batch(Rb[Tb == t], Yb[Tb == t])
            TYwb = np.vstack([Tb, Yb, wb]).T
            rep_loss = self.rep.train_on_batch(Xb, TYwb)
            if save_history:
                history['h_loss'][it] = h_loss
                history['rep_loss'][it] = rep_loss

            if verbose and (it+1) % v_iters == 0:
                print(f'h_loss\t\t{h_loss}\nrep_loss\t{rep_loss}')

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

    def build_rep(self, h_layers, h_dim):
        X = Input((self.n_features,))
        H = X
        for _ in range(h_layers):
            H = Dense(h_dim, activation='elu', kernel_regularizer=l2())(H)
            # H = BatchNormalization()(H)
        R = Lambda(lambda h: K.l2_normalize(100*h, axis=1))(H)
        return Model(X, R)

    def build_pred(self, h_layers, h_dim):
        R = Input((h_dim,))
        H = R
        for _ in range(h_layers):
            H = Dense(h_dim, activation='elu', kernel_regularizer=l2())(H)
        Y = Dense(1)(H)
        return Model(R, Y)
