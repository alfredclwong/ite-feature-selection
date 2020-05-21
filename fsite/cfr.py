import numpy as np
from keras.layers import Dense, Input, Lambda
from keras.models import Model
from keras.regularizers import l2
from sklearn.metrics.pairwise import rbf_kernel
import keras.backend as K
from tqdm import tqdm

from data.synthetic_data import get_ihdp_XT, get_ihdp_Yb

default_hyperparams = {
    'alpha':            1,
    'gamma':            .5,
    'rep_h_layers':     3,
    'rep_h_dim':        50,
    'pred_h_layers':    3,
    'pred_h_dim':       50,
    'batch_size':       128,
}


class CfrNet():
    def __init__(self, n_features, n_treatments, hyperparams=default_hyperparams):
        assert n_treatments == 2
        self.n_features = n_features
        self.n_treatments = n_treatments
        
        alpha = hyperparams['alpha']
        gamma = hyperparams['gamma']
        batch_size = hyperparams['batch_size']

        def k(x, y):
            return K.exp(-gamma * (K.square(x) + K.transpose(K.square(y)) - 2*x@K.transpose(y)))

        def rep_loss_fn(y_true, y_pred):
            T = y_true[:batch_size]

            n0 = batch_size - K.sum(T)
            n1 = K.sum(T)
            y0 = K.reshape(y_pred[T == 0], (-1, 1))
            y1 = K.reshape(y_pred[T == 1], (-1, 1))
            k00 = k(y0, y0)
            k11 = k(y1, y1)
            k01 = k(y0, y1)
            k10 = k(y1, y0)
            mmd2 = K.sum(k00)/n0/(n0-1) + K.sum(k11)/n1/(n1-1) - K.mean(k01) - K.mean(k10)

            return alpha * mmd2

        self.rep = self.build_rep(hyperparams['rep_h_layers'], hyperparams['rep_h_dim'])
        self.rep.compile('adam', rep_loss_fn)

        self.preds = []
        for t in range(self.n_treatments):
            self.preds.append(self.build_pred(hyperparams['pred_h_layers'], hyperparams['pred_h_dim']))
            self.preds[-1].compile('adam', 'mse')

    def train(self, X, T, Y, n_iters, batch_size=128):
        n = X.shape[0]
        # u = np.mean(T)
        # w = (T/u + (1-T)/(1-u)) / 2

        history = {
            'rep_loss': np.zeros(n_iters),
        }
        for it in tqdm(range(n_iters)):
            idx = np.random.choice(n, size=batch_size)
            rep_loss = self.rep.train_on_batch(X[idx], T[idx])
            history['rep_loss'] = rep_loss

        return history

    def build_rep(self, h_layers, h_dim):
        X = Input((self.n_features,))
        H = X
        for _ in range(h_layers):
            H = Dense(h_dim, activation='relu', kernel_regularizer=l2())(H)
        R = Lambda(lambda h: K.l2_normalize(1000*h, axis=1))(H)
        return Model(X, R)

    def build_pred(self, h_layers, h_dim):
        X = Input((h_layers,))
        H = X
        for _ in range(h_layers):
            H = Dense(h_dim, activation='relu', kernel_regularizer=l2())(H)
        return Model(X, H)
