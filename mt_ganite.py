from keras.layers import Input, Dense, Lambda, Concatenate, Multiply, dot
from keras.models import Model
from keras import losses
from keras import initializers
from keras.utils import to_categorical
from keras import backend as K
import numpy as np
import pandas as pd
from tqdm import tqdm

from synthetic_data import synthetic_data

params = {
    'h_layers': 2,
    'h_dim': 16,
    'activation': 'relu',
    'n_iters': 2000,
    'gd_train_ratio': 8,
    'mtg_train_ratio': 4,
    'batch_size': 128,
    'optimizer': 'adam',
    'alpha': 1,
    'gamma': 0.05,
}

class MT_GANITE:
    def __init__(self, n_features, n_treatments):
        self.n_features = n_features
        self.n_treatments = n_treatments

        self.h_layers = params['h_layers']
        self.h_dim = params['h_dim']
        self.activation = params['activation']
        self.n_iters = params['n_iters']
        self.gd_train_ratio = params['gd_train_ratio']
        self.mtg_train_ratio = params['mtg_train_ratio']
        self.batch_size = params['batch_size']
        self.optimizer = params['optimizer']
        self.alpha = params['alpha']
        self.gamma = params['gamma']

        # Build components
        self.selectors = self.build_selectors()
        self.generators = [self.build_generator(t) for t in range(self.n_treatments)]
        self.baseline = self.build_baseline()
        self.discriminator = self.build_discriminator()
        self.inferences = self.build_inferences()

        # Define losses
        def d_loss_fn():
            return losses.categorical_crossentropy
        def b_loss_fn():
            def loss(y_true, y_pred):
                d_loss = y_true[:, 0]
                yf = y_true[:, 1]
                T = y_true[:, 2:]
                supervised_loss = K.square(y_true - dot([T, y_pred], axes=-1))
                return supervised_loss - self.alpha * d_loss
            return loss
        def i_loss_fn():
            return losses.mean_squared_error
        def g_loss_fn(yf, ycf):
            supervised_loss = K.square(yf - ycf)
            def loss(y_true, y_pred):
                d_loss = y_true[:, 0]
                ycf_base = y_true[:, 1]
                supervised_loss = K.square(y
                return supervised_loss - self.alpha * d_loss_fn()(y_true, y_pred)
            return loss
        def s_loss_fn(S):
            def loss(y_true, y_pred):
                complexity_loss = self.gamma * K.mean(y_pred, axis=1)
                s = y_true[:, :self.n_features]
                y_generator = y_true[:, self.n_features]
                y_baseline = y_true[:, self.n_features + 1]
                y_factual = y_true[:, self.n_features + 2]
                imitation_reward = -K.sum(K.square(y_generator - y_baseline))
                policy_grad = imitation_reward * K.sum(s*K.log(y_pred+K.epsilon()) + (1-s)*K.log(1-y_pred+K.epsilon()), axis=1)
                return K.mean(-policy_grad) + complexity_loss
            return loss

        # Compile neural networks
        self.discriminator.compile(loss=d_loss_fn(), optimizer=self.optimizer, metrics=['categorical_accuracy'])
        self.baseline.compile(loss=b_loss_fn(), optimizer=self.optimizer)
        for selector, generator, inference in zip(self.selectors, self.generators, self.inferences):
            selector.compile(loss=s_loss_fn(), optimizer=self.optimizer)
            generator.compile(loss=g_loss_fn(), optimizer=self.optimizer)
            inference.compile(loss=i_loss_fn(), optimizer=self.optimizer)

    def train(self, n_iters, X_train, T_train, yf_train, Y_train, X_val=None, T_val=None, yf_val=None, Y_val=None):
        val = all(val_data is not None for val_data in [X_val, T_val, yf_val, Y_val])
        N_train = X_train.shape[0]
        ts = np.argmax(T_train, axis=1)
        idxs = [np.argwhere(ts==t).flatten() for t in range(self.n_treatments)]

        def get_batch(from_idxs=np.arange(N_train), batch_size=self.batch_size):
            idx = np.random.choice(from_idxs, size=batch_size, replace=False)
            X = X_train[idx]
            Z = self.sample_Z(batch_size)
            T = T_train[idx]
            yf = yf_train[idx]
            return X, Z, T, yf

        for it in tqdm(range(n_iters)):
            # Train discriminator against baseline
            X, Z, T, yf = get_batch()
            Ycf_base = self.baseline.predict([X, Z, T, yf])
            Ym_base = T * yf + (1-T) * Ycf_base
            d_loss_base = self.discriminator.train_on_batch([X, Ym_base], T)

            # Train baseline 


            # Train discriminator on both mt and oh generators
            for i in range(self.gd_train_ratio):
                # oh generator
                X, Z, T, yf = get_batch()
                Ycf_oh = self.oh_generator.predict([X, Z, T, yf])
                Ym_oh = (T.T * yf).T + (1-T) * Ycf_oh
                d_loss_ohg, d_acc_ohg = self.discriminator.train_on_batch([X, Ym_oh], T)

                # mt generator
                X, Z, T, yf = get_batch()
                Ycf_mt = self.mt_ganite.predict([X, Z], batch_size=self.batch_size)
                Ym_mt = (T.T * yf).T + (1-T) * Ycf_mt
                d_loss_mtg, d_acc_mtg = self.discriminator.train_on_batch([X, Ym_mt], T)

            ## Train generators
            for i in range(self.mtg_train_ratio):
                for t in range(self.n_treatments):
                    # mt generator
                    idx = np.random.choice(idxs[t], size=self.batch_size, replace=False)
                    X = X_train[idx]
                    Z = self.sample_Z(self.batch_size)
                    T = T_train[idx]
                    yf = yf_train[idx]
                    mtg_loss = self.mt_gan_generators[t].train_on_batch([X, Z, T, yf], T)

                    # selector
                    idx = np.random.choice(idxs[t], size=self.batch_size, replace=False)
                    X = X_train[idx]
                    Z = self.sample_Z(self.batch_size)
                    T = T_train[idx]
                    yf = yf_train[idx]
                    Ycf_oh = self.oh_generator.predict([X, Z, T, yf])
                    mts_loss = self.mt_gan_selectors[t].train_on_batch([X, Z, T, yf], Y_train[idx])

            # oh generator
            X, Z, T, yf = get_batch()
            ohg_loss = self.oh_gan_generator.train_on_batch([X, Z, T, yf], T)

            if it % (n_iters//10) == 0:
                complexities = np.array([np.mean(mt_selector.predict(X_val), axis=1)[:10] for mt_selector in self.mt_selectors])
                print(complexities)
                print(f'Iter: {it:4d}/{n_iters:4d}')
                print(f'd_loss_mtg: {d_loss_mtg:2.4f} d_acc_mtg {d_acc_mtg:2.4f} mtg_loss: {mtg_loss:2.4f} mts_loss: {mts_loss:2.4f}')
                print(f'd_loss_ohg: {d_loss_ohg:2.4f} d_acc_ohg {d_acc_ohg:2.4f} ohg_loss: {ohg_loss:2.4f}')
                if val:
                    Z_val = self.sample_Z(X_val.shape[0])
                    #Ycf_mt_val = np.concatenate([mt_generator.predict([X_val, Z_val]) for mt_generator in self.mt_generators], axis=1)
                    Ycf_mt_val = self.mt_ganite.predict([X_val, Z_val], batch_size=self.batch_size)
                    mse_mtg = np.mean(np.square(Y_val - Ycf_mt_val))
                    Ycf_oh_val = self.oh_generator.predict([X_val, Z_val, T_val, yf_val])
                    mse_ohg = np.mean(np.square(Y_val - Ycf_oh_val))
                    print(f'mse_mtg: {mse_mtg:2.4f} mse_ohg: {mse_ohg:2.4f}')

        return
        # Train inference
        for it in tqdm(range(n_iters)):
            X, Z, T, yf = get_batch()
            Ycf = self.oh_generator.predict([X, Z, T, yf])
            Ym = (T.T * yf).T + (1-T) * Ycf
            i_loss = self.inference.train_on_batch(X, Ym)

            if it % (n_iters//10) == 0:
                print(f'Iter: {it:4d}/{n_iters:4d}')
                print(f'i_loss: {i_loss:2.4f}')

    def predict(self, X):
        # TODO Sample over multiple Z for each x
        Z = self.sample_Z(X.shape[0])
        return self.mt_ganite.predict([X, Z], batch_size=self.batch_size)

    def build_selectors(self):
        X = Input((self.n_features,), name='X')
        hidden = X
        for h_layer in range(self.h_layers):
            hidden = Dense(self.h_dim, activation=self.activation, name=f'h_{h_layer+1}')(hidden)
        Ss = [Dense(self.n_features, activation='sigmoid', name=f'S_{t+1}')(hidden) for t in range(self.n_treatments)]
        return [Model(X, Ss[t], name=f'Selector {t+1}') for t in range(self.n_treatments)]

    def build_generator(self, t):
        x = Input((self.n_features,), name='x')
        s = Input((self.n_features,), name='s')
        Z = Input((self.n_treatments,), name='Z')
        hidden = Concatenate()([x, s, Z])
        for h_layer in range(self.h_layers):
            hidden = Dense(self.h_dim, activation=self.activation, name=f'h_{h_layer+1}')(hidden)
        ycf = Dense(1, activation='sigmoid', name=f'ycf_{t+1}')(hidden)
        return Model([x, s, Z], ycf, name=f'Generator {t+1}')

    def build_baseline(self):
        X = Input((self.n_features,), name='X')
        Z = Input((self.n_treatments,), name='Z')
        T = Input((self.n_treatments,), name='T')
        yf = Input((1,), name='yf')
        hidden = Concatenate()([X, Z, T, yf])
        for h_layer in range(self.h_layers):
            hidden = Dense(self.h_dim, activation=self.activation, name=f'h_{h_layer+1}')(hidden)
        Ycf = Dense(self.n_treatments, activation='sigmoid', name='Ycf')(hidden)
        return Model([X, Z, T, yf], Ycf, name='Baseline')

    def build_discriminator(self):
        X = Input((self.n_features,), name='X')
        Ym = Input((self.n_treatments,), name='Ym')
        hidden = Concatenate()([X, Ym])
        for h_layer in range(self.h_layers):
            hidden = Dense(self.h_dim, activation=self.activation, name=f'h_{h_layer+1}')(hidden)
        T_pred = Dense(self.n_treatments, activation='softmax', name='T_pred')(hidden)
        return Model([X, Ym], T_pred, name='Discriminator')

    def build_inferences(self):
        X = Input((self.n_features,), name='X')
        hidden = X
        for h_layer in range(self.h_layers):
            hidden = Dense(self.h_dim, activation=self.activation, name=f'h_{h_layer+1}')(hidden)
        y_preds = [Dense(1, activation='sigmoid', name=f'y_pred_{t+1}')(hidden) for t in range(self.n_treatments)]
        return [Model(X, y_preds[t], name='Inference {t+1}') for t in range(self.n_treatments)]

    def sample_Z(self, N):
        return np.random.uniform(-1, 1, size=[N, self.n_treatments])

if __name__ == '__main__':
    np.set_printoptions(formatter={'float_kind': lambda x: f'{x:.4f}'})

    synthetic = True
    if synthetic:
        X, t, Y = synthetic_data(n_features=30, n_treatments=2, models=[1, 2])
        N, n_features = X.shape
        n_treatments = Y.shape[1]
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
        t = np.random.randint(n_treatments, size=N)
    T = to_categorical(t, num_classes=n_treatments, dtype='int32')
    yf = np.choose(t, Y.T)

    # Train/test split
    N_train = N
    while N_train > 0.8 * N:
        N_train -= params['batch_size']
    X_train = X[:N_train]
    yf_train = yf[:N_train]
    T_train = T[:N_train]
    Y_train = Y[:N_train]
    X_test = X[N_train:]
    yf_test = yf[N_train:]
    T_test = T[N_train:]
    Y_test = Y[N_train:]

    # MT_GANITE
    mt_ganite = MT_GANITE(n_features, n_treatments)
    mt_ganite.train(2000, X_train, T_train, yf_train, Y_train, X_test, T_test, yf_test, Y_test)
    print(mt_ganite.predict(X_test))
    print(Y[N_train:])
