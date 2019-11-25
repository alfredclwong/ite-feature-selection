from keras.layers import Input, Dense, Lambda, Concatenate, Multiply, dot
from keras.models import Model
from keras import losses
from keras.utils import to_categorical
from keras import backend as K
import numpy as np
import pandas as pd
from tqdm import tqdm

params = {
    'h_layers': 2,
    'h_dim': 16,
    'activation': 'relu',
    'n_iters': 2000,
    'gd_train_ratio': 5,
    'batch_size': 128,
    'optimizer': 'adam',
    'alpha': 2,
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
        self.batch_size = params['batch_size']
        self.optimizer = params['optimizer']
        self.alpha = params['alpha']
        self.gamma = 0.001

        # Build components
        self.mt_selectors = self.build_selectors()
        self.sampler = Lambda(lambda s: K.random_binomial((self.batch_size, self.n_features), s), name='sampler')
        self.mt_generators = self.build_multi_task_generators()
        self.oh_generator = self.build_one_hot_generator()
        def mix(inputs):
            [yf, T, Ycf] = inputs
            return T * yf + (1-T) * Ycf
        self.mixer = Lambda(mix)
        self.discriminator = self.build_discriminator()
        self.inference = self.build_inference()

        ## Define structure
        # Selector
        X_full = Input((self.n_features,), name='X_full')
        Ss = [mt_selector(X_full) for mt_selector in self.mt_selectors]
        # Generators
        Z = Input((self.n_treatments,), name='Z')
        # Multi-task Generator
        masks = [self.sampler(S) for S in Ss] 
        Xs = [Multiply(name=f'X_{t+1}')([X_full, masks[t]]) for t in range(self.n_treatments)]
        ycfs_mt = [self.mt_generators[t]([Xs[t], Z]) for t in range(self.n_treatments)]
        #ycfs_mt = [self.mt_generators[t]([X, Z]) for t in range(self.n_treatments)] # tmp: skip selector
        Ycf_mt = Concatenate(name='Ycf_mt')(ycfs_mt)
        # One-hot Generator
        T = Input((self.n_treatments,), name='T')
        yf = Input((1,), name='yf')
        Ycf_oh = self.oh_generator([X_full, Z, T, yf])
        # Discriminator
        Ym_mt = self.mixer([yf, T, Ycf_mt])
        T_mt = self.discriminator([X_full, Ym_mt])
        Ym_oh = self.mixer([yf, T, Ycf_oh])
        T_oh = self.discriminator([X_full, Ym_oh])
        # Inference
        Y_oh = self.inference(X_full)

        ## Define models
        # For training
        self.mt_gan_selectors = [Model([X_full, Z, T, yf], T_mt, name=f'MT_GAN_SELECTOR_{t+1}') for t in range(self.n_treatments)]
        self.mt_gan_generators = [Model([X_full, Z, T, yf], T_mt, name=f'MT_GAN_GENERATOR_{t+1}') for t in range(self.n_treatments)]
        self.oh_gan_generator = Model([X_full, Z, T, yf], T_oh, name='OH_GAN_GENERATOR')
        # For predicting
        self.mt_ganite = Model([X_full, Z], Ycf_mt, name='MT_GANITE')
        self.oh_ganite = Model(X_full, Y_oh, name='OH_GANITE')
        # Print summaries
        self.mt_ganite.summary()
        self.oh_gan_generator.summary()
        self.oh_ganite.summary()

        # Define losses
        def d_loss_fn():
            return losses.categorical_crossentropy
        def ohg_loss_fn(yf, T, Ycf):
            supervised_loss = K.square(yf - dot([T, Ycf], axes=-1))
            def loss(y_true, y_pred):
                return supervised_loss - self.alpha * d_loss_fn()(y_true, y_pred)
            return loss
        def i_loss_fn(T):
            def loss(y_true, y_pred):
                supervised_loss = K.square(dot([T, y_true - y_pred], axes=-1))
                cf_loss = K.mean(K.square(y_pred - y_true), axis=-1)
                return supervised_loss + cf_loss
            return loss
        def mtg_loss_fn(yf, ycf):
            supervised_loss = K.square(yf - ycf)
            def loss(y_true, y_pred):
                return supervised_loss - self.alpha * d_loss_fn()(y_true, y_pred)
            return loss
        def s_loss_fn(mask, T_mt, T_oh):
            complexity_loss = K.sum(mask)
            #imitation_loss = losses.categorical_crossentropy(T_mt, T_oh)
            def loss(y_true, y_pred):
                return self.gamma * complexity_loss - d_loss_fn()(y_true, y_pred)
            return loss

        # Compile neural networks
        self.discriminator.compile(loss=d_loss_fn(), optimizer=self.optimizer, metrics=['categorical_accuracy'])
        self.discriminator.trainable = False
        for mt_selector, mt_generator in zip(self.mt_selectors, self.mt_generators):
            mt_selector.trainable = False
            mt_generator.trainable = False
        for t in range(self.n_treatments):
            self.mt_selectors[t].trainable = True
            #self.mt_gan_selectors[t].compile(loss=s_loss_fn(masks[t], T_mt, T_oh), optimizer=self.optimizer)
            self.mt_gan_selectors[t].compile(loss=s_loss_fn(Ss[t], T_mt, T_oh), optimizer=self.optimizer)
            self.mt_selectors[t].trainable = False
            self.mt_generators[t].trainable = True
            self.mt_gan_generators[t].compile(loss=mtg_loss_fn(yf, ycfs_mt[t]), optimizer=self.optimizer)
            self.mt_generators[t].trainable = False
        self.oh_gan_generator.compile(loss=ohg_loss_fn(yf, T, Ycf_oh), optimizer=self.optimizer)
        self.inference.compile(loss=losses.mean_squared_error, optimizer=self.optimizer)  # TODO use i_loss_fn

    def train(self, n_iters, X_train, T_train, yf_train, X_val=None, T_val=None, yf_val=None, Y_val=None):
        def get_batch(batch_size=self.batch_size):
            N_train = X_train.shape[0]
            idx = np.random.randint(N_train, size=batch_size)
            X = X_train[idx]
            Z = self.sample_Z(batch_size)
            T = T_train[idx]
            yf = yf_train[idx]
            return X, Z, T, yf

        ts = np.argmax(T_train, axis=1)
        idxs = [np.argwhere(ts==t).flatten() for t in range(self.n_treatments)]
        for it in tqdm(range(n_iters)):
            # Train discriminator on both mt and oh generators
            for i in range(self.gd_train_ratio):
                # mt generator
                X, Z, T, yf = get_batch()
                Ycf_mt = np.concatenate([self.mt_generators[t].predict([X, Z]) for t in range(self.n_treatments)], axis=1)
                Ym_mt = (T.T * yf).T + (1-T) * Ycf_mt
                d_loss_mtg, d_acc_mtg = self.discriminator.train_on_batch([X, Ym_mt], T)

                # oh generator
                X, Z, T, yf = get_batch()
                Ycf_oh = self.oh_generator.predict([X, Z, T, yf])
                Ym_oh = (T.T * yf).T + (1-T) * Ycf_oh
                d_loss_ohg, d_acc_ohg = self.discriminator.train_on_batch([X, Ym_oh], T)

            ## Train generators
            # mt generator
            for t in range(self.n_treatments):
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
                mts_loss = self.mt_gan_selectors[t].train_on_batch([X, Z, T, yf], T)

            # oh generator
            X, Z, T, yf = get_batch()
            ohg_loss = self.oh_gan_generator.train_on_batch([X, Z, T, yf], T)

            if it % (n_iters//10) == 0:
                print(f'Iter: {it:4d}/{n_iters:4d}')
                print(f'd_loss_mtg: {d_loss_mtg:2.4f} d_acc_mtg {d_acc_mtg:2.4f} mtg_loss: {mtg_loss:2.4f} mts_loss: {mts_loss:2.4f}')
                print(f'd_loss_ohg: {d_loss_ohg:2.4f} d_acc_ohg {d_acc_ohg:2.4f} ohg_loss: {ohg_loss:2.4f}')

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

    def build_selectors(self): # todo init uniform: zeros?
        X = Input((self.n_features,), name='X')
        hidden = X
        for h_layer in range(self.h_layers):
            hidden = Dense(self.h_dim, activation=self.activation, name=f'h_{h_layer+1}')(hidden)
        Ss = [Dense(self.n_features, activation='sigmoid', name=f'S_{t+1}')(hidden) for t in range(self.n_treatments)]
        return [Model(X, Ss[t], name=f'MT_Selector_{t+1}') for t in range(self.n_treatments)]

    def build_multi_task_generators(self):
        xs = Input((self.n_features,), name='xs')
        Z = Input((self.n_treatments,), name='Z')
        hidden = Concatenate()([xs, Z])
        for h_layer in range(self.h_layers):
            hidden = Dense(self.h_dim, activation=self.activation, name=f'h_{h_layer+1}')(hidden)
        ycfs = [Dense(1, name=f'ycf_{t+1}')(hidden) for t in range(self.n_treatments)]
        return [Model([xs, Z], ycfs[t], name=f'MT_Generator_{t+1}') for t in range(self.n_treatments)]

    def build_one_hot_generator(self):
        X = Input((self.n_features,), name='X')
        Z = Input((self.n_treatments,), name='Z')
        T = Input((self.n_treatments,), name='T')
        yf = Input((1,), name='yf')
        hidden = Concatenate()([X, Z, T, yf])
        for h_layer in range(self.h_layers):
            hidden = Dense(self.h_dim, activation=self.activation, name=f'h_{h_layer+1}')(hidden)
        Ycf = Dense(self.n_treatments, name='Ycf')(hidden)
        return Model([X, Z, T, yf], Ycf, name='OH_Generator')

    def build_discriminator(self):
        X = Input((self.n_features,), name='X')
        Ym = Input((self.n_treatments,), name='Ym')
        hidden = Concatenate()([X, Ym])
        for h_layer in range(self.h_layers):
            hidden = Dense(self.h_dim, activation=self.activation, name=f'h_{h_layer+1}')(hidden)
        Tm = Dense(self.n_treatments, activation='softmax', name='Tm')(hidden)
        return Model([X, Ym], Tm, name='Discriminator')

    def build_inference(self):
        X = Input((self.n_features,), name='X')
        hidden = X
        for h_layer in range(self.h_layers):
            hidden = Dense(self.h_dim, activation=self.activation, name=f'h_{h_layer+1}')(hidden)
        Y_pred = Concatenate(name='Y_pred')([Dense(1, activation='sigmoid', name=f'head_{t+1}')(hidden) for t in range(self.n_treatments)])
        return Model(X, Y_pred, name='Inference')

    def sample_Z(self, N):
        return np.random.uniform(-1, 1, size=[N, self.n_treatments])

if __name__ == '__main__':
    np.set_printoptions(formatter={'float_kind': lambda x: f'{x:.4f}'})

    # Load data
    data = pd.read_csv('Twin_Data.csv').values
    N = data.shape[0]
    m = 30
    k = 2

    # Preprocess
    X = data[:, :m]
    X = X - X.min(axis=0)
    X = X / X.max(axis=0)
    Y = data[:, m:]
    Y[Y>365] = 365
    Y = 1 - Y/365.0
    t = np.random.randint(k, size=N)
    T = to_categorical(t, num_classes=k, dtype='int32')
    yf = np.choose(t, Y.T)

    # Train/test split
    N_train = N
    while N_train > 0.8 * N:
        N_train -= params['batch_size']
    X_train = X[:N_train]
    yf_train = yf[:N_train]
    T_train = T[:N_train]
    X_test = X[N_train:]
    yf_test = yf[N_train:]
    T_test = T[N_train:]

    # MT_GANITE
    mt_ganite = MT_GANITE(m, k)
    mt_ganite.train(X_train, T_train, yf_train)
    print(mt_ganite.predict(X_test))
    print(Y[N_train:])
