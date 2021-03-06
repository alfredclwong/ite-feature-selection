from keras.layers import Input, Dense, Lambda, dot, Concatenate, Activation
from keras.models import Model
from keras.utils import to_categorical
from keras import losses
from keras import backend as K
import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm

default_hyperparams = {
    'h_layers':     2,
    'h_dim':        16,
    'alpha':        1,
    'beta':         0.5,
    'batch_size':   128,
    'k1':           10,
    'k2':           1,
}

class GANITE():
    def __init__(self, n_features, n_treatments, hyperparams=default_hyperparams, optimizer='adam'):
        super().__init__(n_features, n_treatments)
        self.h_layers = hyperparams['h_layers']
        self.h_dim = hyperparams['h_dim']
        self.batch_size = hyperparams['batch_size']
        self.alpha = hyperparams['alpha']
        self.beta = hyperparams['beta']
        self.k1 = hyperparams['k1']
        self.k2 = hyperparams['k2']
        self.gan, self.generator, self.discriminator, self.inference = self.build_GANITE(optimizer)

    def train(self, X_train, T_train, Yf_train, n_epochs, X_val=None, T_val=None, Yf_val=None, Y_val=None, verbose=True):
        val = all([_val is not None for _val in [X_val, T_val, Yf_val, Y_val]])
        T_train, T_val = map(to_categorical, [T_train, T_val])
        def get_batch():
            N_train = X_train.shape[0]
            idx = np.random.randint(N_train, size=self.batch_size)
            X = X_train[idx]
            Yf = Yf_train[idx]
            T = T_train[idx]
            Z = self.sample_Z(self.batch_size)
            return X, Yf, T, Z

        # Train GAN (G and D)
        for epoch in tqdm(range(1, 1+n_epochs[0])):
            # Train D over k1 batches per epoch
            for i in range(self.k1):
                X, Yf, T, Z = get_batch()
                Y_pred = self.generator.predict([X, Yf, T, Z])
                Y_bar = (T.T * Yf).T + (1.-T) * Y_pred
                d_loss, d_acc = self.discriminator.evaluate([X, Y_bar], T, verbose=0)
                if d_acc > 0.9:
                    break
                d_loss, d_acc = self.discriminator.train_on_batch([X, Y_bar], T)

            # Train G over k2 batches per epoch
            for i in range(self.k2):
                X, Yf, T, Z = get_batch()
                g_loss = self.gan.train_on_batch([X, Yf, T, Z], T)

            if epoch % (n_epochs[0]//5) == 0 and verbose:
                Z_train = self.sample_Z(X_train.shape[0])
                Y_pred_train = self.generator.predict([X_train, Yf_train, T_train, Z_train])
                Y_bar_train = (T_train.T * Yf_train).T + (1.-T_train) * Y_pred_train
                d_loss_train, d_acc_train = self.discriminator.evaluate([X_train, Y_bar_train], T_train, verbose=0)
                g_loss_train = self.gan.evaluate([X_train, Yf_train, T_train, Z_train], T_train, verbose=0)
                if val:
                    Z_val = self.sample_Z(X_val.shape[0])
                    Y_pred_val = self.generator.predict([X_val, Yf_val, T_val, Z_val])
                    g_mse_val = 0
                    g_pehe_val = 0
                    g_mse_val = np.mean(np.square(Y_pred_val - Y_val))
                    g_pehe_val = PEHE(Y_val, Y_pred_val)
                    g_enormse = PEHE(Y_val, Y_pred_val, norm=True)
                print(f'Epoch: {epoch}\n[batch]\tD loss: {d_loss:.4f}\tD acc: {d_acc:.4f}\tG loss: {g_loss:.4f}\t({g_loss+self.alpha*d_loss:.4f})')
                print(f'[train]\tD loss: {d_loss_train:.4f}\tD acc: {d_acc_train:.4f}\tG loss: {g_loss_train:.4f}\t({g_loss_train+self.alpha*d_loss_train:.4f})')
                if val:
                    print(f'[val]\tMSE: {g_mse_val:.4f}\t\tPEHE: {g_pehe_val:.4f}\t\tENoRMSE: {g_enormse:.4f}')

        # Train I
        for epoch in tqdm(range(1, 1+n_epochs[1])):
            X, Yf, T, Z = get_batch()
            Y_pred = self.generator.predict([X, Yf, T, Z])
            Y_bar = (T.T * Yf).T + (1.-T) * Y_pred
            i_loss = self.inference.train_on_batch(X, Y_bar)

            if epoch % (n_epochs[1]//5) == 0 and verbose:
                print(f'Epoch: {epoch}\nI loss (train): {i_loss:.4f}')
                if val:
                    Y_pred_val = self.predict(X_val)
                    i_mse_val = np.mean(np.square(Y_pred_val - Y_val))
                    i_pehe_val = PEHE(Y_val, Y_pred_val)
                    print(f'I MSE (val): {i_mse_val:.4f}\tI PEHE (val): {i_pehe_val:.4f}')

        return g_loss

    def predict_counterfactuals(self, X, Yf, T, Y_bar=False):
        T = to_categorical(T)
        Z = self.sample_Z(X.shape[0])
        Y_pred = self.generator.predict([X, Yf, T, Z])
        if Y_bar:
            Y_pred = (T.T * Yf).T + (1.-T) * Y_pred
        T_pred = self.gan.predict([X, Yf, T, Z])
        return Y_pred, T_pred

    def predict(self, X):
        Y_pred = self.inference.predict(X)
        return Y_pred

    def build_GANITE(self, optimizer='adam'):
        # Build generator (G), discriminator (D) and inference (I) models
        generator = self.build_generator()
        discriminator = self.build_discriminator()
        inference = self.build_inference()

        # Define D loss and compile
        def d_loss_fn():
            return losses.categorical_crossentropy
        discriminator.compile(loss=d_loss_fn(), optimizer=optimizer, metrics=['categorical_accuracy'])

        # Build GAN model
        X = Input(shape=(self.n_features,), name='X')
        Yf = Input(shape=(1,), name='Yf')
        T = Input(shape=(self.n_treatments,), name='T')
        Z = Input(shape=(self.n_treatments,), name='Z')
        Y_pred = generator([X, Yf, T, Z])
        def make_y_bar(inputs):  # Y_bar = Y_pred with Yf substituted in
            [Yf, T, Y_pred] = inputs
            return T * Yf + (1.-T) * Y_pred
        Y_bar = Lambda(make_y_bar, name='Y_bar')([Yf, T, Y_pred])
        T_pred = discriminator([X, Y_bar])
        gan = Model([X, Yf, T, Z], T_pred, name='GAN')

        # Define G loss and compile
        def g_loss_fn(Yf, T, Y_pred):
            supervised_loss = K.square(Yf - dot([T, Y_pred], axes=-1))
            def loss(y_true, y_pred):
                return supervised_loss - self.alpha * d_loss_fn()(y_true, y_pred)
            return loss
        discriminator.trainable = False
        gan.compile(loss=g_loss_fn(Yf, T, Y_pred), optimizer=optimizer)

        # Define I loss and compile
        def i_loss_fn(Yf, T):
            def loss(y_true, y_pred):
                supervised_loss = K.square(Yf - dot([T, y_pred], axes=-1))
                Y_bar = make_y_bar([Yf, T, y_pred])
                counterfactual_loss = K.mean(K.square(y_pred - y_true), axis=-1)
                return supervised_loss + self.beta * counterfactual_loss
            return loss
        inference.compile(loss=losses.mean_squared_error, optimizer=optimizer)

        #generator.summary()
        #discriminator.summary()
        #gan.summary()
        #inference.summary()
        return gan, generator, discriminator, inference

    def build_generator(self, activation='relu'):
        # Inputs: X (n_features), Yf (1), T (n_treatments), Z (n_treatments?)
        # Output: Y_pred (n_treatments)
        X = Input(shape=(self.n_features,), name='X')
        Yf = Input(shape=(1,), name='Yf')
        T = Input(shape=(self.n_treatments,), name='T')
        Z = Input(shape=(self.n_treatments,), name='Z')
        hidden = Concatenate(name='Inputs')([X, Yf, T, Z])
        #hidden = Concatenate(name='Inputs')([X, Yf, T])
        for h_layer in range(self.h_layers):
            hidden = Dense(self.h_dim, activation=activation, name=f'Hidden_{h_layer+1}')(hidden)
        Y_pred = Concatenate(name='Y_pred')([Dense(1, name=f'Head_{t+1}')(hidden) for t in range(self.n_treatments)])
        return Model([X, Yf, T, Z], Y_pred, name='Generator')

    def build_discriminator(self, activation='relu'):
        # Inputs: X (n_features), Y_bar (n_treatments)
        # Output: T_pred (n_treatments)
        X = Input(shape=(self.n_features,), name='X')
        Y_bar = Input(shape=(self.n_treatments,), name='Y_bar')
        hidden = Concatenate(name='Inputs')([X, Y_bar])
        for h_layer in range(self.h_layers):
            hidden = Dense(self.h_dim, activation=activation, name=f'Hidden_{h_layer+1}')(hidden)
        T_pred = Dense(self.n_treatments, activation='softmax', name='T_pred')(hidden)
        def to_dist(inputs):
            return inputs / K.sum(inputs)
        #T_pred = Concatenate()([Dense(1, activation=None, name=f'Head_{t+1}')(hidden) for t in range(self.n_treatments)])
        #T_pred = Lambda(to_dist)(T_pred)
        return Model([X, Y_bar], T_pred, name='Discriminator')

    def build_inference(self, activation='relu'):
        # Inputs: X (n_features)
        # Output: Y_pred (n_treatments)
        X = Input(shape=(self.n_features,), name='X')
        hidden = X
        for h_layer in range(self.h_layers):
            hidden = Dense(self.h_dim, activation=activation, name=f'Hidden_{h_layer+1}')(hidden)
        Y_pred = Concatenate(name='Y_pred')([Dense(1, name=f'Head_{t+1}')(hidden) for t in range(self.n_treatments)])
        return Model(X, Y_pred, name='Inference')

    def sample_Z(self, N):
        return np.random.uniform(-1, 1, size=[N, self.n_treatments])

    def get_weights(self):
        weights = {}
        weights['generator'] = self.generator.get_weights()
        weights['discriminator'] = self.discriminator.get_weights()
        weights['inference'] = self.inference.get_weights()
        return weights
