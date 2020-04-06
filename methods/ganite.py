from keras.layers import Input, Dense, Lambda, dot, Concatenate, Activation
from keras.models import Model
from keras.utils import to_categorical
from keras import losses
from keras import backend as K
import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm

from methods.method import Method
from synthetic_data import synthetic_data
from metrics import PEHE

hyperparams = {
    'h_layers':     2,
    'h_dim':        16,
    'alpha':        1,
    'beta':         0.5,
    'batch_size':   128,
    'k1':           10,
    'k2':           1,
}

class GANITE(Method):
    def __init__(self, n_features, n_treatments, optimizer='adam'):
        super().__init__(n_features, n_treatments)
        self.h_layers = hyperparams['h_layers']
        self.h_dim = hyperparams['h_dim']
        self.batch_size = hyperparams['batch_size']
        self.alpha = hyperparams['alpha']
        self.beta = hyperparams['beta']
        self.k1 = hyperparams['k1']
        self.k2 = hyperparams['k2']
        self.gan, self.generator, self.discriminator, self.inference = self.build_GANITE(optimizer)

    def train(self, X_train, T_train, Yf_train, n_epochs, X_val=None, T_val=None, Yf_val=None, Y_val=None):
        val = all([_val is not None for _val in [X_val, T_val, Yf_val]])
        T_train = to_categorical(T_train)
        def get_batch():
            N_train = X_train.shape[0]
            idx = np.random.randint(N_train, size=self.batch_size)
            X = X_train[idx]
            Yf = Yf_train[idx]
            T = T_train[idx]
            Z = self.sample_Z(self.batch_size)
            return X, Yf, T, Z

        # Train GAN (G and D)
        for epoch in tqdm(range(n_epochs[0])):
            # Train D over k1 batches per epoch
            for i in range(self.k1):
                X, Yf, T, Z = get_batch()
                Y_pred = self.generator.predict([X, Yf, T, Z])
                Y_bar = (T.T * Yf).T + (1.-T) * Y_pred
                d_loss, d_acc = self.discriminator.train_on_batch([X, Y_bar], T)

            # Train G over k2 batches per epoch
            for i in range(self.k2):
                X, Yf, T, Z = get_batch()
                g_loss = self.gan.train_on_batch([X, Yf, T, Z], T)

            if epoch % (n_epochs[0]//5) == 0:
                print(f'Epoch: {epoch}\nD loss: {d_loss:.4f}\tD acc: {d_acc:.4f}\nG loss: {g_loss:.4f}')
                if val:
                    Z_val = self.sample_Z(X_val.shape[0])
                    Y_pred_val = self.generator.predict([X_val, Yf_val, T_val, Z_val])
                    g_mse_val = 0
                    g_pehe_val = 0
                    g_mse_val = np.mean(np.square(Y_pred_val - Y_val))
                    g_pehe_val = PEHE(Y_val, Y_pred_val)
                    print(f'MSE (val): {g_mse_val:.4f}\tPEHE (val): {g_pehe_val:.4f}')

        # Train I
        for epoch in tqdm(range(n_epochs[1])):
            X, Yf, T, Z = get_batch()
            Y_pred = self.generator.predict([X, Yf, T, Z])
            Y_bar = (T.T * Yf).T + (1.-T) * Y_pred
            i_loss = self.inference.train_on_batch(X, Y_bar)

            if epoch % (n_epochs[1]//5) == 0:
                print(f'Epoch: {epoch}\nI loss (train): {i_loss:.4f}')
                if val:
                    Y_pred_val = self.predict(X_val)
                    i_mse_val = np.mean(np.square(Y_pred_val - Y_val))
                    i_pehe_val = PEHE(Y_val, Y_pred_val)
                    print(f'I MSE (val): {i_mse_val:.4f}\tI PEHE (val): {i_pehe_val:.4f}')

    def predict_counterfactuals(self, X, Yf, T):
        Z = self.sample_Z(X.shape[0])
        Y_pred = self.generator.predict([X, Yf, T, Z])
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
        Y_pred = Concatenate(name='Y_pred')([Dense(1, activation='sigmoid', name=f'Head_{t+1}')(hidden) for t in range(self.n_treatments)])
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
        Y_pred = Concatenate(name='Y_pred')([Dense(1, activation='sigmoid', name=f'Head_{t+1}')(hidden) for t in range(self.n_treatments)])
        return Model(X, Y_pred, name='Inference')

    def sample_Z(self, N):
        return np.random.uniform(-1, 1, size=[N, self.n_treatments])

    def get_weights(self):
        weights = {}
        weights['generator'] = self.generator.get_weights()
        weights['discriminator'] = self.discriminator.get_weights()
        weights['inference'] = self.inference.get_weights()
        return weights


if __name__ == '__main__':
    np.set_printoptions(formatter={'float_kind': lambda x: f'{x:.4f}'})
    
    data = 'Jobs_Lalonde_Data.csv'
    have_cfs = True
    if data is None:
        X, t, Y = synthetic_data(n_features=30, models=[1,2])
        N, n_features = X.shape
        n_treatments = Y.shape[1]
    elif data == 'Twins_Data.csv':
        # Load data
        data = pd.read_csv(data).values
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
    elif data == 'Jobs_Lalonde_Data.csv':
        data = pd.read_csv(data).values
        X = data[:, :-2]
        t = data[:, -2]
        Yf = data[:, -1]
        N = data.shape[0]
        n_features = data.shape[1] - 2
        n_treatments = 2
        have_cfs = False
    T = to_categorical(t, num_classes=n_treatments, dtype='int32')
    if have_cfs:
        Yf = np.choose(t, YT)

    # Train/test split
    N_train = int(N * 0.8)
    X_train = X[:N_train]
    Yf_train = Yf[:N_train]
    T_train = T[:N_train]
    X_test = X[N_train:]
    Yf_test = Yf[N_train:]
    T_test = T[N_train:]
    Y_test = None
    if have_cfs:
        Y_test = Y[N_train:]

    # GANITE
    ganite = GANITE(n_features, n_treatments)
    ganite.train([2000, 1000], X_train, T_train, Yf_train, X_test, T_test, Yf_test, Y_test)

    # Print results
    G_pred, D_pred = ganite.predict_counterfactuals(X_test, Yf_test, T_test)
    I_pred = ganite.predict(X_test)
    results = {'G_pred': G_pred, 'I_pred': I_pred, 'Y_test': Y_test, 'D_pred': D_pred, 'T_test': T_test}
    for desc, result in  results.items():
        print(f'{desc}\n{result}')
