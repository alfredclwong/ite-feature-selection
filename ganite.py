from keras.layers import Input, Dense, Lambda, dot, Concatenate, Activation
from keras.models import Model
from keras.utils import to_categorical
from keras import losses
from keras import backend as K
import tensorflow as tf
import numpy as np
import pandas as pd
from tqdm import tqdm

class GANITE():
    def __init__(self, params, hyperparams, optimizer="adam"):
        self.n_features = params["n_features"]
        self.n_classes = params["n_classes"]
        self.h_layers = hyperparams["h_layers"]
        self.h_dim = hyperparams["h_dim"]
        self.batch_size = hyperparams["batch_size"]
        self.n_epochs = hyperparams["n_epochs"]
        self.alpha = hyperparams["alpha"]
        self.k1 = hyperparams["k1"]
        self.k2 = hyperparams["k2"]
        self.gan, self.generator, self.discriminator = self.build_GAN(params, hyperparams, optimizer)

    def train(self, X_train, Y_fact_train, T_train):
        def get_batch():
            N_train = X_train.shape[0]
            idx = np.random.randint(N_train, size=self.batch_size)
            X = X_train[idx]
            Y_fact = Y_fact_train[idx]
            T = T_train[idx]
            Z = self.sample_Z(self.batch_size)
            return X, Y_fact, T, Z

        for epoch in tqdm(range(self.n_epochs)):
            d_loss = g_loss = 0

            # Train D over k1 batches per epoch
            for i in range(self.k1):
                X, Y_fact, T, Z = get_batch()
                Y_pred = self.generator.predict([X, Y_fact, T, Z])
                Y_bar = (T.T * Y_fact).T + (1.-T) * Y_pred
                d_loss = self.discriminator.train_on_batch([X, Y_bar], T)

            # Train G over k2 batches per epoch
            for i in range(self.k2):
                X, Y_fact, T, Z = get_batch()
                g_loss = self.gan.train_on_batch([X, Y_fact, T, Z], T)

            if epoch % (self.n_epochs/10) == 0:
                print(f"Epoch: {epoch}\nD loss: {d_loss:.4f}\nG loss: {g_loss:.4f}")

    def predict(self, X, Y_fact, T):
        Z = self.sample_Z(X.shape[0])
        Y_pred = self.generator.predict([X, Y_fact, T, Z])
        T_pred = self.gan.predict([X, Y_fact, T, Z])
        return Y_pred, T_pred

    def build_GAN(self, params, hyperparams, optimizer="adam"):
        # Build generator (G) and discriminator (D) models
        generator = self.build_generator()
        discriminator = self.build_discriminator()

        # Define D loss and compile
        #'''
        def d_loss_fn(y_true, y_pred):
            indecisiveness = 1
            for i in range(n_classes):
                indecisiveness = indecisiveness * y_pred[:, i]
            return losses.categorical_crossentropy(y_true, y_pred) + indecisiveness
        #'''
        #d_loss_fn = losses.categorical_crossentropy
        discriminator.compile(loss=d_loss_fn, optimizer=optimizer)

        # Build GAN model
        X = Input(shape=(n_features,), name="X")
        Y_fact = Input(shape=(1,), name="Y_fact")
        T = Input(shape=(n_classes,), name="T")
        Z = Input(shape=(n_classes,), name="Z")
        Y_pred = generator([X, Y_fact, T, Z])
        def make_y_bar(inputs):
            [Y_fact, T, Y_pred] = inputs
            return T * Y_fact + (1.-T) * Y_pred
        Y_bar = Lambda(make_y_bar, name="Y_bar")([Y_fact, T, Y_pred])
        T_pred = discriminator([X, Y_bar])
        gan = Model([X, Y_fact, T, Z], T_pred, name="GAN")

        # Define G loss and compile
        def g_loss_fn(Y_fact, T, Y_pred, T_pred):
            supervised_loss = K.square(Y_fact - dot([T, Y_pred], axes=-1))
            def loss(y_true, y_pred):
                return supervised_loss - self.alpha * d_loss_fn(y_true, y_pred)
            return loss
        discriminator.trainable = False
        gan.compile(loss=g_loss_fn(Y_fact, T, Y_pred, T_pred), optimizer=optimizer)

        generator.summary()
        discriminator.summary()
        gan.summary()
        return gan, generator, discriminator

    def build_generator(self, activation="relu"):
        # Inputs: X (n_features), Y_fact (1), T (n_classes), Z (n_classes?)
        # Output: Y_pred (n_classes)
        X = Input(shape=(self.n_features,), name="X")
        Y_fact = Input(shape=(1,), name="Y_fact")
        T = Input(shape=(self.n_classes,), name="T")
        Z = Input(shape=(self.n_classes,), name="Z")
        hidden = Concatenate(name="Inputs")([X, Y_fact, T, Z])
        for h_layer in range(self.h_layers):
            hidden = Dense(self.h_dim, activation=activation, name=f"Hidden_{h_layer+1}")(hidden)
        Y_pred = Concatenate(name="Y_pred")([Dense(1, activation="sigmoid", name=f"Head_{t+1}")(hidden) for t in range(self.n_classes)])
        #Y_pred = Dense(self.n_classes, activation="sigmoid")(hidden)
        return Model([X, Y_fact, T, Z], Y_pred, name="Generator")

    def build_discriminator(self, activation="relu"):
        # Inputs: X (n_features), Y_bar (n_classes)
        # Output: d (n_classes)
        X = Input(shape=(self.n_features,), name="X")
        Y_bar = Input(shape=(self.n_classes,), name="Y_bar")
        hidden = Concatenate(name="Inputs")([X, Y_bar])
        for h_layer in range(self.h_layers):
            hidden = Dense(self.h_dim, activation=activation, name=f"Hidden_{h_layer+1}")(hidden)
        #T_pred = Concatenate()([Dense(1)(hidden) for _ in range(self.n_classes)])
        #T_pred = Activation("softmax")(T_pred)
        T_pred = Dense(self.n_classes, activation="softmax", name="T_pred")(hidden)
        return Model([X, Y_bar], T_pred, name="Discriminator")

    def sample_Z(self, N):
        return np.random.uniform(-1, 1, size=[N, self.n_classes])


if __name__ == "__main__":
    np.set_printoptions(formatter={"float_kind": lambda x: f"{x:.4f}"})
    
    # Load data
    data = pd.read_csv("Twin_Data.csv").values
    N = data.shape[0]
    n_features = 30
    n_classes = 2

    # Preprocess
    X = data[:, :n_features]
    X = X - X.min(axis=0)
    X = X / X.max(axis=0)
    Y = data[:, n_features:]
    Y[Y>365] = 365
    Y = 1 - Y/365.0
    t = np.random.randint(n_classes, size=N)
    T = to_categorical(t, num_classes=n_classes, dtype="int32")
    Y_fact = np.choose(t, Y.T)

    # Train/test split
    N_train = int(N * 0.8)
    X_train = X[:N_train]
    Y_fact_train = Y_fact[:N_train]
    T_train = T[:N_train]
    X_test = X[N_train:]
    Y_fact_test = Y_fact[N_train:]
    T_test = T[N_train:]

    # GANITE
    params = {
            "n_features":   n_features,
            "n_classes":    n_classes,
    }
    hyperparams = {
            "h_layers":     2,
            "h_dim":        16,
            "alpha":        1.5,
            "batch_size":   128,
            "n_epochs":     2000,
            "k1":           10,
            "k2":           1,
    }
    ganite = GANITE(params, hyperparams)
    ganite.train(X_train, Y_fact_train, T_train)
    Y_pred, T_pred = ganite.predict(X_test, Y_fact_test, T_test)

    # Print results
    idx = np.any(Y[N_train:], axis=1)
    idx = np.ones(N-N_train, dtype="bool")
    idx[np.searchsorted(np.cumsum(idx), 10):] = False
    print(Y_pred[idx])
    #print(((T_test.T * Y_fact_test).T + (1.-T_test) * Y_pred))
    print(Y[N_train:][idx])
    print(T_pred[idx])
    print(T_test[idx])
    Y_bar = (T_test.T * Y_fact_test).T + (1.-T_test) * Y_pred
    print(np.mean(np.square(Y_bar-Y[N_train:]))*n_classes/(n_classes-1))
    print(np.mean(np.square(Y_pred-Y[N_train:]))*n_classes/(n_classes-1))
    print(K.eval(losses.categorical_crossentropy(T_test, T_pred))[idx])
