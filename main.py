from keras.layers import Input, Dense, Lambda, Concatenate, Multiply, RepeatVector
from keras.models import Model
from keras import losses
from keras import backend as K
import numpy as np
from tqdm import tqdm

params = {
    "h_layers": 2,
    "h_dim": 8,
    "activation": "relu",
    "n_iters": 1000,
    "gd_train_ratio": 10,
    "batch_size": 128,
    "optimizer": "adam",
}

class MT_GANITE:
    def __init__(self, m, k):
        # m     Number of features in X
        # k     Number of treatments
        self.m = m
        self.k = k

        self.h_layers = params["h_layers"]
        self.h_dim = params["h_dim"]
        self.activation = params["activation"]
        self.n_iters = params["n_iters"]
        self.gd_train_ratio = params["gd_train_ratio"]
        self.batch_size = params["batch_size"]
        self.optimizer = params["optimizer"]

        # Build components
        self.selector = self.build_selector()
        self.mt_generator = self.build_multi_task_generator()
        self.oh_generator = self.build_one_hot_generator()
        def mix(inputs):
            [Yf, T, Ycfs] = inputs
            return T * Yf + (1-T) * Ycfs
        self.mixer = Lambda(mix)
        self.discriminator = self.build_discriminator()
        self.inference = self.build_inference()

        ## Define structure
        # Selector
        X = Input((self.m,), name="X")
        S = self.selector(X)
        masks = Lambda(lambda s: K.random_binomial((1, self.k, self.m), s), name=f"masks")(S)
        # Generators
        Z = Input((self.k,), name="Z")
        # Multi-task Generator
        Xs = Multiply(name="Xs")([RepeatVector(self.k)(X), masks])
        Ycfs_mt = self.mt_generator([Xs, Z])
        Ycfs_mt = Concatenate(name="Ycfs_mt")(Ycfs_mt)
        # One-hot Generator
        T = Input((self.k,), name="T")
        Yf = Input((1,), name="Yf")
        Ycfs_oh = self.oh_generator([X, Z, T, Yf])
        # Discriminator
        Ym_mt = self.mixer([Yf, T, Ycfs_mt])
        T_mt = self.discriminator([X, Ym_mt])
        Ym_oh = self.mixer([Yf, T, Ycfs_oh])
        T_oh = self.discriminator([X, Ym_oh])
        # Inference
        Y_oh = self.inference(X)

        # Define models
        self.mt_ganite_cf = [Model([X, Z], T_mt, name=f"MT_GANITE_CF_{t+1}") for t in range(self.k)]
        self.oh_ganite_cf = Model([X, Z, T_, Yf], T_oh, name="OH_GANITE_CF")
        self.mt_ganite = Model([X, Z], Ycfs_mt, name="MT_GANITE")
        self.oh_ganite = Model([X, Z, T, Yf], Y_oh, name="OH_GANITE")

        # Define losses
        def d_loss_fn(y_true, y_pred):
            return losses.categorical_crossentropy(y_true, y_pred)
        def mtg_loss_fn():
            pass
        def ohg_loss_fn():
            pass

        # Compile neural networks
        self.discriminator.compile(loss=d_loss_fn, optimizer=self.optimizer)
        self.discriminator.trainable = False
        for Ycf_mt_model in self.mt_ganite:
            Ycf_mt_model.compile(loss=mtg_loss_fn, optimizer=self.optimizer)

    def train(self, X_train, T_train, Yf_train):
        def get_batch(batch_size=self.batch_size):
            N_train = X_train.shape[0]
            idx = np.random.randint(N_train, size=batch_size)
            X = X_train[idx]
            Z = self.sample_Z(batch_size)
            T = T_train[idx]
            Yf = Yf_train[idx]
            return X, Z, T, Yf

        for it in tqdm(range(self.n_iters)):
            # Train discriminator on both mt and oh generators
            for i in range(self.gd_train_ratio):
                # mt generator
                X, Z, T, Yf = get_batch()
                Ycfs_mt = self.mt_ganite([X, Z])
                Ycfs_mt = Concatenate(name="Ycfs_mt")(Ycfs_mt)
                Ym_mt = T * Yf + (1-T) * Ycfs_mt
                d_loss = self.discriminator.train_on_batch([X, Ym_mt], T)

                # oh generator
                X, Z, T, Yf = get_batch()
                Ycfs_oh = self.ganite([X, Z, T, Yf])
                Ym_oh = T * Yf + (1-T) * Ycfs_oh
                d_loss = self.discriminator.train_on_batch([X, Ym_oh], T)

    def predict(self, X):
        return self.mt_ganite.predict([X, Z])

    def build_selector(self):
        X = Input((self.m,), name="X")
        hidden = X
        for h_layer in range(self.h_layers):
            hidden = Dense(self.h_dim, activation=self.activation, name=f"h_{h_layer+1}")(hidden)
        S = [Dense(self.m, activation="sigmoid", name=f"S_{t+1}")(hidden) for t in range(self.k)]
        S = Concatenate(axis=1)([Lambda(lambda x: K.expand_dims(x, axis=1))(s) for s in S])
        return Model(X, S, name="Selector")

    def build_multi_task_generator(self):
        Xs = Input((self.k, self.m), name="Xs")
        Z = Input((self.k,), name="Z")
        Ycfs = []
        for t in range(self.k):
            hidden = Concatenate()([Lambda(lambda x: x[:,t,:])(Xs), Z])
            for h_layer in range(self.h_layers):
                hidden = Dense(self.h_dim, activation=self.activation, name=f"h_{t+1}_{h_layer+1}")(hidden)
            Ycfs.append(Dense(1, name=f"Ycf_{t+1}")(hidden))
        Ycf = Concatenate(name="Ycf")(Ycfs)
        return Model([Xs, Z], Ycfs, name="MT_Generator")

    def build_one_hot_generator(self):
        X = Input((self.m,), name="X")
        Z = Input((self.k,), name="Z")
        T = Input((self.k,), name="T")
        Yf = Input((1,), name="Yf")
        hidden = Concatenate()([X, Z, T, Yf])
        for h_layer in range(self.h_layers):
            hidden = Dense(self.h_dim, activation=self.activation, name=f"h_{h_layer+1}")(hidden)
        Ycf = Dense(self.k, name="Ycf")(hidden)
        return Model([X, Z, T, Yf], Ycf, name="OH_Generator")

    def build_discriminator(self):
        X = Input((self.m,), name="X")
        Ym = Input((self.k,), name="Ym")
        hidden = Concatenate()([X, Ym])
        for h_layer in range(self.h_layers):
            hidden = Dense(self.h_dim, activation=self.activation, name=f"h_{h_layer+1}")(hidden)
        Tm = Dense(self.k, activation="softmax", name="Tm")(hidden)
        return Model([X, Ym], Tm, name="Discriminator")

    def build_inference(self):
        X = Input((self.m,), name="X")
        hidden = X
        for h_layer in range(self.h_layers):
            hidden = Dense(self.h_dim, activation=self.activation, name=f"h_{h_layer+1}")(hidden)
        Y = Dense(self.k, name="Y")(hidden)
        return Model(X, Y, name="Predictor")

    def sample_Z(self, N):
        return np.random.uniform(-1, 1, size=[N, self.k])

if __name__ == "__main__":
    m = 30
    k = 2
    mt_ganite = MT_GANITE(m, k)
