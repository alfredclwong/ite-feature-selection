from keras.layers import Input, Dense, Lambda, Concatenate, Multiply, dot
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
    "alpha": 2,
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
        self.alpha = params["alpha"]

        # Build components
        self.selector = self.build_selector()
        self.sampler = Lambda(lambda s: K.random_binomial((1, self.m), s))
        self.mt_generators = [self.build_multi_task_generator(t) for t in range(self.k)]
        self.oh_generator = self.build_one_hot_generator()
        def mix(inputs):
            [yf, T, Ycfs] = inputs
            return T * yf + (1-T) * Ycfs
        self.mixer = Lambda(mix)
        self.discriminator = self.build_discriminator()
        self.inference = self.build_inference()

        ## Define structure
        # Selector
        X = Input((self.m,), name="X")
        S = self.selector(X)
        # Generators
        Z = Input((self.k,), name="Z")
        # Multi-task Generator
        masks = [self.sampler(s) for s in S]
        Xs = [Multiply(name=f"X_{t+1}")([X, masks[t]]) for t in range(self.k)]
        ycfs_mt = [self.mt_generators[t]([Xs[t], Z]) for t in range(self.k)]
        Ycf_mt = Concatenate(name="Ycfs_mt")(ycfs_mt)
        # One-hot Generator
        T = Input((self.k,), name="T")
        yf = Input((1,), name="yf")
        Ycf_oh = self.oh_generator([X, Z, T, yf])
        # Discriminator
        Ym_mt = self.mixer([yf, T, Ycf_mt])
        T_mt = self.discriminator([X, Ym_mt])
        Ym_oh = self.mixer([yf, T, Ycf_oh])
        T_oh = self.discriminator([X, Ym_oh])
        # Inference
        Y_oh = self.inference(X)

        # Define models
        self.mt_ganite_cfs = [Model([X, Z, T, yf], T_mt, name=f"MT_GANITE_CF_{t+1}") for t in range(self.k)]
        self.oh_ganite_cf = Model([X, Z, T, yf], T_oh, name="OH_GANITE_CF")
        #self.mt_ganite = Model([X, Z], Ycfs_mt, name="MT_GANITE")
        #self.oh_ganite = Model([X, Z, T, yf], Y_oh, name="OH_GANITE")

        # Define losses
        def d_loss_fn():
            return losses.categorical_crossentropy
        def mtg_loss_fn(yf, ycf):
            supervised_loss = K.square(yf - ycf)
            def loss(y_true, y_pred):
                return supervised_loss - self.alpha * d_loss_fn()(y_true, y_pred)
            return loss
        def ohg_loss_fn(yf, T, Ycf):
            supervised_loss = K.square(yf - dot([T, Ycf], axes=-1))
            def loss(y_true, y_pred):
                return supervised_loss - self.alpha * d_loss_fn()(y_true, y_pred)
            return loss
        def s_loss_fn():
            pass

        # Compile neural networks
        self.discriminator.compile(loss=d_loss_fn(), optimizer=self.optimizer)
        self.discriminator.trainable = False
        for mt_generator in self.mt_generators:
            mt_generator.trainable = False
        for t in range(self.k):
            self.mt_generators[t].trainable = True
            self.mt_ganite_cfs[t].compile(loss=mtg_loss_fn(yf, ycfs_mt[t]), optimizer=self.optimizer)
            self.mt_generators[t].trainable = False
        self.oh_generator.compile(loss=ohg_loss_fn(yf, T, Ycf_oh), optimizer=self.optimizer)
        #self.selector.compile(...)

    def train(self, X_train, T_train, yf_train):
        def get_batch(batch_size=self.batch_size):
            N_train = X_train.shape[0]
            idx = np.random.randint(N_train, size=batch_size)
            X = X_train[idx]
            Z = self.sample_Z(batch_size)
            T = T_train[idx]
            yf = yf_train[idx]
            return X, Z, T, yf

        for it in tqdm(range(self.n_iters)):
            # Train discriminator on both mt and oh generators
            for i in range(self.gd_train_ratio):
                # mt generator
                X, Z, T, yf = get_batch()
                Ycf_mt = [self.mt_generators[t].predict([X, Z])]
                # TODO concat and sub in yfs
                #Ycf_mt = Concatenate(axis=1)([Lambda(lambda x: K.expand_dims(x, axis=1))(x) for x in Ycf_mt])
                #d_loss = self.discriminator.train_on_batch([X, Ym_mt], T)

                # oh generator
                X, Z, T, yf = get_batch()
                Ycf_oh = self.ganite([X, Z, T, yf])
                Ym_oh = T * yf + (1-T) * Ycf_oh
                d_loss = self.discriminator.train_on_batch([X, Ym_oh], T)

    def predict(self, X):
        return self.mt_ganite.predict([X, Z])

    def build_selector(self): # todo init uniform: zeros?
        X = Input((self.m,), name="X")
        hidden = X
        for h_layer in range(self.h_layers):
            hidden = Dense(self.h_dim, activation=self.activation, name=f"h_{h_layer+1}")(hidden)
        S = [Dense(self.m, activation="sigmoid", name=f"S_{t+1}")(hidden) for t in range(self.k)]
        return Model(X, S, name="Selector")

    def build_multi_task_generator(self, t):
        xs = Input((self.m,), name="xs")
        Z = Input((self.k,), name="Z")
        hidden = Concatenate()([xs, Z])
        for h_layer in range(self.h_layers):
            hidden = Dense(self.h_dim, activation=self.activation, name=f"h_{h_layer+1}")(hidden)
        ycf = Dense(1, name=f"ycf")(hidden)
        return Model([xs, Z], ycf, name=f"MT_Generator_{t+1}")

    def build_one_hot_generator(self):
        X = Input((self.m,), name="X")
        Z = Input((self.k,), name="Z")
        T = Input((self.k,), name="T")
        yf = Input((1,), name="yf")
        hidden = Concatenate()([X, Z, T, yf])
        for h_layer in range(self.h_layers):
            hidden = Dense(self.h_dim, activation=self.activation, name=f"h_{h_layer+1}")(hidden)
        Ycf = Dense(self.k, name="Ycf")(hidden)
        return Model([X, Z, T, yf], Ycf, name="OH_Generator")

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
