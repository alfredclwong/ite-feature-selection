from keras.layers import Input, Dense, Lambda, Concatenate, Multiply, RepeatVector
from keras.models import Model
from keras import backend as K

params = {
    "h_layers": 2,
    "h_dim": 8,
    "activation": "relu",
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

        # Build components
        self.selector = self.build_selector()
        self.mt_generator = self.build_multi_task_generator()
        self.oh_generator = self.build_one_hot_generator()
        self.discriminator = self.build_discriminator()
        self.predictor = self.build_predictor()

        # Define structure
        X = Input((self.m,), name="X")
        S = self.selector(X)
        masks = Lambda(lambda s: K.random_binomial((1, self.k, self.m), s), name=f"masks")(S)
        Xs = Multiply(name="Xs")([RepeatVector(self.k)(X), masks])
        Z = Input((self.k,), name="Z")
        Ycf_mt = self.mt_generator([Xs, Z])

        T = Input((self.k,), name="T")
        Yf = Input((1,), name="Yf")
        Ycf_oh = self.oh_generator([X, Z, T, Yf])
        Ym_mt = T * Yf + (1-T) * Ycf_mt
        Ym_oh = T * Yf + (1-T) * Ycf_oh
        Y = self.predictor(X)

        # Define models
        self.mt_ganite = Model([X, Z], Ycf_mt, name="MT_GANITE")
        self.ganite = Model([X, Z, T, Yf], Y, name="GANITE")

        # Define losses

        # Compile neural networks

    def train(self, X_train, T_train, Yf_train):
        #
        pass

    def predict(self, X):
        return self.mt_ganite.predict(X)

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
        return Model([Xs, Z], Ycf, name="MT_Generator")

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

    def build_predictor(self):
        X = Input((self.m,), name="X")
        hidden = X
        for h_layer in range(self.h_layers):
            hidden = Dense(self.h_dim, activation=self.activation, name=f"h_{h_layer+1}")(hidden)
        Y = Dense(self.k, name="Y")(hidden)
        return Model(X, Y, name="Predictor")

if __name__ == "__main__":
    m = 30
    k = 2
    mt_ganite = MT_GANITE(m, k)
