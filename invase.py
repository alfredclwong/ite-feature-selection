import os
from keras.layers import Dense, Input, Concatenate, Multiply
from keras.initializers import Constant
from keras.models import Model
import keras.backend as K
import numpy as np
from tqdm import tqdm

from synthetic_data import synthetic_data

eps = 1e-8

class INVASE:
    def __init__(self, n_features, gamma=0.05, relevant_features=None):
        self.n_features = n_features
        self.gamma = gamma
        self.relevant_features = relevant_features

        self.selector = self.build_selector()
        self.predictor = self.build_predictor()
        self.baseline = self.build_baseline()

        def selector_loss_fn(Ys, S_pred):
            # Extract s and y_pred/y_base/y_fact
            s = Ys[:, :self.n_features]
            Y_pred = Ys[:, self.n_features]
            Y_base = Ys[:, self.n_features + 1]
            Y_true = Ys[:, self.n_features + 2]

            # Reward regressions that are close to the baseline
            mse_pred = K.sum(K.square(Y_true - Y_pred))
            mse_base = K.sum(K.square(Y_true - Y_base))
            imitation_reward = mse_pred - mse_base

            # policy gradient
            loss1 = imitation_reward * K.sum(s*K.log(S_pred+eps) + (1-s)*K.log(1-S_pred+eps), axis=1)
            # Penalise complexity
            complexity_loss = self.gamma * K.mean(S_pred, axis=1)
            return K.mean(-loss1) + complexity_loss

        self.selector.compile(loss=selector_loss_fn, optimizer='adam')
        self.predictor.compile(loss='mse', optimizer='adam')
        self.baseline.compile(loss='mse', optimizer='adam')

    def build_selector(self):
        X = Input((self.n_features,))
        H = Dense(self.n_features, activation='relu')(X)
        H = Dense(self.n_features, activation='relu')(H)
        S = Dense(self.n_features, activation='sigmoid')(H)
        return Model(X, S)

    def build_predictor(self):
        X = Input((self.n_features,))  # not suppressed
        s = Input((self.n_features,))
        H = Multiply()([X, s])
        H = Dense(self.n_features, activation='relu')(H)
        H = Dense(self.n_features, activation='relu')(H)
        y = Dense(1)(H)
        return Model([X, s], y)

    def build_baseline(self):
        X = Input((self.n_features,))
        H = Dense(self.n_features, activation='relu')(X)
        H = Dense(self.n_features, activation='relu')(H)
        y = Dense(1)(H)
        return Model(X, y)

    def train(self, X, Y, n_iters, X_val=None, Y_val=None, batch_size=32):
        # Check params
        val = (X_val, Y_val) != (None, None)
        if type(n_iters) == int:
            n_iters = [n_iters, n_iters]
        N = X.shape[0]

        for it in tqdm(range(n_iters[1])):
            idx = np.random.randint(N, size=batch_size)
            S = np.nan_to_num(self.selector.predict(X[idx])) # selector probs
            s = np.random.binomial(1, S, size=(batch_size, self.n_features)) # sample from S

            Y_pred = self.predictor.predict([X[idx], s])
            pred_loss = self.predictor.train_on_batch([X[idx], s], Y[idx])

            Y_base = self.baseline.predict(X[idx])
            base_loss = self.baseline.train_on_batch(X[idx], Y[idx])

            Ys = np.concatenate([s, Y_pred, Y_base, Y[idx].reshape(-1,1)], axis=1)
            sele_loss = self.selector.train_on_batch(X[idx], Ys)

            feat_prob = np.mean(S, axis=0)
            if (it+1) % (n_iters[1]//10) == 0:
                if val:
                    Y_pred, _ = self.predict(X_val)
                    pred_mse = np.mean(np.square(Y_val - Y_pred))
                    Y_base = self.baseline.predict(X_val)
                    base_mse = np.mean(np.square(Y_val - Y_base))
                    pred_loss_str += f'\tval mse {pred_mse:.4f}'
                    base_loss_str += f'\tval mse {base_mse:.4f}'
                print(f'#{it}:\tsele loss {sele_loss}\n\tpred loss {pred_loss}\n\tbase loss {base_loss}')
                print(f'features\n{np.array2string(feat_prob)}')
                if self.relevant_features is not None:
                    print(f'true features\n{np.array2string(self.relevant_features)}')

    def predict(self, X, threshold=0.5):
        s = self.predict_features(X, threshold)
        Y = self.predictor.predict([X, s])
        return Y

    def predict_features(self, X, threshold=0.5):
        N = X.shape[0]
        S = self.selector.predict(X)
        if threshold == None:
            return S
        s = S > threshold
        return s


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    np.set_printoptions(formatter={"float_kind": lambda x: f"{x:.4f}"})
    X, t, Y = synthetic_data(n_features=30, models=[1,2,3])
    N, n_features = X.shape
    n_treatments = Y.shape[1]
    invase = INVASE(n_features, n_treatments)

    N_train = int(0.8 * N)
    X_train = X[N_train:]
    Y_train = Y[N_train:]
    X_test = X[:N_train]
    Y_test = Y[:N_train]
    invase.train(X_train, Y_train, [10000, 10000], X_test, Y_test)
    Y_pred, ss = invase.predict(X_test)
    X_str, Y_str, t_str, Y_pred_str, ss_str = map(np.array2string, [X, Y, t, Y_pred, ss.astype(int)])
    print('\n'.join(['X', X_str, 'Y', Y_str, 't', t_str, 'Y_pred', Y_pred_str, 'ss', ss_str]))
