from keras.layers import Dense, Input, Concatenate
from keras.initializers import Constant
from keras.models import Model
import keras.backend as K
import numpy as np
from tqdm import tqdm

from synthetic_data import synthetic_data


class INVASE:
    def __init__(self, n_features, n_treatments, gamma=0.01):
        self.n_features = n_features
        self.n_treatments = n_treatments
        self.gamma = gamma

        self.selectors = self.build_selectors()
        self.predictors = [self.build_predictor() for t in range(self.n_treatments)]
        self.baselines = [self.build_baseline() for t in range(self.n_treatments)]

        def selector_loss_fn(y_true, y_pred):
            complexity_loss = self.gamma * K.mean(y_pred, axis=1)
            s = y_true[:, :self.n_features]
            y_predictor = y_true[:, self.n_features]
            y_baseline = y_true[:, self.n_features + 1]
            y_factual = y_true[:, self.n_features + 2]
            #mse_pred = K.sum(K.square(y_factual - y_predictor))
            #mse_base = K.sum(K.square(y_factual - y_baseline))
            imitation_reward = -K.sum(K.square(y_predictor - y_baseline))
            policy_grad = imitation_reward * K.sum(s*K.log(y_pred+1e-8) + (1-s)*K.log(1-y_pred+1e-8), axis=1)
            return K.mean(-policy_grad) + complexity_loss

        for selector, predictor, baseline in zip(self.selectors, self.predictors, self.baselines):
            selector.compile(loss=selector_loss_fn, optimizer='adam')
            predictor.compile(loss='mse', optimizer='adam')
            baseline.compile(loss='mse', optimizer='adam')

    def build_selectors(self):
        X = Input((self.n_features,))
        H = Dense(self.n_features, activation='relu')(X)
        H = Dense(self.n_features, activation='relu')(H)
        Ss = [Dense(self.n_features, activation='sigmoid')(H) for t in range(self.n_treatments)]
        return [Model(X, S) for S in Ss]

    def build_predictor(self):
        x = Input((self.n_features,))  # suppressed
        s = Input((self.n_features,))
        H = Concatenate()([x, s])
        H = Dense(self.n_features, activation='relu')(H)
        H = Dense(self.n_features, activation='relu')(H)
        y = Dense(1, activation='sigmoid')(H)
        return Model([x, s], y)

    def build_baseline(self):
        X = Input((self.n_features,))
        H = Dense(self.n_features, activation='relu')(X)
        H = Dense(self.n_features, activation='relu')(H)
        y = Dense(1, activation='sigmoid')(H)
        return Model(X, y)

    def train(self, n_iters, X, Y, batch_size=32):
        N = X.shape[0]
        sele_loss = np.zeros(self.n_treatments)
        pred_loss = np.zeros(self.n_treatments)
        base_loss = np.zeros(self.n_treatments)
        feat_prob = np.zeros((self.n_treatments, self.n_features))
        for it in tqdm(range(n_iters)):
            idx = np.random.randint(N, size=batch_size)
            for t in range(self.n_treatments):
                S = np.nan_to_num(self.selectors[t].predict(X[idx]))
                s = np.random.binomial(1, S, size=(batch_size, self.n_features))
                x = X[idx] * s
                y_pred = self.predictors[t].predict([x, S])
                y_base = self.baselines[t].predict(X[idx])
                y_fact = Y[idx, t].reshape(batch_size, 1)
                ys = np.concatenate([s, y_pred, y_base, y_fact], axis=1)
                sele_loss[t] = self.selectors[t].train_on_batch(X[idx], ys)
                pred_loss[t] = self.predictors[t].train_on_batch([x, s], y_fact)
                base_loss[t] = self.baselines[t].train_on_batch(X[idx], y_fact)
                feat_prob[t,:] = np.mean(S, axis=0)
            if it % (n_iters//10) == 0:
                sele_loss_str, pred_loss_str, base_loss_str = map(np.array2string, [sele_loss, pred_loss, base_loss])
                print(f'#{it}:\tsele loss {sele_loss_str}\n\tpred loss {pred_loss_str}\n\tbase loss {base_loss_str}')
                print(f'features\n{np.array2string(feat_prob)}')

    def predict(self, X):
        N = X.shape[0]
        Y = np.zeros((N, self.n_treatments))
        ss = np.zeros((N, self.n_treatments, self.n_features))
        for t in range(self.n_treatments):
            S = self.selectors[t].predict(X)
            s = np.random.binomial(1, S, size=(N, self.n_features))
            x = X * s
            Y[:,t] = self.predictors[t].predict([x, s]).flatten()
            ss[:,t,:] = s
        return Y, ss


if __name__ == '__main__':
    np.set_printoptions(formatter={"float_kind": lambda x: f"{x:.4f}"})
    X, t, Y = synthetic_data(models=[1,2,3])
    N, n_features = X.shape
    n_treatments = Y.shape[1]
    invase = INVASE(n_features, n_treatments)
    invase.train(10000, X, Y)
    Y_pred, ss = invase.predict(X)
    X_str, Y_str, t_str, Y_pred_str, ss_str = map(np.array2string, [X, Y, t, Y_pred, ss.astype(int)])
    print('\n'.join(['X', X_str, 'Y', Y_str, 't', t_str, 'Y_pred', Y_pred_str, 'ss', ss_str]))
