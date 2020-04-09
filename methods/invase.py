import os
from keras.layers import Dense, Input, Concatenate, Multiply
from keras.initializers import Constant
from keras.models import Model
import keras.backend as K
from keras.utils import to_categorical
import numpy as np
from tqdm import tqdm
from sklearn.utils import class_weight

#from synthetic_data import synthetic_data

eps = 1e-8

class INVASE:
    def __init__(self, n_features, n_classes=0, lam=0.05, relevant_features=None, verbose=True):
        self.n_features = n_features
        self.n_classes = n_classes  # 0 = regression, 2+ = classification
        self.lam = lam
        self.relevant_features = relevant_features

        self.selector = self.build_selector()
        self.predictor = self.build_predictor()
        self.baseline = self.build_baseline()

        def selector_loss_fn(Ys, S_pred):
            # Extract s and y_pred/y_base/y_fact
            s = Ys[:, :self.n_features]
            if self.n_classes:
                Y_pred = Ys[:, self.n_features : self.n_features+self.n_classes]
                Y_base = Ys[:, self.n_features+self.n_classes : self.n_features+self.n_classes*2]
                Y_true = Ys[: ,self.n_features+self.n_classes*2 : ]

                L_pred = -K.sum(Y_pred * K.log(Y_true+eps), axis=1)
                L_base = -K.sum(Y_base * K.log(Y_true+eps), axis=1)
            else:
                Y_pred = Ys[:, self.n_features]
                Y_base = Ys[:, self.n_features + 1]
                Y_true = Ys[:, self.n_features + 2]

                # Reward regressions that are close to the baseline
                L_pred = -(K.square(Y_true - Y_pred))
                L_base = -(K.square(Y_true - Y_base))
            imitation_reward = L_pred - L_base

            # policy gradient
            loss1 = imitation_reward * K.sum(s*K.log(S_pred+eps) + (1-s)*K.log(1-S_pred+eps), axis=1)
            # Penalise complexity
            complexity_loss = self.lam * K.mean(S_pred, axis=1)
            return K.mean(-loss1) + complexity_loss

        self.selector.compile(loss=selector_loss_fn, optimizer='adam')
        loss = 'categorical_crossentropy' if self.n_classes else 'mse'
        self.predictor.compile(loss=loss, optimizer='adam')
        self.baseline.compile(loss=loss, optimizer='adam')

    def build_selector(self):
        X = Input((self.n_features,))
        H = Dense(self.n_features*2, activation='relu')(X)
        H = Dense(self.n_features*2, activation='relu')(H)
        S = Dense(self.n_features, activation='sigmoid')(H)
        return Model(X, S)

    def build_predictor(self):
        X = Input((self.n_features,))  # not suppressed
        s = Input((self.n_features,))
        H = Multiply()([X, s])         # suppressed
        H = Dense(self.n_features*2, activation='relu')(H)
        H = Dense(self.n_features*2, activation='relu')(H)
        y = Dense(self.n_classes, activation='softmax')(H) if self.n_classes else Dense(1)(H)
        return Model([X, s], y)

    def build_baseline(self):
        X = Input((self.n_features,))
        H = Dense(self.n_features*2, activation='relu')(X)
        H = Dense(self.n_features*2, activation='relu')(H)
        y = Dense(self.n_classes, activation='softmax')(H) if self.n_classes else Dense(1)(H)
        return Model(X, y)

    def train(self, X, Y, n_iters, X_test=None, Y_test=None, batch_size=32, verbose=True):
        weights = class_weight.compute_class_weight('balanced', np.unique(Y), Y)

        # Check params
        test = X_test is not None and Y_test is not None
        if self.n_classes and (len(Y.shape) == 1 or Y.shape[2] == 1):
            Y = to_categorical(Y, num_classes=self.n_classes)
            if test:
                Y_test = to_categorical(Y_test, num_classes=self.n_classes)
        N = X.shape[0]

        metric_str = 'acc' if self.n_classes else 'mse'
        history = {
            'loss':                 np.zeros((n_iters, 3)),  # pred, base, sele
            f'{metric_str}':        np.zeros((n_iters, 2)),  # pred, base
            f'{metric_str}-test':   np.zeros((n_iters, 2)),  # pred, base
            'S':                    np.zeros((n_iters, batch_size, self.n_features)),
        }
        for it in tqdm(range(n_iters)):
            idx = np.random.randint(N, size=batch_size)
            S = np.nan_to_num(self.selector.predict(X[idx])) # selector probs
            s = np.random.binomial(1, S, size=(batch_size, self.n_features)) # sample from S

            Y_base = self.baseline.predict(X[idx])
            #if it % 20 == 0:
            base_loss = self.baseline.train_on_batch(X[idx], Y[idx], class_weight=weights)

            Y_pred = self.predictor.predict([X[idx], s])
            pred_loss = self.predictor.train_on_batch([X[idx], s], Y[idx], class_weight=weights)
            #pred_loss = self.predictor.train_on_batch([X[idx], s], Y_base)

            Ys = np.concatenate([s, Y_pred, Y_base, Y[idx].reshape(batch_size,-1)], axis=1)
            sele_loss = self.selector.train_on_batch(X[idx], Ys)

            history['loss'][it] = [pred_loss, base_loss, sele_loss]
            history['S'][it] = S
            if self.n_classes:
                Y_pred = Y_pred > 0.5
                Y_base = Y_base > 0.5
                metric = np.array([
                    np.sum(Y_pred[:,0]==Y[idx,0]),
                    np.sum(Y_base[:,0]==Y[idx,0])]) / batch_size
                history[metric_str][it] = metric
                if test:
                    Y_pred = self.predict(X_test) > 0.5
                    Y_base = self.baseline.predict(X_test) > 0.5
                    metric_test = np.array([
                        np.sum(Y_pred[:,0]==Y_test[:,0]),
                        np.sum(Y_base[:,0]==Y_test[:,0])]) / X_test.shape[0]
                    history[f'{metric_str}-test'][it] = metric_test
            else:
                metric = np.array([
                    np.mean(np.square(Y_pred-Y)),
                    np.mean(np.square(Y_base-Y))])
                history[metric_str][it] = metric
                if test:
                    Y_pred = self.predict(X_test)
                    Y_base = self.baseline.predict(X_test)
                    metric_test = np.array([
                        np.mean(np.square(Y_pred-Y_test)),
                        np.mean(np.square(Y_base-Y_test))])
                    history[f'{metric_str}-test'][it] = metric_test

            feat_prob_mean = np.mean(S, axis=0)
            if verbose and (it+1) % (n_iters//10) == 0:
                print(f'#{it}:\tsele loss\t{sele_loss:.4f}\n\tpred loss\t{pred_loss:.4f}\n\tbase loss\t{base_loss:.4f}')
                print(f'\tpred {metric_str}\t{metric[0]:.4f}\tbase {metric_str}\t{metric[1]:.4f}')
                if test:
                    print(f'\tpred {metric_str} (test)\t{metric_test[0]:.4f}\tbase {metric_str} (test)\t{metric_test[1]:.4f}')
                feat_prob_str = np.array2string(feat_prob_mean, formatter={'float_kind': '{0:.2f}'.format})
                print(f'features\n{feat_prob_str}')
                if self.relevant_features is not None:
                    print(f'true features\n{np.array2string(self.relevant_features)}')

        return history

    def predict(self, X, threshold=0.5, use_baseline=False):
        if use_baseline:
            Y = self.baseline.predict(X)
        else:
            s = self.predict_features(X, threshold)
            Y = self.predictor.predict([X, s])
        return Y

    def predict_features(self, X, threshold=0.5):
        S = self.selector.predict(X)
        if threshold == None:
            return S
        s = S > threshold
        return s


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    np.set_printoptions(formatter={"float_kind": lambda x: f"{x:.4f}"})
    X, t, Y = synthetic_data(n_features=30, models=[1])
    N, n_features = X.shape
    n_treatments = Y.shape[1]
    invase = INVASE(n_features)
    Y += np.random.randn(N, 1) * 0.1

    N_train = int(0.8 * N)
    X_train = X[N_train:]
    Y_train = Y[N_train:]
    X_test = X[:N_train]
    Y_test = Y[:N_train]
    invase.train(X_train, Y_train, [10000, 10000], X_test, Y_test)
    Y_pred = invase.predict(X_test)
    ss = invase.predict_features(X_test)
    X_str, Y_str, t_str, Y_pred_str, ss_str = map(np.array2string, [X, Y, t, Y_pred, ss.astype(int)])
    print('\n'.join(['X', X_str, 'Y', Y_str, 't', t_str, 'Y_pred', Y_pred_str, 'ss', ss_str]))
