import os
from keras.layers import Dense, Input, Concatenate, Multiply
from keras.initializers import Constant
from keras.models import Model
import keras.backend as K
from keras.utils import to_categorical
import numpy as np
from tqdm import tqdm
from sklearn.utils import class_weight
import seaborn as sns
import matplotlib.pyplot as plt


eps = 1e-8

default_params = {
    'h_dim_pred':   lambda x: 2*x,
    'h_dim_base':   lambda x: 2*x,
    'h_layers':     2,
}

class Invase:
    def __init__(self, n_features, n_classes, lam=0.05, relevant_features=None, verbose=True):
        self.n_features = n_features
        self.n_classes = n_classes  # 0 = regression, 2+ = classification
        self.lam = lam
        self.relevant_features = relevant_features
        self.s_h_layers = 2
        self.s_h_dim = 2 * self.n_features
        self.h_layers = 2
        self.h_dim = 2* self.n_features

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

                L_pred = K.sum(Y_true * K.log(Y_pred+eps), axis=-1)
                L_base = K.sum(Y_true * K.log(Y_base+eps), axis=-1)
            else:
                Y_pred = Ys[:, self.n_features]
                Y_base = Ys[:, self.n_features + 1]
                Y_true = Ys[:, self.n_features + 2]

                L_pred = -(K.square(Y_true - Y_pred))
                L_base = -(K.square(Y_true - Y_base))
            # Reward regressions that are close to the baseline
            imitation_loss = L_pred - L_base

            # policy gradient
            loss1 = imitation_loss * K.sum(s*K.log(S_pred+eps) + (1-s)*K.log(1-S_pred+eps), axis=-1)
            # Penalise complexity
            complexity_loss = self.lam * K.mean(S_pred, axis=-1)
            return K.mean(-loss1) + complexity_loss

        self.selector.compile(loss=selector_loss_fn, optimizer='adam')
        loss = 'categorical_crossentropy' if self.n_classes else 'mse'
        self.predictor.compile(loss=loss, optimizer='adam')
        self.baseline.compile(loss=loss, optimizer='adam')

    def build_selector(self):
        X = Input((self.n_features,))
        H = X
        for _ in range(self.s_h_layers):
            H = Dense(self.s_h_dim, activation='relu')(X)
        S = Dense(self.n_features, activation='sigmoid')(H)
        return Model(X, S)

    def build_predictor(self):
        X = Input((self.n_features,))  # not suppressed
        s = Input((self.n_features,))
        H = Multiply()([X, s])         # suppressed
        for _ in range(self.h_layers):
            H = Dense(self.h_dim, activation='relu')(H)
        y = Dense(self.n_classes, activation='softmax')(H) if self.n_classes else Dense(1)(H)
        return Model([X, s], y)

    def build_baseline(self):
        X = Input((self.n_features,))
        H = X
        for _ in range(self.h_layers):
            H = Dense(self.h_dim, activation='relu')(X)
        y = Dense(self.n_classes, activation='softmax')(H) if self.n_classes else Dense(1)(H)
        return Model(X, y)

    def train(self, X, Y, n_iters, X_test=None, Y_test=None, batch_size=1024, verbose=True, save_history=True, h_iters=10):
        # Check params
        test = X_test is not None and Y_test is not None
        if self.n_classes:# and (len(Y.shape) == 1 or Y.shape[2] == 1):
            #weights = class_weight.compute_class_weight('balanced', np.unique(Y), Y)
            Y = to_categorical(Y, num_classes=self.n_classes)
            if test:
                Y_test = to_categorical(Y_test, num_classes=self.n_classes)
        N = X.shape[0]
        v_iters = n_iters // 10 if verbose else 0

        metric_str = 'acc' if self.n_classes else 'mse'
        history = {
            'loss':                 np.zeros((n_iters//h_iters, 3)),  # pred, base, sele
            f'{metric_str}':        np.zeros((n_iters//h_iters, 2)),  # pred, base
            f'{metric_str}-test':   np.zeros((n_iters//h_iters, 2)),  # pred, base
            'S':                    np.zeros((n_iters//h_iters, batch_size, self.n_features)),
        }
        for it in tqdm(range(1, 1+n_iters)):
            idx = np.random.randint(N, size=batch_size)
            S = np.nan_to_num(self.selector.predict(X[idx])) # selector probs
            s = np.random.binomial(1, S, size=(batch_size, self.n_features)) # sample from S

            if self.n_classes:
                base_loss = self.baseline.train_on_batch(X[idx], Y[idx])#, class_weight=weights)
                pred_loss = self.predictor.train_on_batch([X[idx], s], Y[idx])#, class_weight=weights)
            else:
                base_loss = self.baseline.train_on_batch(X[idx], Y[idx])
                pred_loss = self.predictor.train_on_batch([X[idx], s], Y[idx])

            Y_pred = self.predictor.predict([X[idx], s])
            Y_base = self.baseline.predict(X[idx])
            Ys = np.concatenate([s, Y_pred, Y_base, Y[idx].reshape(batch_size,-1)], axis=1)
            sele_loss = self.selector.train_on_batch(X[idx], Ys)

            if it % h_iters == 0 or it % v_iters == 0:
                if self.n_classes:
                    Y_pred = np.argmax(Y_pred, axis=1)
                    Y_base = np.argmax(Y_base, axis=1)
                    metric = np.array([
                        np.sum(Y_pred==np.argmax(Y[idx], axis=1)),
                        np.sum(Y_base==np.argmax(Y[idx], axis=1))]) / batch_size
                    if test:
                        Y_pred = self.predict(X_test) > 0.5
                        Y_base = self.baseline.predict(X_test) > 0.5
                        metric_test = np.array([
                            np.sum(Y_pred[:,0]==Y_test[:,0]),
                            np.sum(Y_base[:,0]==Y_test[:,0])]) / X_test.shape[0]
                else:
                    metric = np.array([
                        np.mean(np.square(Y_pred-Y[idx])),
                        np.mean(np.square(Y_base-Y[idx]))])
                    if test:
                        Y_pred = self.predict(X_test)
                        Y_base = self.baseline.predict(X_test)
                        metric_test = np.array([
                            np.mean(np.square(Y_pred-Y_test)),
                            np.mean(np.square(Y_base-Y_test))])

                # save history
                if save_history and it % h_iters == 0:
                    h_it = it // h_iters - 1
                    history['loss'][h_it] = [pred_loss, base_loss, sele_loss]
                    history['S'][h_it] = S
                    history[metric_str][h_it] = metric
                    if test:
                        history[f'{metric_str}-test'][h_it] = metric_test

                if verbose and it % v_iters == 0:
                    print(f'#{it}:\tsele loss\t{sele_loss:.4f}\n\tpred loss\t{pred_loss:.4f}\n\tbase loss\t{base_loss:.4f}')
                    print(f'\tpred {metric_str}\t{metric[0]:.4f}\tbase {metric_str}\t{metric[1]:.4f}')
                    if test:
                        print(f'\tpred {metric_str} (test)\t{metric_test[0]:.4f}\tbase {metric_str} (test)\t{metric_test[1]:.4f}')
                    feat_prob_mean = np.mean(S, axis=0)
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
