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

class INVASE:
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

    def train(self, X, Y, n_iters, X_test=None, Y_test=None, batch_size=1024, verbose=True, save_history=True):
        # Check params
        test = X_test is not None and Y_test is not None
        if self.n_classes:# and (len(Y.shape) == 1 or Y.shape[2] == 1):
            #weights = class_weight.compute_class_weight('balanced', np.unique(Y), Y)
            Y = to_categorical(Y, num_classes=self.n_classes)
            if test:
                Y_test = to_categorical(Y_test, num_classes=self.n_classes)
        N = X.shape[0]
        v_iters = n_iters // 10 if verbose else 0
        h_iters = 100 if save_history else 0

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

            # save history
            if h_iters and it % h_iters == 0:
                h_it = (it-1) // h_iters - 1
                history['loss'][h_it] = [pred_loss, base_loss, sele_loss]
                history['S'][h_it] = S
                if self.n_classes:
                    Y_pred = np.argmax(Y_pred, axis=1)
                    Y_base = np.argmax(Y_base, axis=1)
                    metric = np.array([
                        np.sum(Y_pred==np.argmax(Y[idx], axis=1)),
                        np.sum(Y_base==np.argmax(Y[idx], axis=1))]) / batch_size
                    history[metric_str][h_it] = metric
                    if test:
                        Y_pred = self.predict(X_test) > 0.5
                        Y_base = self.baseline.predict(X_test) > 0.5
                        metric_test = np.array([
                            np.sum(Y_pred[:,0]==Y_test[:,0]),
                            np.sum(Y_base[:,0]==Y_test[:,0])]) / X_test.shape[0]
                        history[f'{metric_str}-test'][h_it] = metric_test
                else:
                    metric = np.array([
                        np.mean(np.square(Y_pred-Y[idx])),
                        np.mean(np.square(Y_base-Y[idx]))])
                    history[metric_str][h_it] = metric
                    if test:
                        Y_pred = self.predict(X_test)
                        Y_base = self.baseline.predict(X_test)
                        metric_test = np.array([
                            np.mean(np.square(Y_pred-Y_test)),
                            np.mean(np.square(Y_base-Y_test))])
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


if __name__ == '__main__':
    from synthetic_data import synthetic_data
    from utils.metrics import tpr_fdr, roc

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    np.set_printoptions(formatter={"float_kind": lambda x: f"{x:.4f}"})

    for i in range(1, 7):
        X, t, Y, S = synthetic_data(N=20000, n_features=11, models=[i], corr=False)
        #corr = np.corrcoef(X.T)
        #plt.figure()
        #sns.heatmap(corr, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 220, n=200), square=True)
        #plt.show()
        N, n_features = X.shape
        n_treatments = Y.shape[1]
        #Y += np.random.randn(N, 1) * 0.1

        N_train = int(.5 * N)
        X_train = X[N_train:]
        Y_train = Y[N_train:]
        X_test = X[:N_train]
        Y_test = Y[:N_train]
        S_test = S[0, :N_train]

        invase = INVASE(n_features, n_classes=2, lam=.1)
        invase.train(X_train, Y_train, 10000, X_test, Y_test, batch_size=1024)
        Y_pred = invase.predict(X_test)
        Y_base = invase.predict(X_test, use_baseline=True)
        S_pred = invase.predict_features(X_test)
        #X_str, Y_str, t_str, Y_pred_str, S_pred_str = map(np.array2string, [X, Y, t, Y_pred, S_pred.astype(int)])
        #print('\n'.join(['X', X_str, 'Y', Y_str, 't', t_str, 'Y_pred', Y_pred_str, 'S_pred', S_pred_str]))
        tpr, fdr = tpr_fdr(S_test, S_pred)
        print(f'TPR: {tpr*100:.1f}%\nFDR: {fdr*100:.1f}%')
        s_pred = invase.predict_features(X_test, threshold=None)
        r = roc(Y_test, Y_pred[:,1])
        plt.plot(r[:,0], r[:,1])
        r = roc(Y_test, Y_base[:,1])
        plt.plot(r[:,0], r[:,1])
        plt.show()
        #sns.heatmap(s_pred[:100].T, center=.5, vmin=0, vmax=1, cmap='gray', square=True, cbar=False, linewidth=.5)
        #plt.show()
