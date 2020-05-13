from keras.layers import Dense, Input, Multiply, Concatenate
from keras.models import Model
from keras.utils import to_categorical
import keras.backend as K
import numpy as np
from tqdm import tqdm


eps = 1e-8

default_hyperparams = {
    'h_layers_pred':    2,
    'h_dim_pred':       lambda x: 100,  # noqa 272
    'h_layers_base':    2,
    'h_dim_base':       lambda x: 100,  # noqa 272
    'h_layers_sel':     1,
    'h_dim_sel':        lambda x: 2*x,  # noqa 272
    'lam':              0.1,
    'optimizer':        'adam'
}


class Invase:
    def __init__(self, n_features, n_classes, hyperparams=default_hyperparams, verbose=True):
        self.n_features = n_features
        self.n_classes = n_classes  # 0 = regression, 2+ = classification
        pred_base_loss = 'categorical_crossentropy' if self.n_classes else 'mse'
        optimizer = hyperparams['optimizer']

        # Build and compile baseline net
        h_layers_base = hyperparams['h_layers_base']
        h_dim_base = hyperparams['h_dim_base'](self.n_features)
        self.baseline = self.__build_baseline(h_layers_base, h_dim_base)
        self.baseline.compile(loss=pred_base_loss, optimizer=optimizer)

        # Build and compile predictor net
        h_layers_pred = hyperparams['h_layers_pred']
        h_dim_pred = hyperparams['h_dim_pred'](self.n_features)
        self.predictor = self.__build_predictor(h_layers_pred, h_dim_pred)
        self.predictor.compile(loss=pred_base_loss, optimizer=optimizer)

        # Build and compile selector net, with custom loss function
        h_layers_sel = hyperparams['h_layers_sel']
        h_dim_sel = hyperparams['h_dim_sel'](self.n_features)
        self.selector = self.__build_selector(h_layers_sel, h_dim_sel)
        lam = hyperparams['lam']

        def selector_loss_fn(sY, S):
            # Extract s (sampled selection vector) and Y_pred, Y_base, Y_true (factual) from sY
            # Have S (probability vector from selector net)
            s = sY[:, :self.n_features]
            if self.n_classes:
                sep = [self.n_features + self.n_classes * i for i in range(3)]
                Y_pred = sY[:, sep[0]:sep[1]]
                Y_base = sY[:, sep[1]:sep[2]]
                Y_true = sY[:, sep[2]:]

                L_pred = -K.sum(Y_true * K.log(Y_pred+eps), axis=-1)
                L_base = -K.sum(Y_true * K.log(Y_base+eps), axis=-1)
            else:
                Y_pred = sY[:, self.n_features]
                Y_base = sY[:, self.n_features + 1]
                Y_true = sY[:, self.n_features + 2]

                L_pred = K.square(Y_true - Y_pred)
                L_base = K.square(Y_true - Y_base)

            # Reward predictors that have low loss and are close to the baseline
            imitation_loss = L_pred - L_base

            # Policy gradient (actor-critic)
            loss = imitation_loss * K.sum(s*K.log(S+eps) + (1-s)*K.log(1-S+eps), axis=-1)

            # Penalise complexity
            complexity = lam * K.mean(S, axis=-1)

            return K.mean(loss) + complexity

        self.selector.compile(loss=selector_loss_fn, optimizer=optimizer)

    def __build_baseline(self, h_layers, h_dim):
        X = Input((self.n_features,))
        H = X
        for _ in range(h_layers):
            H = Dense(h_dim, activation='relu')(X)
        y = Dense(self.n_classes, activation='softmax')(H) if self.n_classes else Dense(1)(H)
        return Model(X, y)

    def __build_predictor(self, h_layers, h_dim):
        X = Input((self.n_features,))  # not suppressed
        s = Input((self.n_features,))  # binary selection vector
        H = Multiply()([X, s])         # suppressed
        H = Concatenate()([H, s])      # concatenated for distinguishing suppressed features from 0-valued features
        for _ in range(h_layers):
            H = Dense(h_dim, activation='relu')(H)
        y = Dense(self.n_classes, activation='softmax')(H) if self.n_classes else Dense(1)(H)
        return Model([X, s], y)

    def __build_selector(self, h_layers, h_dim):
        X = Input((self.n_features,))
        H = X
        for _ in range(h_layers):
            H = Dense(h_dim, activation='relu')(H)
        S = Dense(self.n_features, activation='sigmoid')(H)  # selection probability vector
        return Model(X, S)

    def train(self, X, Y, n_iters, X_test=None, Y_test=None, S_true=None,
              batch_size=1024, verbose=True, save_history=True):
        # Check array dims. If Y is a class vector (e.g. [0, 1, 2, 0]) then change it to one-hot encoding
        test = X_test is not None and Y_test is not None
        if self.n_classes and (len(Y.shape) == 1 or Y.shape[1] == 1):
            Y = to_categorical(Y, num_classes=self.n_classes)
            if test:
                Y_test = to_categorical(Y_test, num_classes=self.n_classes)
        n = X.shape[0]

        # Prep for metric outputs during and after training
        # TODO parameterise v_iters and h_iters
        v_iters = n_iters // 10 if verbose else 0
        h_iters = 10 if save_history else 0
        metric_str = 'acc' if self.n_classes else 'mse'
        history = {
            'loss':                 np.zeros((n_iters//h_iters, 3)),  # pred, base, sele
            f'{metric_str}':        np.zeros((n_iters//h_iters, 2)),  # pred, base
            f'{metric_str}-test':   np.zeros((n_iters//h_iters, 2)),  # pred, base
            's':                    np.zeros((n_iters//h_iters, batch_size, self.n_features)),
        } if h_iters else None

        # Train
        for it in tqdm(range(1, 1+n_iters)):
            idx = np.random.randint(n, size=batch_size)                       # random batch
            S = np.nan_to_num(self.selector.predict(X[idx]))                  # selector probs
            s = np.random.binomial(1, S, size=(batch_size, self.n_features))  # sample from S

            # Train baseline and predictor
            if self.n_classes:
                base_loss = self.baseline.train_on_batch(X[idx], Y[idx])
                pred_loss = self.predictor.train_on_batch([X[idx], s], Y[idx])
            else:
                base_loss = self.baseline.train_on_batch(X[idx], Y[idx])
                pred_loss = self.predictor.train_on_batch([X[idx], s], Y[idx])

            # Train selector
            Y_pred = self.predictor.predict([X[idx], s])
            Y_base = self.baseline.predict(X[idx])
            sY = np.concatenate([s, Y_pred, Y_base, Y[idx].reshape(batch_size, -1)], axis=-1)
            sele_loss = self.selector.train_on_batch(X[idx], sY)

            # Record/output metrics at appropriate intervals
            if (h_iters and it % h_iters == 0) or (v_iters and it % v_iters == 0):
                # Calculate
                if self.n_classes:
                    Y_pred = np.argmax(Y_pred, axis=-1)
                    Y_base = np.argmax(Y_base, axis=-1)
                    metric = np.array([
                        np.sum(Y_pred == np.argmax(Y[idx], axis=-1)),
                        np.sum(Y_base == np.argmax(Y[idx], axis=-1))]) / batch_size
                    if test:
                        Y_pred = self.predict(X_test) > 0.5
                        Y_base = self.baseline.predict(X_test) > 0.5
                        metric_test = np.array([
                            np.sum(Y_pred[:, 0] == Y_test[:, 0]),
                            np.sum(Y_base[:, 0] == Y_test[:, 0])]) / X_test.shape[0]
                else:
                    metric = np.array([
                        np.mean(np.square(Y_pred - Y[idx])),
                        np.mean(np.square(Y_base - Y[idx]))])
                    if test:
                        Y_pred = self.predict(X_test)
                        Y_base = self.baseline.predict(X_test)
                        metric_test = np.array([
                            np.mean(np.square(Y_pred - Y_test)),
                            np.mean(np.square(Y_base - Y_test))])

                # Save history
                if save_history and it % h_iters == 0:
                    h_it = it // h_iters - 1
                    history['loss'][h_it] = [pred_loss, base_loss, sele_loss]
                    history['s'][h_it] = S
                    history[metric_str][h_it] = metric
                    if test:
                        history[f'{metric_str}-test'][h_it] = metric_test

                # Output
                if verbose and it % v_iters == 0:
                    print(f'#{it}:' +
                          f'\tsele loss\t{sele_loss:.4f}\n' +
                          f'\tpred loss\t{pred_loss:.4f}\n' +
                          f'\tbase loss\t{base_loss:.4f}\n' +
                          f'\tpred {metric_str}\t{metric[0]:.4f}' +
                          f'\tbase {metric_str}\t{metric[1]:.4f}')
                    if test:
                        print(f'\tpred {metric_str} (test)\t{metric_test[0]:.4f}' +
                              f'\tbase {metric_str} (test)\t{metric_test[1]:.4f}')
                    S_mean = np.mean(S, axis=0)
                    S_mean_str = np.array2string(S_mean, formatter={'float_kind': '{0:.2f}'.format})
                    print(f'features\n{S_mean_str}')
                    if S_true is not None:
                        print(f'true features\n{np.array2string(S_true)}')

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
        if threshold is None:
            return S
        s = S > threshold
        return s
