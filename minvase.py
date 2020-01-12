import numpy as np
import keras.backend as K
from tqdm import tqdm

from invase import INVASE


eps = 1e-8

class MINVASE(INVASE):
    def __init__(self, n_features, n_specialisms, gamma=0.05, relevant_features=None):
        assert n_specialisms > 1
        self.n_features = n_features
        self.n_specialisms = n_specialisms
        self.gamma = gamma
        self.relevant_features = relevant_features

        self.selectors = [self.build_selector() for i in range(self.n_specialisms)]
        self.predictors = [self.build_predictor() for i in range(self.n_specialisms)]
        self.baseline = self.build_baseline()

        def selector_loss_fn(Ys, S_pred):
            # Extract s, ss and y_pred/y_base/y_fact
            n_selections = self.n_features * self.n_specialisms
            s = Ys[:, :self.n_features]
            others = [Ys[:, i*self.n_features:(i+1)*self.n_features] for i in range(1, self.n_specialisms)]
            Y_pred = Ys[:, n_selections]
            Y_base = Ys[:, n_selections + 1]
            Y_true = Ys[:, n_selections + 2]

            # Reward regressions that are close to the baseline
            mse_pred = K.sum(K.square(Y_true - Y_pred))
            mse_base = K.sum(K.square(Y_true - Y_base))
            imitation_reward = mse_pred - mse_base

            # policy gradient
            loss1 = imitation_reward * K.sum(s*K.log(S_pred+eps) + (1-s)*K.log(1-S_pred+eps), axis=1)

            # Penalise complexity
            complexity_loss = self.gamma * K.mean(S_pred, axis=1)

            # Penalise similarity
            similarity_loss = -0.01 * K.sum([s*(K.log(s+eps)-K.log(other+eps)) for other in others])

            return K.mean(-loss1) + complexity_loss + similarity_loss

        for (selector, predictor) in zip(self.selectors, self.predictors):
            selector.compile(loss=selector_loss_fn, optimizer='adam')
            predictor.compile(loss='mse', optimizer='adam')
        self.baseline.compile(loss='mse', optimizer='adam')

    def train(self, X, Y, n_iters, X_val=None, Y_val=None, batch_size=32):
        # Check params
        val = (X_val, Y_val) != (None, None)
        if type(n_iters) == int:
            n_iters = [n_iters, n_iters]
        N = X.shape[0]

        for it in tqdm(range(n_iters[1])):
            idx = np.random.randint(N, size=batch_size)

            base_loss = self.baseline.train_on_batch(X[idx], Y[idx])

            Ss = [np.nan_to_num(selector.predict(X[idx])) for selector in self.selectors] # selector probs
            ss = [np.random.binomial(1, S, size=(batch_size, self.n_features)) for S in Ss] # sample from S
            pred_loss = np.zeros(self.n_specialisms)
            sele_loss = np.zeros(self.n_specialisms)
            for i in range(self.n_specialisms):
                s = ss[i]
                others = np.concatenate([ss[j] for j in range(self.n_specialisms) if j!=i], axis=1)

                Y_pred = self.predictors[i].predict([X[idx], s]) 
                pred_loss[i] = self.predictors[i].train_on_batch([X[idx], s], Y[idx])

                Y_base = self.baseline.predict(X[idx])

                Ys = np.concatenate([s, others, Y_pred, Y_base, Y[idx].reshape(-1,1)], axis=1)
                sele_loss[i] = self.selectors[i].train_on_batch(X[idx], Ys) 

            if (it+1) % (n_iters[1]//10) == 0:
                if val:
                    Y_pred, _ = self.predict(X_val)
                    pred_mse = np.mean(np.square(Y_val - Y_pred))
                    Y_base = self.baseline.predict(X_val)
                    base_mse = np.mean(np.square(Y_val - Y_base))
                    pred_loss_str += f'\tval mse {pred_mse:.4f}'
                    base_loss_str += f'\tval mse {base_mse:.4f}'
                print(f'#{it}:\tsele loss {sele_loss}\n\tpred loss {pred_loss}\n\tbase loss {base_loss}')
                for S in Ss:
                    feat_prob = np.mean(S, axis=0)
                    print(f'features\n{np.array2string(feat_prob)}')
                if self.relevant_features is not None:
                    print(f'true features\n{np.array2string(self.relevant_features)}')

    def predict(self, X, threshold=0.5):
        ss = self.predict_features(X, threshold)
        Y = np.mean([predictor.predict([X, s]) for (predictor, s) in zip(self.predictors, ss)])
        return Y

    def predict_features(self, X, threshold=0.5):
        Ss = [selector.predict(X) for selector in self.selectors]
        if threshold == None:
            return Ss
        ss = S > threshold
        return s
