from keras.layers import Input, Dense
from keras.models import Model
from keras import losses
from keras.utils import to_categorical
from tqdm import tqdm
import numpy as np

from synthetic_data import synthetic_data


class Derp:
    def __init__(self, n_features, n_treatments):
        self.n_features = n_features
        self.n_treatments = n_treatments

        X = Input((n_features,))
        hidden = Dense(16, activation='relu')(X)
        hidden = Dense(16, activation='relu')(hidden)
        Y = [Dense(1, activation='sigmoid')(hidden) for t in range(n_treatments)]
        self.models = [Model(X, y) for y in Y]
        for model in self.models:
            model.compile(loss=losses.mean_squared_error, optimizer='adam')
        model.summary()
    
    def train(self, n_iters, X_train, T_train, Y_train, X_val, Y_val):
        ts = np.argmax(T_train, axis=1)
        idxs = np.array([np.argwhere(ts==t).flatten() for t in range(self.n_treatments)])
        losses = [0] * self.n_treatments
        for it in tqdm(range(n_iters)):
            for t in range(self.n_treatments):
                idx = np.random.choice(idxs[t].flatten(), size=16, replace=False)
                losses[t] = self.models[t].train_on_batch(X_train[idx], Y_train[idx])

            if it % (n_iters//10) == 0:
                Y_pred = self.predict(X_val)
                val_loss = np.mean(np.square(Y_pred - Y_val))
                str_losses = ' '.join(f'{loss:4.2f}' for loss in losses)
                print(f'Iter {it:4d} losses [{str_losses}] val_loss {val_loss}')

    def predict(self, X):
        return np.array([model.predict(X) for model in self.models]).T[0]


if __name__ == '__main__':
    np.set_printoptions(formatter={'float_kind': lambda x: f'{x:.4f}'})
    
    N = 10000
    N_train = int(0.8 * N)
    n_features = 10
    n_treatments = 2
    X, t, Y = synthetic_data(N=N, n_features=n_features, n_treatments=n_treatments)
    T = to_categorical(t, num_classes=n_treatments)
    Y_fact = np.choose(t, Y.T)

    X_train = X[:N_train]
    T_train = T[:N_train]
    Y_train = Y_fact[:N_train]
    X_test = X[N_train:]
    Y_test = Y[N_train:]
        
    derp = Derp(n_features, n_treatments)
    derp.train(10000, X_train, T_train, Y_train, X_test, Y_test)

    print('Y_pred')
    print(derp.predict(X_test))
    print('Y_test')
    print(Y_test)
