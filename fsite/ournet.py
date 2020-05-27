from keras.models import Sequential, Model
from keras.layers import Dense, Lambda, Input, Concatenate
from keras.optimizers import Adam
from keras.regularizers import l2
import keras.backend as K
from utils.loss import mmd2, weighted_factual
import numpy as np
from utils.metrics import PEHE
from utils.utils import make_Y

# can put this into a hyperparam dict later if necessary
activation = None
sigma = 1
alpha = 1
batch_size = 128
rep_dim = 100
lr = 1e-3
reg = 1e-2


class OurNet:
    def __init__(self, n_features):
        self.nt, self.nc, self.ny = n_features
        assert self.nt + self.nc > 0 and self.nc + self.ny > 0

        # Build nets
        rep = self.build_rep(self.nc, 100, 2, rep_dim)
        h0 = self.build_pred(self.ny + rep_dim, 100, 2, 0)
        h1 = self.build_pred(self.ny + rep_dim, 100, 2, 1)
        prop = self.build_prop(self.nt + self.nc, 100, 2)

        # Build graph, checking for zeros
        XT = Input((self.nt,))
        XC = Input((self.nc,))
        XY = Input((self.ny,))
        if self.nt and self.nc:
            XTC = Concatenate()([XT, XC])
            T = prop(XTC)
            self.prop = Model([XT, XC], T)
        elif self.nt:
            pass  # no need for weights
        elif self.nc:
            T = prop(XC)
            self.prop = Model(XC, T)
        if self.nc and self.ny:
            R = rep(XC)
            RXY = Concatenate()([R, XY])
            Y0 = h0(RXY)
            Y1 = h1(RXY)
            self.h0 = Model([XC, XY], Y0)
            self.h1 = Model([XC, XY], Y1)
            self.cfr = Model([XC, XY], [R, Y0, Y1])
        elif self.nc:
            R = rep(XC)
            Y0 = h0(R)
            Y1 = h1(R)
            self.h0 = Model(XC, Y0)
            self.h1 = Model(XC, Y1)
            self.cfr = Model(XC, [R, Y0, Y1])
        elif self.ny:
            Y0 = h0(XY)
            Y1 = h1(XY)
            self.h0 = Model(XY, Y0)
            self.h1 = Model(XY, Y1)
            pass  # no cfr model in this case

        # Compile models prop, h0, h1 and cfr
        self.h0.compile(Adam(lr), 'mse')
        self.h1.compile(Adam(lr), 'mse')
        if self.nc:
            self.h0.trainable = False
            self.h1.trainable = False
            self.prop.compile(Adam(lr), 'binary_crossentropy')
            self.cfr.compile(
                Adam(lr),
                loss={
                    'rep': lambda T, R_pred: mmd2(R_pred, T, 1),
                    'h0': lambda TYw0, Y0_pred: weighted_factual(TYw0, Y0_pred, 0),
                    'h1': lambda TYw1, Y1_pred: weighted_factual(TYw1, Y1_pred, 1)
                }, loss_weights={
                    'rep': alpha,
                    'h0': 1,
                    'h1': 1
                })

    def train(self, X, T, Yf, n_iters, batch_size=batch_size, Ycf=None,
              val_data=None, test_data=None, verbose=False, save_history=False):
        # Preprocess X: disentangle
        n = X.shape[0]
        XT = X[:, :self.nt]
        XC = X[:, self.nt:self.nt+self.nc]
        XY = X[:, self.nt+self.nc:]

        # Preprocess T: get marginal propensity score
        pt = np.mean(T)

        # Preprocess Y: construct full outcome matrix Y
        if Ycf is not None:
            Y = make_Y(T, Yf, Ycf)
        val = val_data is not None
        if val:
            assert len(val_data) == 4
            X_val, T_val, Yf_val, Ycf_val = val_data
            Y_val = make_Y(T_val, Yf_val, Ycf_val)
        test = test_data is not None
        if test:
            assert len(test_data) == 4
            X_test, T_test, Yf_test, Ycf_test = test_data
            Y_test = make_Y(T_test, Yf_test, Ycf_test)

        # Prep for training loop
        if verbose:
            v_iters = 100  # n_iters // 10
        if save_history:
            n_savepoints = 100
            history = {m: np.zeros(n_savepoints) for m in 'obj h0 h1 imb'.split()}
            if Ycf is not None:
                history['pehe'] = np.zeros(n_savepoints)
            if val:
                history['pehe_val'] = np.zeros(n_savepoints)
            if test:
                history['pehe_test'] = np.zeros(n_savepoints)
            h_iters = n_iters // n_savepoints

        # TRAINING LOOP
        for it in range(n_iters):
            # Random (imbalanced) batch
            idx = np.random.choice(n, size=batch_size)
            XTb, XCb, XYb, Tb, Yfb = map(lambda x: x[idx], [XT, XC, XY, T, Yf])

            # Train cfr using weighted losses from prop, then train prop
            if self.nc:
                if self.nt:
                    Tb_pred = self.prop.predict([XTb, XCb])
                else:
                    Tb_pred = self.prop.predict(XCb)
                w0b = 1 + (1-pt)/pt * Tb_pred/(1-Tb_pred)
                w1b = 1 + pt/(1-pt) * (1-Tb_pred)/Tb_pred
                TYw0b = np.vstack([Tb, Yfb, w0b.flatten()]).T
                TYw1b = np.vstack([Tb, Yfb, w1b.flatten()]).T
                if self.ny:
                    cfr_loss = self.cfr.train_on_batch([XCb, XYb], [Tb, TYw0b, TYw1b])
                else:
                    cfr_loss = self.cfr.train_on_batch(XCb, [Tb, TYw0b, TYw1b])
                if self.nt:
                    prop_loss = self.prop.train_on_batch([XTb, XCb], Tb)
                else:
                    prop_loss = self.prop.train_on_batch(XCb, Tb)

            # Train h0 and h1
            if self.nc and self.ny:
                h0_loss = self.h0.train_on_batch([XCb[Tb == 0], XYb[Tb == 0]], Yfb[Tb == 0])
                h1_loss = self.h1.train_on_batch([XCb[Tb == 1], XYb[Tb == 1]], Yfb[Tb == 1])
            elif self.nc:
                h0_loss = self.h0.train_on_batch(XCb[Tb == 0], Yfb[Tb == 0])
                h1_loss = self.h1.train_on_batch(XCb[Tb == 1], Yfb[Tb == 1])
            elif self.ny:
                h0_loss = self.h0.train_on_batch(XYb[Tb == 0], Yfb[Tb == 0])
                h1_loss = self.h1.train_on_batch(XYb[Tb == 1], Yfb[Tb == 1])

            # Report training progress
            do_history = save_history and ((it+1) % h_iters == 0)
            do_verbose = verbose and ((it+1) % v_iters == 0) or (it == 0) or ((it+1) == n_iters)
            if not do_history and not do_verbose:
                continue

            if self.nc:
                obj, imb, h0_cfr, h1_cfr = cfr_loss
                # obj = imb = cfr_loss
                if alpha != 0:
                    imb /= alpha
            else:
                obj = imb = 0
            if Ycf is not None:
                Y_pred = self.predict(X)
                pehe = PEHE(Y, Y_pred)
            if val:
                Y_val_pred = self.predict(X_val)
                pehe_val = PEHE(Y_val, Y_val_pred)
            if test:
                Y_test_pred = self.predict(X_test)
                pehe_test = PEHE(Y_test, Y_test_pred)

            if do_history:
                idx = it//h_iters
                history['obj'][idx] = obj
                history['h0'][idx] = h0_loss
                history['h1'][idx] = h1_loss
                history['imb'][idx] = imb
                if Ycf is not None:
                    history['pehe'][idx] = pehe
                if val:
                    history['pehe_val'][idx] = pehe_val
                if test:
                    history['pehe_test'][idx] = pehe_test

            if do_verbose:
                table_row = '\t'.join(f'{a} {b:.4f}' for a, b in zip('obj h0 h1 imb'.split(),
                                                                     [obj, h0_loss, h1_loss, imb]))
                print(f'{it+1:<4}\t{table_row}', end='')
                if Ycf is not None:
                    print(f'\tpehe {pehe:.4f}', end='')
                if val:
                    print(f'\tval {pehe_val:.4f}', end='')
                if test:
                    print(f'\ttest {pehe_test:.4f}', end='')
                print()
                # print(f'\t{h0_cfr:.4f}\t{h1_cfr:.4f}')

        if save_history:
            return history

    def predict(self, X):
        XC = X[:, self.nt:self.nt+self.nc]
        XY = X[:, self.nt+self.nc:]
        if self.nc and self.ny:
            _, Y0_pred, Y1_pred = self.cfr.predict([XC, XY])
        elif self.nc:
            _, Y0_pred, Y1_pred = self.cfr.predict(XC)
        elif self.ny:
            Y0_pred = self.h0.predict(XY)
            Y1_pred = self.h1.predict(XY)
        return np.hstack([Y0_pred, Y1_pred])

    def build_rep(self, in_dim, h_dim, h_layers, out_dim):
        rep = Sequential(name='rep')
        for _ in range(h_layers):
            rep.add(Dense(h_dim, activation=activation, kernel_regularizer=l2(reg)))
        rep.add(Dense(out_dim, kernel_regularizer=l2(reg)))
        rep.add(Lambda(lambda x: K.l2_normalize(1000*x, axis=1)))
        return rep

    def build_pred(self, in_dim, h_dim, h_layers, t):
        pred = Sequential(name=f'h{t}')
        for _ in range(h_layers):
            pred.add(Dense(h_dim, activation=activation, kernel_regularizer=l2(reg)))
        pred.add(Dense(1, kernel_regularizer=l2(reg)))
        return pred

    def build_prop(self, in_dim, h_dim, h_layers):
        prop = Sequential()
        for _ in range(h_layers):
            prop.add(Dense(h_dim, activation=activation, kernel_regularizer=l2(reg)))
        prop.add(Dense(1, activation='sigmoid', kernel_regularizer=l2(reg)))
        return prop
