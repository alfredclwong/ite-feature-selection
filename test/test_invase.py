import unittest
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from fsite.invase import Invase
from data.synthetic_data import synthetic_data
from utils.metrics import tpr_fdr, auc, roc
from utils.utils import default_env


class Test_Invase(unittest.TestCase):  # not really a proper unit test but the framework is convenient
    def testAll(self):
        default_env()
        for i in range(1, 7):
            X, t, Y, S = synthetic_data(N=20000, n_features=11, models=[i], corr=True)
            N, n_features = X.shape
            # Y += np.random.randn(N, 1) * 0.1

            N_train = int(.5 * N)
            X_train = X[N_train:]
            Y_train = Y[N_train:]
            X_test = X[:N_train]
            Y_test = Y[:N_train]
            S_test = S[0, :N_train]

            invase = Invase(n_features, n_classes=2, lam=0.1)
            invase.train(X_train, Y_train, 5000, X_test, Y_test, batch_size=512, verbose=True)
            Y_pred = invase.predict(X_test)
            Y_base = invase.predict(X_test, use_baseline=True)
            S_pred = invase.predict_features(X_test)

            X_str, Y_str, t_str, Y_pred_str, S_pred_str = map(np.array2string, [X, Y, t, Y_pred, S_pred.astype(int)])
            print('\n'.join(['X', X_str, 'Y', Y_str, 't', t_str, 'Y_pred', Y_pred_str, 'S_pred', S_pred_str]))

            # Test: tpr and fdr within acceptable ranges (replicate the paper)
            tpr, fdr = tpr_fdr(S_test, S_pred)
            tpr10, fdr10 = tpr_fdr(S_test[:, :-1], S_pred[:, :-1])
            print(f'TPR: {tpr*100:.1f}% ({tpr10*100:.1f}%)\nFDR: {fdr*100:.1f}% ({fdr10*100:.1f}%)')

            # Test: AUC-ROC improved in predictor
            r_base = roc(Y_test, Y_base[:, 1])
            r_pred = roc(Y_test, Y_pred[:, 1])
            a_base = auc(Y_test, Y_base[:, 1])
            a_pred = auc(Y_test, Y_pred[:, 1])
            plt.plot(r_base[:, 0], r_base[:, 1])
            plt.plot(r_pred[:, 0], r_pred[:, 1])
            plt.legend(['baseline', 'predictor'])
            plt.show()

            # Plot feature selection heatmap for diagnostics
            s_pred = invase.predict_features(X_test, threshold=None)
            sns.heatmap(s_pred[:100].T, center=.5, vmin=0, vmax=1, cmap='gray', square=True, cbar=False, linewidth=.5)
            plt.show()

            self.assertGreater(tpr10, 0.9 if i < 4 else 0.5)
            self.assertLess(fdr10, 0.1 if i < 4 else 0.5)
            self.assertGreater(a_pred, a_base)


if __name__ == '__main__':
    unittest.main()
