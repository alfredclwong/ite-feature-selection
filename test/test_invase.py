import unittest

from fsite.invase import Invase
from utils.synthetic_data import synthetic_data
from utils.metrics import tpr_fdr, roc
from utils.utils import default_env


class Test_Invase(unittest.TestCase):
    def testAll(self):
        default_env()
        for i in range(3, 7):
            X, t, Y, S = synthetic_data(N=20000, n_features=11, models=[i], corr=False)
            N, n_features = X.shape
            #Y += np.random.randn(N, 1) * 0.1

            N_train = int(.5 * N)
            X_train = X[N_train:]
            Y_train = Y[N_train:]
            X_test = X[:N_train]
            Y_test = Y[:N_train]
            S_test = S[0, :N_train]

            invase = Invase(n_features, n_classes=2)
            invase.train(X_train, Y_train, 5000, X_test, Y_test, batch_size=512)
            Y_pred = invase.predict(X_test)
            Y_base = invase.predict(X_test, use_baseline=True)
            S_pred = invase.predict_features(X_test)
            #X_str, Y_str, t_str, Y_pred_str, S_pred_str = map(np.array2string, [X, Y, t, Y_pred, S_pred.astype(int)])
            #print('\n'.join(['X', X_str, 'Y', Y_str, 't', t_str, 'Y_pred', Y_pred_str, 'S_pred', S_pred_str]))
            tpr, fdr = tpr_fdr(S_test, S_pred)
            tpr10, fdr10 = tpr_fdr(S_test[:, :-1], S_pred[:, :-1])
            print(f'TPR: {tpr*100:.1f}% ({tpr10*100:.1f}%)\nFDR: {fdr*100:.1f}% ({fdr10*100:.1f}%)')
            s_pred = invase.predict_features(X_test, threshold=None)
            r = roc(Y_test, Y_pred[:, 1])
            #plt.plot(r[:,0], r[:,1])
            r = roc(Y_test, Y_base[:, 1])
            #plt.plot(r[:,0], r[:,1])
            #plt.show()
            #sns.heatmap(s_pred[:100].T, center=.5, vmin=0, vmax=1, cmap='gray', square=True, cbar=False, linewidth=.5)
            #plt.show()


if __name__ == '__main__':
    unittest.main()
