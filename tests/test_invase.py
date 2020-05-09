import unittest

from synthetic_data import synthetic_data
from utils.metrics import tpr_fdr, roc

## not really a proper test, but the infrastructure is nice to use
class TestInvase(unittest.TestCase):
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

if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    np.set_printoptions(formatter={"float_kind": lambda x: f"{x:.4f}"})
    unittest.main()
