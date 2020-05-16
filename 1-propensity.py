import numpy as np
from sklearn.model_selection import train_test_split
from scipy.special import expit
import matplotlib.pyplot as plt
from fsite.invase import Invase
from utils.utils import default_env, est_pdf
from tqdm import tqdm

default_env()
hyperparams = {
    'h_layers_pred':    1,
    'h_dim_pred':       lambda x: 100,  # noqa 272
    'h_layers_base':    1,
    'h_dim_base':       lambda x: 100,  # noqa 272
    'h_layers_sel':     1,
    'h_dim_sel':        lambda x: 2*x,  # noqa 272
    'optimizer':        'adam'
}


def predict(X, T, Y, Y_true=None):
    T = T.astype(int)
    n_treatments = len(set(T))
    n, n_features = X.shape

    X_train, X_test, T_train, T_test = train_test_split(X, T, test_size=.2)
    invase = Invase(n_features, n_treatments, hyperparams=hyperparams)
    invase.train(X_train, T_train, 500, X_test, T_test, verbose=False, save_history=False)

    # Store two sets of propensity scores: one from full INVASE, the other from the baseline model
    propensity_scores = np.zeros((2, n, 2))
    propensity_scores[0] = invase.predict(X, threshold=None)
    propensity_scores[1] = invase.predict(X, threshold=None, use_baseline=True)

    # Calculate two sets of corresponding weights
    weights = np.zeros((2, n))
    pT = [np.mean(T == 0)]
    pT.append(1 - pT[0])
    for i in range(n):
        weights[0, i] = pT[T[i]] / propensity_scores[0, i, T[i]]
        weights[1, i] = pT[T[i]] / propensity_scores[1, i, T[i]]

    # Output max weights from each model for diagnostics
    print(np.max(weights, axis=1))

    # Return IPW ATE estimates for each model
    Yw = Y * weights[0]
    ate_invase = np.mean(Yw[T == 1]) - np.mean(Yw[T == 0])
    Yw = Y * weights[1]
    ate_baseline = np.mean(Yw[T == 1]) - np.mean(Yw[T == 0])
    return ate_invase, ate_baseline


# Load saved progress file and perform more trials if necessary
n_trials = 100
progress = 0
results = np.zeros((n_trials, 4))
try:
    csv = np.loadtxt('1-propensity.csv')
    assert(csv.shape[1] == results.shape[1])
    progress = csv.shape[0]
    if progress < n_trials:
        results[:progress] = csv
        raise IndexError
    results = csv
except (IOError, IndexError):
    for i in tqdm(range(progress, n_trials)):
        # Generate X, T, Y such that X0, X1 and X2 are confounders
        X = np.random.standard_normal(size=(1000, 11))
        n, n_features = X.shape

        T = np.random.binomial(1, p=expit(X[:, 0] * X[:, 1] + X[:, 2]))
        n_treatments = int(np.max(T)) + 1

        beta = np.random.normal(np.ones(n_features), 1)
        Y = np.zeros((n, 2))
        Y[:, 0] = np.random.normal(X @ beta, 1)
        Y[:, 1] = np.random.normal(X @ beta + 1, 1)
        Yf = Y[np.arange(n), T]

        # Perform IPW, report all estimates and biases, save progress
        ate_true = np.mean(Y[:, 1] - Y[:, 0])
        ate_naive = np.mean(Yf[T == 1]) - np.mean(Yf[T == 0])
        ate_invase, ate_baseline = predict(X, T, Yf, Y_true=Y)
        results[i] = [ate_true, ate_naive, ate_invase, ate_baseline]
        ebias = 2 * 0.373 * beta[2]
        bias = ate_naive - ate_true
        print('\n'.join(map(lambda t: ''.join(t), zip(
            map(lambda s: f'{s+":":10}', 'beta E[bias] bias true naive invase baseline'.split()),
            [str(beta)] + list(map(lambda x: f'{x:8f}', [ebias, bias] + list(results[i])))
        ))))
        np.savetxt('1-propensity.csv', results[:i+1])

# Loaded. Report means, stds, and plot estimated pdfs.
print('\t'.join([''] + 'true naive invase baseline'.split()))
print('\t'.join(['mean'] + list(map(lambda x: f'{x:.3f}', np.mean(results, axis=0)))))
print('\t'.join(['std'] + list(map(lambda x: f'{x:.3f}', np.std(results, axis=0)))))

plt.figure()
x_grid = np.linspace(-2, 5, 100)
for i in range(4):
    pdf = est_pdf(results[:, i], x_grid)
    plt.plot(x_grid, pdf)
plt.legend('true naive invase baseline'.split())
plt.xlim([np.min(x_grid), np.max(x_grid)])
plt.ylim([0, 4])
plt.show()
