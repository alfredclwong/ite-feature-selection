import numpy as np
from sklearn.model_selection import train_test_split
from scipy.special import expit
import matplotlib.pyplot as plt
from fsite.invase import Invase
from utils.utils import default_env, est_pdf
from tqdm import tqdm
import pandas as pd


def p(X): return expit(X[:, 0] * X[:, 1] + X[:, 2])


def predict(X, T, Y, Y_true=None):
    n_treatments = len(set(T))
    n, n_features = X.shape

    X_train, X_test, T_train, T_test = train_test_split(X, T, test_size=.2)
    invase = Invase(n_features, n_treatments, hyperparams=hyperparams)
    invase.train(X_train, T_train, 500, X_test, T_test, verbose=False, save_history=False)

    # Store three sets of propensity scores: oracle, INVASE, baseline
    n_models = 3
    propensity_scores = np.zeros((n_models, n, n_treatments))
    propensity_scores[0, :, 1] = p(X)
    propensity_scores[0, :, 0] = 1 - propensity_scores[0, :, 1]
    propensity_scores[1] = invase.predict(X, threshold=None)
    propensity_scores[2] = invase.predict(X, threshold=None, use_baseline=True)

    # Calculate three sets of corresponding weights
    pT = np.array([np.mean(T == t) for t in range(n_treatments)])
    weights = pT[T] / propensity_scores[:, np.arange(n), T]  # (3, n)

    # Output max weights from each model for diagnostics
    max_weights = np.max(weights, axis=1)

    # Return IPW ATE estimates for each model
    Y_weighted = Y * weights
    ates_ipw = np.mean(Y_weighted[:, T == 1], axis=1) - np.mean(Y_weighted[:, T == 0], axis=1)
    return ates_ipw, max_weights


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

# Load saved progress file and perform more trials if necessary
save_stuff = False
results_path = 'results/1-propensity.csv'
headers = 'true naive oracle invase base max(W) max(Wi) max(Wb)'
n_trials = 1000
progress = 0
results = np.zeros((n_trials, 8))
try:
    _results = pd.read_csv(results_path, index_col=0).values
    assert(_results.shape[1] == results.shape[1])
    progress = _results.shape[0]
    if progress < n_trials:
        results[:progress] = _results
        raise IndexError
    results = _results
except (IOError, IndexError):
    for i in tqdm(range(progress, n_trials)):
        # Generate X, T, Y such that X0, X1 and X2 are confounders
        X = np.random.standard_normal(size=(1000, 11))
        n, n_features = X.shape

        T = np.random.binomial(1, p=p(X))
        n_treatments = int(np.max(T)) + 1

        beta = np.random.normal(np.ones(n_features), 1)
        Y = np.zeros((n, 2))
        Y[:, 0] = np.random.normal(X @ beta, 1)
        Y[:, 1] = np.random.normal(X @ beta + 1, 1)
        Yf = Y[np.arange(n), T]

        # Perform IPW, report all estimates and biases, save progress
        ate_true = np.mean(Y[:, 1] - Y[:, 0])
        ate_naive = np.mean(Yf[T == 1]) - np.mean(Yf[T == 0])
        ates_ipw, max_weights = predict(X, T, Yf, Y_true=Y)
        results[i] = [ate_true, ate_naive] + list(ates_ipw) + list(max_weights)
        ebias = 2 * 0.373 * beta[2]
        bias = ate_naive - ate_true
        print('\n'.join(map(lambda t: ''.join(t), zip(
            map(lambda s: f'{s+":":10}', f'beta E[bias] bias {headers}'.split()),
            [str(beta)] + list(map(lambda x: f'{x:8f}', [ebias, bias] + list(results[i])))
        ))))
        df = pd.DataFrame(results[:i+1], columns=headers.split())
        if save_stuff:
            df.to_csv(results_path)

# Loaded. Report means, stds, and plot estimated pdfs.
print('\t'.join([''] + headers.split()))
print('\t'.join(['mean'] + list(map(lambda x: f'{x:.3f}', np.mean(results, axis=0)))))
print('\t'.join(['std'] + list(map(lambda x: f'{x:.3f}', np.std(results, axis=0)))))

plt.figure(figsize=(4, 2.5))
x_grid = np.linspace(-1, 3, 100)
for i in range(5):
    pdf = est_pdf(results[:, i], x_grid)
    plt.plot(x_grid, pdf)
plt.legend(headers.split())
plt.xlabel('ATE estimate')
plt.ylabel('Probability')
plt.xticks(np.arange(4))
plt.xlim([np.min(x_grid), np.max(x_grid)])
plt.ylim([0, 4])
if save_stuff:
    plt.savefig('../iib-diss/1-propensity.pdf', bbox_inches='tight')
plt.show()
