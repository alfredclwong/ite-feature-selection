import pandas as pd
import numpy as np
from data.synthetic_data import get_ihdp_XT, get_ihdp_Yb
import matplotlib.pyplot as plt
from fsite.invase import Invase
from utils.utils import default_env
from sklearn.model_selection import train_test_split
from tqdm import tqdm

display_stuff = False
save_stuff = True
headers = 'E[Y0] E[Y1] ATE CATT CATC Var[Y0] Var[Y1]'.split()
descs = 'true pred base'.split()


def report(Ys, T):
    result = np.zeros((len(Ys), 7))
    for i, Y in enumerate(Ys):
        means = np.mean(Y, axis=0)
        catt = np.mean(Y[T == 1, 1]) - np.mean(Y[T == 1, 0])
        catc = np.mean(Y[T == 0, 1]) - np.mean(Y[T == 0, 0])
        result[i, 0:2] = means
        result[i, 2] = means[1] - means[0]
        result[i, 3] = catt
        result[i, 4] = catc
        result[i, 5:7] = np.var(Y, axis=0)
    return result


# Get IHDP data
df = pd.read_csv('data/ihdp.csv', index_col=0)
features = list(df.columns)[1:]
X, T = get_ihdp_XT()
n, n_features = X.shape
n_treatments = int(np.max(T)) + 1

# Plot (confounded) control vs treated outcomes wrt each variable in separate subplots
if display_stuff:
    Y, beta = get_ihdp_Yb(X, T, 'B1')
    Yf = Y[np.arange(n), T]

    # Maintain same x-axis scaling across similar variables to see the different skews
    binary_from = 6
    xlim_cont = np.max(np.abs(X[:, :binary_from])) * .8
    xlim_binary = np.max(np.abs(X[:, binary_from:])) * 1.2
    plt.figure(figsize=(11, 9))
    for i in range(n_features):
        ax = plt.subplot(5, 5, i+1)
        # Highlight relevant features (beta_i != 0) in bold
        ax.set_title('$\\bf{' + f'{features[i]}}}$' if beta[i] else features[i])
        # Only use factual data in plots
        ax.scatter(X[T == 0, i], Yf[T == 0], s=.2)
        ax.scatter(X[T == 1, i], Yf[T == 1], s=.2)
        ax.set_ylim([0, 15])
        if i < binary_from:
            ax.set_xticks([-3, 0, 3])
            ax.set_xlim([-xlim_cont, xlim_cont])
        else:
            ax.set_xticks([-5, 0, 5])
            ax.set_xlim([-xlim_binary, xlim_binary])
    plt.subplots_adjust(wspace=.3, hspace=.8)
    plt.savefig('../iib-diss/ihdp-b1-sample.pdf', bbox_inches='tight')
    plt.show()

default_env(gpu=True)
n_trials = 1000 * len(descs)
progress = 0
results = np.zeros((n_trials, len(headers)))
results_path = 'results/2-direct.csv'
try:
    _results = pd.read_csv(results_path, index_col=0).values[:, :-1]
    assert(_results.shape[1] == results.shape[1])
    progress = _results.shape[0]
    if progress < n_trials:
        results[:progress] = _results
        raise IndexError
    results = _results
except (IOError, IndexError):
    for i in tqdm(range(progress, n_trials, len(descs))):
        Y, beta = get_ihdp_Yb(X, T, 'B1')
        Yf = Y[np.arange(n), T]

        invases = [Invase(n_features, n_classes=0, lam=1) for t in range(n_treatments)]
        Y_pred = np.zeros((n, n_treatments))
        Y_base = np.zeros((n, n_treatments))
        for t in range(n_treatments):
            Xt = X[T == t]
            Yft = Yf[T == t]
            X_train, X_test, Y_train, Y_test = train_test_split(Xt, Yft, test_size=.2)
            invases[t].train(X_train, Y_train, 1000, X_test, Y_test, verbose=False, save_history=False)  # , S_true=(beta > 0))
            print(np.array2string(beta, formatter={'float_kind': '{0:.2f}'.format}))
            Y_pred[:, t] = invases[t].predict(X).flatten()
            Y_base[:, t] = invases[t].predict(X, use_baseline=True).flatten()

        results[i:i+len(descs)] = report([Y, Y_pred, Y_base], T)
        df = pd.DataFrame(results[:i+len(descs)], columns=headers).round(4)
        df['desc'] = pd.Series(dtype='string')
        for j, desc in enumerate(descs):
            df.iloc[j:i+len(descs):len(descs), -1] = desc
        if save_stuff:
            df.to_csv(results_path)
