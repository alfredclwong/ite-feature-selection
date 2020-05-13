import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from graphviz import Digraph
from sklearn.model_selection import train_test_split
from fsite.invase import Invase
from utils.utils import corr_cmap, default_env, crp
from data.synthetic_data import get_YS


default_env()
ord_A = ord('A')
hyperparams = {
    'h_layers_pred':    2,
    'h_dim_pred':       lambda x: 100,  # noqa 272
    'h_layers_base':    2,
    'h_dim_base':       lambda x: 100,  # noqa 272
    'h_layers_sel':     1,
    'h_dim_sel':        lambda x: 2*x,  # noqa 272
    'lam':              0.2,
    'optimizer':        'adam'
}

N = 10000
n_features = 12
biases = np.random.uniform(size=n_features)
stds = .1 * np.ones(n_features)  #np.random.uniform(size=n_features)


# Generate causal graph
def random_graph(n):
    E_arr = np.random.choice(2, size=n*(n-1)//2)
    E = np.zeros((n, n))
    E[np.tril_indices(n, k=-1)] = E_arr
    return E


E = np.zeros((n_features, n_features))
partition = crp(n_features, alpha=2)
print(partition)
for subset in partition:
    E[np.ix_(subset, subset)] = random_graph(len(subset))
dot_E = Digraph()
for i in range(n_features):
    dot_E.node(chr(ord_A+i))#, xlabel=f'{stds[i]:.2f}')
    for j in range(i):
        if E[i, j]:
            dot_E.edge(chr(ord_A+j), chr(ord_A+i))#, label=f'{E[i, j]}')
dot_E.view()
print(E)

# Generate data
roots, = np.where(np.sum(E, axis=1)==0)
X = np.zeros((N, n_features))
X[:, roots] = np.random.standard_normal(size=(N, roots.size))
for i in range(n_features):
    if i in roots:
        continue
    #X[:, i] = np.random.normal(np.product(np.sin(X * E[i]), axis=-1), stds[i])
    X[:, i] = np.random.normal(np.sin(2 * np.pi * X @ E[i].T), stds[i])
#print(X)

T, S_T = get_YS(X, models=[6], noise=True)
Y, S_Y = get_YS(X, models='B')
n_treatments = int(np.max(T)) + 1

TS = np.concatenate([T, S_T[0]], axis=-1)
X_train, X_test, TS_train, TS_test = train_test_split(X, TS, test_size=.2)
(T_train, S_train), (T_test, S_test) = map(lambda arr: np.hsplit(arr, [1]), [TS_train, TS_test])
invase = Invase(n_features, n_treatments)
invase.train(X_train, T_train, 5000, X_test, T_test, S_test, save_history=False)
#S_pred = invase.predict_features(X_test)

s_pred = invase.predict_features(X_test, threshold=None)
sns.heatmap(s_pred[:100].T, center=.5, vmin=0, vmax=1, cmap='gray', square=True, cbar=False, linewidth=.5)
plt.show()


'''
s = np.zeros((n_features, n_features))
for i in range(n_features):
    other_idxs = [j for j in range(n_features) if j != i]
    X_others = X[:, other_idxs]
    invase = Invase(n_features-1, 0, hyperparams=hyperparams)
    invase.train(X_others, X[:, i], 2000, save_history=False, verbose=False)
    s[i, other_idxs] = np.mean(invase.predict_features(X_others), axis=0)
    print(s[i])
dot_s = Digraph()
for i in range(n_features):
    dot_s.node(chr(ord_A+i))
    for j in range(n_features):
        if s[i, j]:
            if s[j, i]:
                if i < j:
                    dot_s.edge(chr(ord_A+j), chr(ord_A+i), arrowhead='none')
            else:
                dot_s.edge(chr(ord_A+j), chr(ord_A+i), label=f'{s[i, j]}')
dot_s.view()
print(s)
'''

'''
import pandas as pd
# Read X and T, standardise X
df = pd.read_csv('data/ihdp.csv')
data = df.values; headers = df.columns[2:-3]
X = data[:, 2:-3]
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
T = data[:, 1].astype(np.int)

# Introduce selection bias
not_white = data[:, -3] == 0
treated = T == 1
keep = np.invert(not_white & treated)  # remove non-whites who were treated
X = X[keep]
T = T[keep]

# Extract dims
n, n_features = X.shape
n_treatments = int(np.max(T)) + 1
assert n_treatments == 2

print(X)

# Predict predictive power scores
corr_matrix = np.corrcoef(X, rowvar=False)
invase_matrix = np.zeros((n_features, n_features))
for i in range(n_features):
    n_classes = len(set(X[:, i]))
    if n_classes > 3:
        n_classes = 0
    invase = INVASE(n_features=n_features-1, n_classes=n_classes, lam=0.05)
    other_features = [j for j in range(n_features) if i!=j]
    history = invase.train(X[:, other_features], X[:, i], 8000, verbose=False)
    S = invase.predict_features(X[:, other_features], threshold=None)
    sns.heatmap(S[:100].T, center=0.5, vmin=0, vmax=1, cmap='gray', square=True, cbar=False, yticklabels=headers[other_features])
    plt.title(headers[i])
    plt.show()
    invase_matrix[i, other_features] = np.mean(invase.predict_features(X[:, other_features]), axis=0)
print(corr_matrix)
print(invase_matrix)
#'''
