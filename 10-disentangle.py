from fsite.ournet import OurNet
from fsite.invase import Invase
from data.synthetic_data import partitioned_X, get_YS
from utils.utils import default_env, corr_cmap
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain
import networkx as nx
from graphviz import Digraph
default_env(gpu=True)

n = 10000
n_features = 12
n_treatments = 2


results = np.zeros((10, 3, 3))
for i in range(results.shape[0]):
    X, partitions = partitioned_X(n, n_features, 4, contiguous=False)
    print(partitions)
    part_map = {}
    for j, part in enumerate(partitions):
        for x in part:
            part_map[x] = j
    T, ST = get_YS(X, models=[1], permute=True)
    T = T.flatten().astype(int)
    ST = np.mean(ST[0], axis=0)
    print(ST)
    Y, SY = get_YS(X, models=[2, 3], binary=False, noise=True, permute=True)
    SY = np.mean(SY, axis=1)
    print(SY)
    # sns.heatmap(np.corrcoef(X.T), cmap=corr_cmap(), vmin=-1, vmax=1)
    # plt.scatter(X[T == 0, 0], X[T == 0, 1])
    # plt.scatter(X[T == 1, 0], X[T == 1, 1])
    # plt.show()

    invases = [
        Invase(n_features, n_treatments, 0.1),
        Invase(n_features, 0, 1.5),
        Invase(n_features, 0, 1.5)
    ]
    S_hats = np.zeros((3, n_features))
    for j, (target, S) in enumerate(zip([T, Y[:, 0, None], Y[:, 1, None]], [ST, SY[0], SY[1]])):
        invases[j].train(X, target, 2000 if j > 0 else 1000)
        S_pred = invases[j].predict_features(X, threshold=None)
        S_hats[j] = np.mean(S_pred, axis=0)
        markov_blanket = np.where(S)[0]
        correlated = list(chain(*[partitions[j] for j in set([part_map[x] for x in markov_blanket])]))
        results[i, j] = [np.mean(S_hats[j, idx]) for idx in [
            markov_blanket,
            list(set(correlated)-set(markov_blanket)),
            list(set(range(n_features)) - set(markov_blanket))
        ]]
    print(np.nanmean(results[:i], axis=0))

    G = Digraph()
    with G.subgraph() as S:
        S.attr(rank='same')
        for name, color in zip('T Y0 Y1'.split(), 'red green blue'.split()):
            S.node(name, style='filled', fillcolor=color)
    for j in range(n_features):
        print(S_hats[:, j])
        if np.max(S_hats[:, j]) < 0.25:
            G.node(f'{j}', label='', style='filled', fillcolor='lightgray')
        else:
            weights = S_hats[:, j] / np.sum(S_hats[:, j])
            G.node(f'{j}', label='', style='wedged', fillcolor=f'red;{weights[0]}:green;{weights[1]}:blue;{weights[2]}')
    for part in partitions:
        with G.subgraph() as S:
            for a, b in nx.complete_graph(len(part)).edges():
                S.edge(f'{part[a]}', f'{part[b]}', arrowhead='none')
    for j in range(n_features):
        if ST[j]:
            G.edge(f'{j}', 'T')
        if SY[0, j]:
            G.edge(f'{j}', 'Y0')
        if SY[1, j]:
            G.edge(f'{j}', 'Y1')
    G.view()
