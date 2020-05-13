import numpy as np
from itertools import product
import os
import pandas as pd
import seaborn as sns


def process(x, n=3):
    cs = np.cumsum(x, axis=0)
    return (cs[n:]-cs[:-n])/n


def param_search(params):
    for idxs in product(*(range(len(l)) for l in params.values())):
        yield {k: v[idxs[i]] for i, (k, v) in enumerate(params.items())}


def default_env():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'    # suppress warnings
    os.environ['CUDA_VISIBLE_DEVICES'] = ''     # disable GPU
    np.set_printoptions(
            linewidth=160,
            formatter={'float_kind': lambda x: f'{x:.4f}'},
            )


def read_ihdp(sel_bias=True):
    data = pd.read_csv('data/ihdp.csv').values
    x = data[:, 2:-3]
    x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
    t = data[:, 1].astype(np.int)

    if sel_bias:
        # remove non-whites who were treated
        not_white = data[:, -3] == 0
        treated = t == 1
        keep = np.invert(not_white & treated)
        x = x[keep]
        t = t[keep]


def corr_cmap():
    return sns.diverging_palette(220, 10, as_cmap=True)


# Chinese restaurant process
def crp(n, alpha=1):
    tables = []
    n_at_tables = []
    n_of_tables = 0
    for i in range(n):
        p = [x / (i + alpha) for x in n_at_tables + [alpha]]
        #p.append(1 - sum(p))
        table_choice = np.random.choice(n_of_tables + 1, p=p)
        if table_choice == n_of_tables:
            tables.append([i])
            n_at_tables.append(1)
            n_of_tables += 1
        else:
            tables[table_choice].append(i)
            n_at_tables[table_choice] += 1
    return tables
