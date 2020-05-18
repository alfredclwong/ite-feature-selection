import numpy as np
from itertools import product
import os
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity


# n-width mean filter
def process(x, n=3):
    cs = np.cumsum(x, axis=0)
    return (cs[n:]-cs[:-n])/n


def param_search(params):
    for idxs in product(*(range(len(l)) for l in params.values())):
        yield {k: v[idxs[i]] for i, (k, v) in enumerate(params.items())}


def default_env(gpu=False):
    if not gpu:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress warnings
        os.environ['CUDA_VISIBLE_DEVICES'] = ''   # disable GPU
    np.set_printoptions(
            linewidth=160,
            formatter={'float_kind': lambda x: f'{x:.4f}'},
            )


# Nice colormap for sns correlation heatmaps
def corr_cmap():
    return sns.diverging_palette(220, 10, as_cmap=True)


# Chinese restaurant process
def crp(n, alpha=1):
    tables = []
    n_at_tables = []
    n_of_tables = 0
    for i in range(n):
        p = [x / (i + alpha) for x in n_at_tables + [alpha]]
        # p.append(1 - sum(p))
        table_choice = np.random.choice(n_of_tables + 1, p=p)
        if table_choice == n_of_tables:
            tables.append([i])
            n_at_tables.append(1)
            n_of_tables += 1
        else:
            tables[table_choice].append(i)
            n_at_tables[table_choice] += 1
    return tables


# Use KDE method to estimate a pdf with bandwidth determined by CV20 grid search
def est_pdf(x, x_grid, bws=np.linspace(0.1, 1.0, 30), cv=20):
    grid = GridSearchCV(KernelDensity(), {'bandwidth': bws}, cv=cv)
    grid.fit(x[:, None])
    # print(grid.best_params_)
    kde = grid.best_estimator_
    pdf = np.exp(kde.score_samples(x_grid[:, None]))
    return pdf
