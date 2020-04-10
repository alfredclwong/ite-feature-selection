import numpy as np
from itertools import product


def process(x, n=3):
    cs = np.cumsum(x, axis=0)
    return (cs[n:]-cs[:-n])/n

def param_search(params):
    lens = [len(l) for l in params.values()]
    for idxs in product(*([i for i in range(l)] for l in lens)):
        yield {k: v[idxs[i]] for i, (k,v) in enumerate(params.items())}
