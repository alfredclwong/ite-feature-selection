import numpy as np
from itertools import product


def process(x, n=3):
    cs = np.cumsum(x, axis=0)
    return (cs[n:]-cs[:-n])/n

def param_search(params):
    for idxs in product(*(range(len(l)) for l in params.values())):
        yield {k: v[idxs[i]] for i, (k,v) in enumerate(params.items())}
