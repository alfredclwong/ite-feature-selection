import numpy as np


def process(x, n=3):
    cs = np.cumsum(x, axis=0)
    return (cs[n:]-cs[:-n])/n

