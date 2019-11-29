import numpy as np

def PEHE(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    assert len(y_true.shape) == 2
    assert y_true.shape[1] == 2
    return np.mean(np.square((y_true[:,0] - y_true[:,1]) - (y_pred[:,0] - y_pred[:,1])))
