import numpy as np


def shape_check(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    assert len(y_true.shape) == 2
    assert y_true.shape[1] == 2

def PEHE(y_true, y_pred, norm=False):
    shape_check(y_true, y_pred)
    true_effect = y_true[:, 1] - y_true[:, 0]
    pred_effect = y_pred[:, 1] - y_pred[:, 0]
    if norm:
        eno_sq_error = np.square(1 - pred_effect / true_effect)
        enormse = np.sqrt(np.mean(eno_sq_error))
        return enormse
    else:
        sq_error = np.square(true_effect - pred_effect)
        rmse = np.sqrt(np.mean(sq_error))
        return rmse

def r2(y_true, y_pred):
    rss = np.sum(np.square(y_true - y_pred))
    tss = np.sum(np.square(y_true - np.mean(y_true)))
    return 1 - rss/tss
