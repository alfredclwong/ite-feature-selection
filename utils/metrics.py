import numpy as np
from sklearn.metrics import confusion_matrix


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


def tpr_fdr(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true.flatten(), y_pred.flatten()).ravel()
    tpr = tp / (tp + fn)
    fdr = fp / (tp + fp)
    return tpr, fdr


def roc(y_true, y_pred):
    thresholds = np.arange(0, 1+1e-5, 1e-2)
    r = np.zeros((thresholds.size, 2))
    for i in range(thresholds.size):
        tn, fp, fn, tp = confusion_matrix(y_true.flatten(), y_pred.flatten() > thresholds[i]).ravel()
        tpr = tp / (tp + fn)
        fpr = fp / (tn + fp)
        r[i] = [fpr, tpr]
    return np.unique(r, axis=0)


def auc(y_true, y_pred):
    r = roc(y_true, y_pred)
    a = np.trapz(r[:, 1], x=r[:, 0])
    return a
