import numpy as np

def PEHE(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    assert len(y_true.shape) == 2
    assert y_true.shape[1] == 2
    true_treatment_effects = y_true[:, 1] - y_true[:, 0]
    pred_treatment_effects = y_pred[:, 1] - y_pred[:, 0]
    squared_errors = np.square(true_treatment_effects - pred_treatment_effects)
    return np.sqrt(np.mean(squared_errors))
