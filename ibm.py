from causalbenchmark.utils import combine_covariates_with_observed
from causalbenchmark.evaluate import evaluate

from methods.method import Method
from methods.ols import OLS
from methods.knn import KNN
from methods.nn import NN
from methods.invase import INVASE
from methods.ganite import GANITE
from utils.utils import param_search

import os
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np


def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress warnings
os.environ['CUDA_VISIBLE_DEVICES'] = ''   # disable gpu
np.set_printoptions(linewidth=160)

N_PROCESSES = 1
COUNTERFACTUAL_FILE_SUFFIX = "_cf"
FILENAME_EXTENSION = ".csv"
HEADER_Y0, HEADER_Y1 = ["y0", "y1"]

paths_small = {
    "covariates":       "data/LBIDD-small/x.csv",
    "factuals":         "data/LBIDD-small/scaling/",
    "counterfactuals":  "data/LBIDD-small/scaling/",
    "predictions":      "results/",
}
paths_big = {
    "covariates":       "data/LBIDD/x.csv",
    "factuals":         "data/LBIDD/scaling/factuals/",
    "counterfactuals":  "data/LBIDD/scaling/counterfactuals/",
    "predictions":      "results/",
}
paths = paths_small
try:
    os.mkdir(paths["predictions"])
except:
    pass

is_counterfactual_file = lambda file: file.endswith(COUNTERFACTUAL_FILE_SUFFIX + FILENAME_EXTENSION)
is_factual_file = lambda file: file.endswith(FILENAME_EXTENSION) and not is_counterfactual_file(file)
file_to_ufid = lambda file: file.rsplit(sep=(COUNTERFACTUAL_FILE_SUFFIX if is_counterfactual_file(file) else "") + FILENAME_EXTENSION, maxsplit=1)[0]
files = list(filter(is_factual_file, os.listdir(paths["factuals"])))
ufids = list(map(file_to_ufid, files))
nrows = sum([sum(1 for row in open(os.path.join(paths["factuals"], file)))-1 for file in files])

def predict(file_dataset):
    file, dataset = file_dataset
    ufid = file_to_ufid(file)

    X = dataset.values[:, :-2]
    #X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    Z = dataset.values[:, -2]
    Yf = dataset.values[:, -1]
    n_treatments = len(set(Z))
    n, n_features = X.shape
    n_train = int(0.8 * n)
    X_train, Z_train, Yf_train = map(lambda arr: arr[:n_train], [X, Z, Yf])
    X_test, Z_test, Yf_test = map(lambda arr: arr[n_train:], [X, Z, Yf])

    print(f'{ufid}: {n} samples')

    '''
    # linear regression
    XZ = dataset.values[:, 10:-1]
    q, r = np.linalg.qr(XZ)
    #print(q)
    #print(r)
    features = np.nonzero(np.diag(r)>1e-10)[0]
    Yf = dataset.values[:,-1]
    XZ = np.concatenate([XZ, np.ones((XZ.shape[0],1))], axis=1)
    XZ = XZ[:, features]
    beta = np.linalg.inv(XZ.T @ XZ) @ XZ.T @ Yf
    predictions = np.zeros((XZ.shape[0], 2))
    for i in range(2):
        XZ[:,-1] = i
        predictions[:,i] = XZ @ beta
    #'''

    #'''
    hyperparams = {
        'h_layers':     [2],
        'h_dim':        [16, 32],
        'alpha':        [0.5],
        'beta':         [0.5],
        'batch_size':   [64],
        'k1':           [1, 10, 100],
        'k2':           [10],
    }
    best_loss = 99999
    for p in param_search(hyperparams):
        print(p)
        ganite = GANITE(n_features, n_treatments, hyperparams=p)
        g_loss = ganite.train(X_train, Z_train, Yf_train, [1000, 5000], X_test, Z_test, Yf_test, verbose=True)
        #predictions = ganite.predict(X)
        if g_loss < best_loss:
            best_loss = g_loss
            predictions, _ = ganite.predict_counterfactuals(X, Yf, Z, Y_bar=True)
    #'''

    predictions = pd.DataFrame(predictions, index=dataset.index, columns=[HEADER_Y0, HEADER_Y1])

    answers = pd.read_csv(os.path.join(paths["counterfactuals"], f"{ufid}_cf.csv"))

    predictions_file = os.path.join(paths["predictions"], f"{ufid}.csv")
    predictions.to_csv(predictions_file)
    return ufid, len(dataset.index)

pool = Pool(processes=N_PROCESSES)
with tqdm(total=nrows) as pbar:
    for ufid, size in pool.imap_unordered(predict, combine_covariates_with_observed(paths["covariates"], paths["factuals"])):
        pbar.set_description(ufid)
        pbar.update(size)

print("evaluating...")
evaluation = evaluate(paths["predictions"], paths["counterfactuals"], is_individual_prediction=True)
print(evaluation)
