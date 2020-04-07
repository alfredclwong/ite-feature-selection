from causalbenchmark.utils import combine_covariates_with_observed
from causalbenchmark.evaluate import evaluate

from methods.method import Method

import os
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np

N_PROCESSES = 4
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

    '''
    XZ = dataset.values[:,:-1]
    q, r = np.linalg.qr(XZ)
    features = np.nonzero(np.diag(r)>1e-10)[0]
    Yf = dataset.values[:,-1]
    XZ = np.concatenate([XZ, np.ones((XZ.shape[0],1))], axis=1)
    XZ = XZ[:, features]
    beta = np.linalg.inv(XZ.T @ XZ) @ XZ.T @ Yf
    '''

    #predictions = np.zeros((XZ.shape[0], 2))
    #for i in range(2):
    #    XZ[:,-1] = i
    #    predictions[:,i] = XZ @ beta
    #predictions = pd.DataFrame(predictions, index=dataset.index, columns=[HEADER_Y0, HEADER_Y1])
    predictions = pd.read_csv(os.path.join(paths["counterfactuals"], f"{ufid}_cf.csv"))

    predictions_file = os.path.join(paths["predictions"], f"{ufid}.csv")
    predictions.to_csv(predictions_file)
    return ufid, len(dataset.index)

pool = Pool(processes=N_PROCESSES)
with tqdm(total=nrows) as pbar:
    for ufid, size in pool.imap_unordered(predict, combine_covariates_with_observed(paths["covariates"], paths["factuals"])):
        pbar.update(size)
        pbar.set_description(ufid)

print("evaluating...")
evaluation = evaluate(paths["predictions"], paths["counterfactuals"], is_individual_prediction=1) 
print(evaluation)
