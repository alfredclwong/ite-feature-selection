import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from multiprocessing.pool import ThreadPool as Pool

from causalbenchmark.utils import combine_covariates_with_observed, COUNTERFACTUAL_FILE_SUFFIX, FILENAME_EXTENSION
from causalbenchmark.evaluate import evaluate

N_PROCESSES = 1
HEADER_Y0, HEADER_Y1 = ['y0', 'y1']
paths_50k = {
    'covariates':       'data/LBIDD-50k/x.csv',
    'factuals':         'data/LBIDD-50k/scaling/',
    'counterfactuals':  'data/LBIDD-50k/scaling/',
    'predictions':      'data/results-50k/',
}
paths_small = {
    'covariates':       'data/LBIDD-small/x.csv',
    'factuals':         'data/LBIDD-small/scaling/',
    'counterfactuals':  'data/LBIDD-small/scaling/',
    'predictions':      'data/results-small/',
}
paths_big = {
    'covariates':       'data/LBIDD/x.csv',
    'factuals':         'data/LBIDD/scaling/factuals/',
    'counterfactuals':  'data/LBIDD/scaling/counterfactuals/',
    'predictions':      'data/results/',
}


class IBM():
    def __init__(self, paths=paths_50k):
        self.paths = paths
        cf_dir = self.paths['counterfactuals']
        cf_ext = f'{COUNTERFACTUAL_FILE_SUFFIX}{FILENAME_EXTENSION}'
        fs = [os.path.join(cf_dir, f) for f in os.listdir(cf_dir) if f.endswith(cf_ext)]
        self.n_rows = sum(sum(1 for l in open(f)) - 1 for f in fs)

    def benchmark(self, predict, n_processes=N_PROCESSES, repeat=False):
        predictions_dir = self.paths['predictions']
        if os.path.exists(predictions_dir):
            processed = set(os.listdir(self.paths['predictions']))
        else:
            os.mkdir(self.paths['predictions'])
            processed = set()

        def make_prediction(fname_and_dataset):
            fname, dataset = fname_and_dataset
            ufid = fname[:-len(FILENAME_EXTENSION)]
            if repeat or fname not in processed:
                X = dataset.values[:, :-2]
                T = dataset.values[:, -2]
                Y = dataset.values[:, -1]
                Y_true = pd.read_csv(os.path.join(self.paths['counterfactuals'], f'{ufid}{COUNTERFACTUAL_FILE_SUFFIX}{FILENAME_EXTENSION}')).values[:, 1:]
                predictions = pd.DataFrame(predict(X, T, Y, Y_true=Y_true), index=dataset.index, columns=[HEADER_Y0, HEADER_Y1])
                predictions.to_csv(os.path.join(self.paths['predictions'], f'{ufid}.csv'))
            return ufid, len(dataset.index)
        Path(self.paths['predictions']).mkdir(exist_ok=True)
        pool = Pool(processes=n_processes)
        with tqdm(total=self.n_rows) as pbar:
            #for ufid, size in pool.imap_unordered(make_prediction, combine_covariates_with_observed(self.paths['covariates'], self.paths['factuals'])):
            for ufid, size in map(make_prediction, combine_covariates_with_observed(self.paths['covariates'], self.paths['factuals'])):
                pbar.set_description(ufid)
                pbar.update(size)

    def evaluate(self, is_individual_prediction=True):
        if is_individual_prediction:
            evaluation = evaluate(self.paths['predictions'], self.paths['counterfactuals'], is_individual_prediction=True)
        else:
            pass
            #ate = 
        print(evaluation)
