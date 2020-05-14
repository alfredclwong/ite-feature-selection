import os
import pandas as pd
import numpy as np
from shutil import copyfile
import scipy.stats

cf_dir = 'data/LBIDD/scaling/counterfactuals/'
f_dir = 'data/LBIDD/scaling/factuals/'
dst_dir = 'data/LBIDD-50k/scaling/'

n_files = 5  # len(os.listdir(cf_dir))
ufids = []
#ates = np.zeros((n_files, 3))

count = 0
for file in os.listdir(cf_dir):
    Y = pd.read_csv(os.path.join(cf_dir, file)).values
    n = Y.shape[0]
    ites = Y[:, 2] - Y[:, 1]
    n_unique = len(set(np.round(ites, 2)))
    if n == 50000 and n_unique > 1:
        ufids.append(file[:-7])
        print(f'{ufids[count]}\t{n_unique}')
        copyfile(os.path.join(cf_dir, file), os.path.join(dst_dir, file))
        copyfile(os.path.join(f_dir, f'{ufids[count]}.csv'), os.path.join(dst_dir, f'{ufids[count]}.csv'))
        #ate = np.mean(ites)
        #ci = scipy.stats.t.interval(.95, 49999, loc=ate, scale=scipy.stats.sem(ites))
        #ates[count] = [ate] + list(ci)
        count += 1
        if count >= n_files:
            break
#ates = pd.DataFrame(ates, index=ufids, columns=['effect_size', 'li', 'ri']).rename_axis(index='ufid')
#ates.to_csv(os.path.join(dst_dir, 'ates.csv'))
