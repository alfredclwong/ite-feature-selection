import os
import pandas as pd
import numpy as np
from shutil import copyfile

cf_dir = 'data/LBIDD/scaling/counterfactuals/'
dst_dir = 'data/LBIDD-50k/scaling/'
count = 0
for file in os.listdir(cf_dir):
    df = pd.read_csv(os.path.join(cf_dir, file))
    n = df.values.shape[0]
    ites = set(np.round(df.values[:,2]-df.values[:,1], 2))
    if n == 50000 and len(ites) > 1:
        ufid = file[:-7]
        print(ufid)
        print(len(set(ites)))
        copyfile(os.path.join(cf_dir, file), os.path.join(dst_dir, file))
        copyfile(os.path.join('data/LBIDD/scaling/factuals/', f'{ufid}.csv'), os.path.join(dst_dir, f'{ufid}.csv'))
        count += 1
        if count >= 5:
            break
