import pandas as pd
from itertools import chain
import gc

path = "data/linkco2013us_den.csv"
match = ['dob_yy', 'dob_mm', 'dob_wk', 'mager41', 'mager14', 'mager9', 'restatus', 'mbrace', 'mracerec', 'umhisp', 'mracehisp', 'mar', 'meduc', 'fagecomb', 'fagerec11', 'fbrace', 'fracerec', 'ufhisp', 'fracehisp']
chunk = pd.read_csv(path, nrows=100000)
features = [key for key in chunk.keys() if sum(chunk[key].isnull()) == 0]
features += ["aged"]
print(features)
print(chunk[features].head())

twin_deaths = pd.DataFrame()
n = 100000
for i in range(4):
    chunk = pd.read_csv(path, usecols=match, skiprows=range(1,i*n+1), nrows=n)
    twins_idx = (chunk.shift() == chunk).all(axis=1)
    chunk = None
    twins_idx = [idx for idx, b in enumerate(twins_idx) if b]
    twins_idx = list(chain.from_iterable((idx-1+i*n, idx+i*n) for idx in twins_idx))
    #print(twins_idx)
    twins = pd.read_csv(path, usecols=features, nrows=n)
    twins.index = range(i*n,(i+1)*n)
    twins = twins.loc[twins_idx]
    twin_deaths = twin_deaths.append(twins.loc[pd.notna(twins['aged'])])
print(twin_deaths.head())
x1 = twin_deaths.iloc[0]
x2 = twin_deaths.iloc[1]
unique_cols = x1 != x2
unique_cols = [col for col, unique in zip(unique_cols.keys(), unique_cols.values) if unique]
print(twin_deaths[unique_cols].head())

magers = ['mager41', 'mager14', 'mager9']
print(twin_deaths[magers].values)
