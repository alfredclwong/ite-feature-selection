import pandas as pd
from itertools import chain

path = "data/linkco2013us_den.csv"
match = ['dob_yy', 'dob_mm', 'dob_wk', 'mager41', 'mager14', 'mager9', 'restatus', 'mbrace', 'mracerec', 'umhisp', 'mracehisp', 'mar', 'meduc', 'fagecomb', 'fagerec11', 'fbrace', 'fracerec', 'ufhisp', 'fracehisp']
chunk = pd.read_csv(path, nrows=1)
features = [key for key in chunk.keys() if sum(chunk[key].isnull()) == 0]
features += ["aged"]
print(features)
n = 1000000
chunk = pd.read_csv(path, usecols=match, nrows=n)
twins_idx = (chunk.shift() == chunk).all(axis=1)
twins_idx = [i for i, b in enumerate(twins_idx) if b]
twins_idx = list(chain.from_iterable((idx-1, idx) for idx in twins_idx))
print(twins_idx)
twins = pd.read_csv(path, usecols=features, nrows=n)
twins = twins.loc[twins_idx]
print(twins.loc[pd.notna(twins["aged"])].head(30))
