import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from scipy.special import expit

from fsite.invase import Invase
from data.synthetic_data import get_ihdp_XT, get_ihdp_Yb
from utils.utils import default_env, XTY_split
from fsite.cfr import CfrNet

default_env()
X, T = get_ihdp_XT()
Y, beta = get_ihdp_Yb(X, T, 'B1')
n, n_features = X.shape
n_treatments = T.max() + 1
cfrnet = CfrNet(n_features, n_treatments)
# R = cfrnet.rep.predict(X)
# print(np.sum(np.square(R), axis=1))
cfrnet.train(X, T, Y, 1000)
R = cfrnet.rep.predict(X)

Xe = TSNE().fit_transform(X)
Re = TSNE().fit_transform(R)
plt.subplot(121)
for t in range(n_treatments):
    plt.scatter(Xe[T == t, 0], Xe[T == t, 1])
plt.subplot(122)
for t in range(n_treatments):
    plt.scatter(Re[T == t, 0], Re[T == t, 1])
plt.show()
