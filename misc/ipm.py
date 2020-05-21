import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt


k = lambda x, y: rbf_kernel(x, y, .5)
g = lambda x: np.exp(-xs**2/2) / np.sqrt(2*np.pi)
l = lambda x: np.exp(-abs(xs))/2

n = int(2e3)
a = np.random.normal(0, 1, size=n).reshape(-1, 1)
b = np.random.laplace(0, 1, size=n).reshape(-1, 1)
xs = np.linspace(-6, 6, n).reshape(-1, 1)

w = np.mean(k(xs, b) - k(xs, a), axis=1)   # emp witness
h = k(a, a) + k(b, b) - k(a, b) - k(b, a)
print(np.mean(h[~np.eye(n, dtype=bool)]))  # emp mmd
print(np.mean(w))

plt.plot(xs, g)
plt.plot(xs, l)
plt.plot(xs, w)
plt.show()
