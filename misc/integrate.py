import numpy as np
from math import factorial
from scipy.integrate import quad, tplquad
import matplotlib.pyplot as plt
from scipy.stats import norm


def s(x):
    return 1 / (1 + np.exp(-x))


def ts(a, n):
    return s(a) * np.product([1 - i * s(a) for i in range(n + 1)])


l = lambda x: x * np.exp(-x**2/2) / (1 + np.exp(-x))
r = lambda x, y, z: z * s(x*y+z) * np.exp(-(x**2+y**2+z**2)/2)
r0 = lambda x, y, z: z * (1-s(x*y+z)) * np.exp(-(x**2+y**2+z**2)/2)
approx = lambda a, x, y, z: z * s(a) * (1 + (1-s(a))*(x*y+z-a) + .5*(1-s(a))*(1-2*s(a))*(x*y+z-a)**2) * np.exp(-(x**2+y**2+z**2)/2)
test1 = lambda x, y, z: 1 / (1 + np.exp(-(x*y+z)))
test2 = lambda a, x, y, z: s(a) * (1 + (1-s(a))*(x*y+z-a) + .5*(1-s(a))*(1-2*s(a))*(x*y+z-a)**2 + (1-s(a))*(1-2*s(a))*(1-3*s(a))*(x*y+z-a)**3/6)
test3 = lambda x, y, z: norm.cdf(np.sqrt(np.pi/8)*(x*y+z))

'''
xs = np.arange(-5, 5, .1)
plt.plot(xs, test1(0, 0, xs))
plt.plot(xs, test2(1, 0, 0, xs))
plt.plot(xs, test3(0, 0, xs))
plt.show()
'''

#Il = quad(l, -np.inf, np.inf)[0]
Ir = tplquad(r, -np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf)[0]
print(Ir / np.sqrt(2*np.pi**3))
Ir0 = tplquad(r0, -np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf)[0]
print(Ir0 / np.sqrt(2*np.pi**3))
#Ia = tplquad(lambda x,y,z: approx(1,x,y,z), -np.inf, np.inf, -np.inf, np.inf, -np.inf, np.inf)[0]
#Ir = tplquad(r, -5, 5, -5, 5, -5, 5)[0]
#print(Il * np.sqrt(2/np.pi))

print([ts(1, i) / factorial(i) for i in range(10)])
