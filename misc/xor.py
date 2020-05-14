import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from scipy.special import expit
import matplotlib.pyplot as plt
m = 100
alpha = .3
s = .1

n = 10000
n_features = 2
x = np.random.multivariate_normal(np.zeros(n_features), np.eye(n_features), size=n)
for i, c in enumerate([0, 1, 2, 4]):
    y = np.random.binomial(1, p=expit(c * x[:, 0] * x[:, 1]))
    x0 = x[np.where(y == 0)]
    x1 = x[np.where(y == 1)]
    fig = plt.figure(figsize=(1, 1), frameon=False)
    plt.gca().set_axis_off()
    plt.xticks([])
    plt.yticks([])
    for j in range(0, n, m):
        plt.scatter(x0[j:j+m, 0], x0[j:j+m, 1], color='r', alpha=alpha, s=s)
        plt.scatter(x1[j:j+m, 0], x1[j:j+m, 1], color='b', alpha=alpha, s=s)
    plt.savefig(f'../iib-diss/xor{i}.pdf', bbox_inches='tight', pad_inches=0, transparent=True)

'''
model = Sequential()
model.add(Dense(10, input_shape=(n_features,), activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(x, y, batch_size=1024, epochs=100)
y_pred = model.predict(x).flatten()
print(y_pred[:100])
print(y[:100])
print(np.mean((y_pred > .5) == y))
'''
