import pandas as pd
import numpy as np
import time

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, plot_roc_curve
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

np.random.seed(17)

N = 10000
X1 = np.random.normal(loc=0, size=(N, 1))
X2 = np.random.normal(loc=10, size=(N, 1))
X = np.vstack([X1, X2])
y = np.array([0]*N + [10]*N)
model = SVC(kernel = 'linear')
start = time.time()
model.fit(X, y)
end = time.time()
print(end-start)
print(accuracy_score(model.predict(X), y))
M = confusion_matrix(model.predict(X), y)
TPR = M[0, 0] / (M[0, 0] + M[0, 1])
TNR = M[1, 1] / (M[1, 0] + M[1, 1])
print(TPR, TNR)
plot_roc_curve(model, X, y)
plt.plot(1 - TPR, TNR, 'x', c='red')
plt.savefig('test51.png', dpi=300)

N = 1000
X1 = np.random.normal(loc=0, size=(N, 1))
X2 = np.random.normal(loc=0, size=(N, 1))
X = np.vstack([X1, X2])
y = np.array([0]*N + [10]*N)
model = SVC(kernel = 'linear')
start = time.time()
model.fit(X, y)
end = time.time()
print(end-start)
print(accuracy_score(model.predict(X), y))
M = confusion_matrix(model.predict(X), y)
TPR = M[0, 0] / (M[0, 0] + M[0, 1])
TNR = M[1, 1] / (M[1, 0] + M[1, 1])
print(TPR, TNR)
plot_roc_curve(model, X, y)
plt.plot(1 - TPR, TNR, 'x', c='red')
plt.savefig('test52.png', dpi=300)