import pandas as pd
import numpy as np

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, plot_roc_curve

import matplotlib.pyplot as plt

df = pd.read_csv('BRCA_pam50.tsv', sep='\t', index_col=0)
df = df.loc[df["Subtype"].isin(["Luminal A", "Luminal B"])]
print(df)
X = df.iloc[:, :-1].to_numpy()
y = df["Subtype"].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=17)
model = SVC(kernel="linear")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
M = confusion_matrix(y_test, y_pred)
TPR = M[0, 0] / (M[0, 0] + M[0, 1])
TNR = M[1, 1] / (M[1, 0] + M[1, 1])
print(TPR, TNR)
plot_roc_curve(model, X_test, y_test)
plt.plot(1 - TPR, TNR, "x", c="red")
plt.savefig("test1.png", dpi=300)

w = model.coef_[0]
genes = df.iloc[:,:-1].T
genes['coef'] = np.abs(w)
print(genes.sort_values(by='coef').iloc[-5:])

df = df[['KRT17', 'ACTR3B', 'MIA', 'BIRC5', 'BAG1', 'Subtype']]
print(df)
X = df.iloc[:, :-1].to_numpy()
y = df["Subtype"].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=17)
model = SVC(kernel="linear")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
M = confusion_matrix(y_test, y_pred)
TPR = M[0, 0] / (M[0, 0] + M[0, 1])
TNR = M[1, 1] / (M[1, 0] + M[1, 1])
print(TPR, TNR)
plot_roc_curve(model, X_test, y_test)
plt.plot(1 - TPR, TNR, "x", c="red")
plt.savefig("test2.png", dpi=300)