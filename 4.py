import pandas as pd

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, plot_roc_curve
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt

df = pd.read_csv('BRCA_pam50.tsv', sep='\t', index_col=0)
df = df.loc[df['Subtype'].isin(['Luminal A', 'Luminal B'])]
print(df)
X = df.iloc[:, :-1].to_numpy()
y = df['Subtype'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=17)
model = SVC(kernel='linear')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
M = confusion_matrix(y_test, y_pred)
TPR = M[0, 0] / (M[0, 0] + M[0, 1])
TNR = M[1, 1] / (M[1, 0] + M[1, 1])
print(TPR, TNR)
plot_roc_curve(model, X_test, y_test)
plt.plot(1 - TPR, TNR, 'x', c='red')
plt.savefig('test1.png', dpi=300)

model = PCA()
X = model.fit_transform(X)[:, :2]
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=17)
model = SVC(kernel='linear')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))
M = confusion_matrix(y_test, y_pred)
TPR = M[0, 0] / (M[0, 0] + M[0, 1])
TNR = M[1, 1] / (M[1, 0] + M[1, 1])
print(TPR, TNR)
plot_roc_curve(model, X_test, y_test)
plt.plot(1 - TPR, TNR, 'x', c='red')
plt.savefig('test2.png', dpi=300)