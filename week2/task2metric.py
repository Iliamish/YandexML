from  sklearn import datasets
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import scale
from numpy import linspace

import pandas as pd

data = datasets.load_boston()
X = data.data
y = data.target
X = scale(X)

p = linspace(1, 10, num=200)

kf = KFold(n_splits=5,shuffle=True, random_state=42)
scores = list()
cols = list()
for k in range(1, 200):
    cols.append(k)
    kNR = KNeighborsRegressor(n_neighbors=5, weights='distance', p = p[k])
    arr = cross_val_score(kNR, X, y, cv=kf, scoring='neg_mean_squared_error')
    scores.append(arr.max())


result = pd.DataFrame([scores], columns= cols).sort_values(ascending=False, axis= 1, by=[0])
print(result)