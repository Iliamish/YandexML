from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import scale
import pandas as pd

from sklearn.model_selection import cross_val_score

data = pd.read_csv('Wine.data', index_col=False)
test = pd.read_csv('Test.data', index_col=False, header=None)
y = data['Type']
data.drop('Type', inplace=True, axis=1)

kf = KFold(n_splits=5,shuffle=True,random_state=42)
scores = list()
cols = list()

for k in range(1,51):
    cols.append(k)
    kNN = KNeighborsClassifier(n_neighbors=k)
    arr = cross_val_score(kNN, data, y, cv=kf, scoring='accuracy')
    scores.append(arr.mean())


result = pd.DataFrame([scores], columns= cols).sort_values(ascending=False, axis= 1, by=[0])

del scores,cols,result

scores = list()
cols = list()

data_scale = scale(data)

for k in range(1,51):
    cols.append(k)
    kNN = KNeighborsClassifier(n_neighbors=k)
    arr = cross_val_score(kNN, data_scale, y, cv=kf, scoring='accuracy')
    scores.append(arr.mean())


result = pd.DataFrame([scores], columns= cols).sort_values(ascending=False, axis= 1, by=[0])

print(result)



