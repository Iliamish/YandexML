import pandas
import numpy
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score

df = pandas.read_csv('abalone.csv')

df['Sex'] = df['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))

X = df.loc[:, 'Sex':'ShellWeight']
y = df['Rings']

kf = KFold(n_splits=5,shuffle=True,random_state=1)
scores = list()
for n in range(1, 51):
    print(n)
    model = RandomForestRegressor(n_estimators=n, random_state=1)
    score = numpy.mean(cross_val_score(model, X, y, cv=kf, scoring='r2'))
    scores.append(score)

for n, score in enumerate(scores):
    if score > 0.52:
        print(n)
        break

plt.plot(scores)
plt.xlabel('n_estimators')
plt.ylabel('score')
plt.savefig('estimators_score.png')