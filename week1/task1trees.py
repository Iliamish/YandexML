import numpy as np
import pandas as pn
from sklearn.tree import DecisionTreeClassifier

data = pn.read_csv('titanic.csv', index_col='PassengerId')

columns = ['Name','SibSp','Parch','Ticket','Cabin','Embarked']
data.drop(columns, inplace=True, axis=1)

data.dropna(inplace=True);
survived = data['Survived']
data.drop('Survived', inplace=True, axis=1)

data['Sex'] = data['Sex'] == 'male'

clf = DecisionTreeClassifier(random_state=241)
clf.fit(data, survived)

importances = clf.feature_importances_

print(importances)
print(data.head())
