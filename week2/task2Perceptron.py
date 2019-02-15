from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

scaler = StandardScaler()
data_train = pd.read_csv('perceptron-train.csv', header=None, index_col=False)
data_test = pd.read_csv('perceptron-test.csv', header=None, index_col=False)
y_train = data_train[0]
data_train.drop(0, inplace=True,axis=1)
X_train = data_train
y_test = data_test[0]
data_test.drop(0, inplace=True,axis=1)
X_test = data_test

Pers = Perceptron(random_state=241)
Pers.fit(X_train,y_train)
predict = Pers.predict(X_test)
accuracy = accuracy_score(y_test, predict)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

Pers.fit(X_train_scaled,y_train)
predict = Pers.predict(X_test_scaled)
accuracy_scaled = accuracy_score(y_test, predict)

print(accuracy_scaled-accuracy)
