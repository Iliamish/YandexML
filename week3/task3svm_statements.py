import numpy as np
import pandas as pd
import xlrd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import scale, MinMaxScaler
from sklearn.svm import SVC

clf = SVC(random_state=241, C = 100000)

data = pd.read_csv('svm-data.csv', index_col=False, header=None)
y = data[0]
data.drop(0, inplace=True, axis=1)

clf.fit(data,y)
print(clf.support_)