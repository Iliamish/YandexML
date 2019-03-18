import math
import pandas as pd
import cmath
from sklearn.metrics import roc_auc_score
import sys

data = pd.read_csv("data-logistic.csv", index_col = None, header = None )

y = data[0]
data.drop(0, inplace=True, axis=1)

def fw1(w1, w2, y, X, k, C):
    l = len(y)
    S = 0
    for i in range(0, l):
        S += y[i] * X[1][i] * (1.0 - 1.0 / (1.0 + cmath.exp(-y[i] * (w1 * X[1][i] + w2 * X[2][i]))))

    return w1 + (k * (1.0 / l) * S) - k * C * w1

def fw2(w1, w2, y, X, k, C):
    l = len(y)
    S = 0
    for i in range(0, l):
        S += y[i] * X[2][i] * (1.0 - 1.0 / (1.0 + cmath.exp(-y[i] * (w1 * X[1][i] + w2 * X[2][i]))))

    return w2 + (k * (1.0 / l) * S) - k * C * w2

def grad(y, X, C=0.0, w1=0.0, w2=0.0, k=0.1, err=1e-5):
    i = 0
    i_max = 10000
    w1_new, w2_new = w1, w2

    while True:
        i += 1
        w1_new, w2_new = fw1(w1, w2, y, X, k, C), fw2(w1, w2, y, X, k, C)
        e = math.sqrt((w1_new - w1) ** 2 + (w2_new - w2) ** 2)

        if i >= i_max or e <= err:
            break
        else:
            w1, w2 = w1_new, w2_new

    return [w1_new, w2_new]

w1, w2 = grad(y, data)
rw1, rw2 = grad(y, data, 10.0)

def a(X, w1, w2):
    return 1.0 / (1.0 + cmath.exp(-w1 * X[1] - w2 * X[2]))


y_score = data.apply(lambda x: a(x, w1, w2), axis=1)
y_rscore = data.apply(lambda x: a(x, rw1, rw2), axis=1)

auc = roc_auc_score(y, y_score)
rauc = roc_auc_score(y, y_rscore)

print(auc, rauc)