import pandas
from sklearn.decomposition import PCA
from numpy import corrcoef

df = pandas.read_csv('close_prices.csv')
X = df.loc[:, 'AXP':]

pca = PCA(n_components=10)
pca.fit(X.values)

var = 0
n_var = 0
for v in pca.explained_variance_ratio_:
    n_var += 1
    var += v
    if var >= 0.9:
        break

print( n_var)

df_comp = pandas.DataFrame(pca.transform(X))
comp0 = df_comp[0]

df2 = pandas.read_csv('djia_index.csv')
dji = df2['^DJI']
corr = corrcoef(comp0, dji)
print( corr)

comp0_w = pandas.Series(pca.components_[0])
comp0_w_top = comp0_w.sort_values(ascending=False).head(1).index[0]
company = X.columns[comp0_w_top]
print( company)




