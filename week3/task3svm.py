from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import SVC
from numpy import power, arange

newsgroups = datasets.fetch_20newsgroups(
                    subset='all',
                    categories=['alt.atheism', 'sci.space']
             )
X = newsgroups.data
y = newsgroups.target

vectorizer = TfidfVectorizer()
vectorizer.fit_transform(X)

'''
grid = {'C': power(10.0, arange(-5, 6))}
cv = KFold(n_splits=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(vectorizer.transform(X), y)

C = gs.best_params_.get('C')

model = SVC(kernel='linear', random_state=241, C=C)
model.fit(vectorizer.transform(X), y)



coef = pd.DataFrame(model.coef_.data, model.coef_.indices).sort_values(ascending=False, axis= 1, by=[0])
coefr = coef[0].map(lambda w: abs(w)).sort_values(ascending=False)
print(coefr)'''
words = vectorizer.get_feature_names()
map = {24019, 12871, 5088, 5093, 17802, 23673, 21850,5776, 15606, 22936}
word = list()
for k in map:
   word.append( words[k])
word.sort()
print(word)