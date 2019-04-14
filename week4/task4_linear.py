import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from scipy.sparse import hstack

train = pandas.read_csv('salary-train.csv')

def text_transform(text):

    text = text.map(lambda t: t.lower())
    text = text.replace('[^a-zA-Z0-9]', ' ', regex=True)
    return text

vec = TfidfVectorizer(min_df=5)
X_train_text = vec.fit_transform(text_transform(train['FullDescription']))

train['LocationNormalized'].fillna('nan', inplace=True)
train['ContractTime'].fillna('nan', inplace=True)

enc = DictVectorizer()
X_train_cat = enc.fit_transform(train[['LocationNormalized', 'ContractTime']].to_dict('records'))

X_train = hstack([X_train_text, X_train_cat])

y_train = train['SalaryNormalized']
model = Ridge(alpha=1)
model.fit(X_train, y_train)

test = pandas.read_csv('salary-test-mini.csv')
X_test_text = vec.transform(text_transform(test['FullDescription']))
X_test_cat = enc.transform(test[['LocationNormalized', 'ContractTime']].to_dict('records'))
X_test = hstack([X_test_text, X_test_cat])

y_test = model.predict(X_test)

print(y_test)