import pandas
data = pandas.read_csv('titanic.csv', index_col='PassengerId')
correl = pandas.DataFrame([data['SibSp'], data['Parch']])

print(data.corr())