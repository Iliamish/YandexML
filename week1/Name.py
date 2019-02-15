import pandas
import re

import sys
sys.path.append("..")


data = pandas.read_csv('titanic.csv', index_col='PassengerId')
data['Pclass'] = data['Pclass'].astype(object)

def clean_name(name):
    # Первое слово до запятой - фамилия
    s = re.search('^[^,]+, (.*)', name)
    if s:
        name = s.group(1)

    # Если есть скобки - то имя пассажира в них
    s = re.search('\(([^)]+)\)', name)
    if s:
        name = s.group(1)

    # Удаляем обращения
    name = re.sub('(Miss\. |Mrs\. |Ms\. )', '', name)

    # Берем первое оставшееся слово и удаляем кавычки
    name = name.split(' ')[0].replace('"', '')

    return name


names = data[data['Sex'] == 'female']['Name'].map(clean_name)
name_counts = names.value_counts()
print(name_counts.head(1).index.values[0])