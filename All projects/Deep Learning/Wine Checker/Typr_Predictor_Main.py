import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn
from sklearn.utils import shuffle
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.metrics import f1_score,confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

# Reading Data
data_red = pd.read_csv('Wine Checker Dataset/winequality-red.csv')
data_white = pd.read_csv('Wine Checker Dataset/winequality-white.csv')

data_red.describe()
data_white.describe()

# Creating Test Data
data_red['type'] = '0'
data_white['type'] = '1'

data = pd.concat([data_red, data_white])

data = shuffle(data)

y = data['type']
X = data.drop(columns=['type'])

f,ax = plt.subplots(figsize=(10, 10))
sns.heatmap(X.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)

ax = sns.countplot(y,label="Count") 

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

list_models = {
  0 : "RandomForestClassifier",
  1 : "LinearDiscriminantAnalysis",
  2 : 'KNeighborsClassifier',
  3 : 'GaussianNB',
  4 : 'DecisionTreeClassifier',
  5 : 'SVC'
}

def model(i, x_train=x_train, y_train=y_train, list_models=list_models):
    if list_models[i] == 'RandomForestClassifier':
        print(f'using {list_models[i]}')
        model = RandomForestClassifier().fit(x_train, y_train)
    if list_models[i] == 'LinearDiscriminantAnalysis':
        print(f'using {list_models[i]}')
        model = LinearDiscriminantAnalysis().fit(x_train, y_train)
    if list_models[i] == 'KNeighborsClassifier':
        print(f'using {list_models[i]}')
        model = KNeighborsClassifier().fit(x_train, y_train)
    if list_models[i] == 'GaussianNB':
        print(f'using {list_models[i]}')
        model = GaussianNB().fit(x_train, y_train)
    if list_models[i] == 'DecisionTreeClassifier':
        print(f'using {list_models[i]}')
        model = DecisionTreeClassifier().fit(x_train, y_train)
    if list_models[i] == 'SVC':
        print(f'using {list_models[i]}')
        model = SVC().fit(x_train, y_train)
    return model, list_models[i]

accuracy = {
  "RandomForestClassifier" : 0,
  "LinearDiscriminantAnalysis" : 0,
  'KNeighborsClassifier' : 0,
  'GaussianNB' : 0,
  'DecisionTreeClassifier' : 0,
  'SVC' : 0 
}

# Without data engineering

for i in range(len(list_models)):
    classifier, model_name = model(i)
    acc = accuracy_score(y_test, classifier.predict(x_test))
    accuracy[model_name] = acc


x = []
y = []
for i, j in accuracy.items():
    x.append(i)
    y.append(j)

# Select K Best Features

select_feature = SelectKBest(chi2, k=11).fit(x_train, y_train)

print('Score list:', select_feature.scores_)

x_train_2 = select_feature.transform(x_train)
x_test_2 = select_feature.transform(x_test)

for i in range(len(list_models)):
    classifier, model_name = model(i, x_train_2)
    acc = accuracy_score(y_test, classifier.predict(x_test_2))
    accuracy[model_name] = acc

x = []
y = []
for i, j in accuracy.items():
    x.append(i)
    y.append(j)
plt.plot(x, y)
plt.show()

accuracies = []
for i in range(12):
    select_feature = SelectKBest(chi2, k=i+1).fit(x_train, y_train)
    x_train_3 = select_feature.transform(x_train)
    x_test_3 = select_feature.transform(x_test)
    classifier, model_name = model(0, x_train_2)
    acc = accuracy_score(y_test, classifier.predict(x_test_2))
    accuracies.append(acc)

plt.plot(accuracies)
plt.show()












