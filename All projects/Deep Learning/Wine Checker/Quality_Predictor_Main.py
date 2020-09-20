# Essentials
import pandas as pd
import numpy as np
import tensorflow as tf
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
import time

# Sklearn stuff
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix

# Sklearn models
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Tensorflow
from tensorflow import keras
from tensorflow.python.keras.utils import np_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam as adam


# Reading Data

data_red = pd.read_csv('Wine Checker Dataset/winequality-red.csv')
data_white = pd.read_csv('Wine Checker Dataset/winequality-white.csv')

data_red
data_white

data = data_red.append(data_white)

data

# Merging Red and White
is_red = []
for i in range(len(data_red)):
    is_red.append(1)
for i in range(len(data_white)):
    is_red.append(0)   
    
is_white = []
for i in range(len(data_red)):
    is_white.append(0)
for i in range(len(data_white)):
    is_white.append(1)    
    
data['is_red'] = is_red
data['is_white'] = is_white

# Shuffle
data = shuffle(data)

# Quantizing
quality = data['quality'].values
is_good = [1]*len(data)
for i in range(len(data)):
    if quality[i]<7:
        is_good[i] = 0
data['is_good'] = is_good

# Quantizing
quality = data['quality'].values
is_good = [1]*len(data)
for i in range(len(data)):
    if quality[i]<7:
        is_good[i] = 0
data['is_good'] = is_good


# Heatmap
sns.heatmap(X.corr(), annot=True, linewidths=.5, fmt= '.1f')

# Normalizing
X = X.values
X_norm = MinMaxScaler().fit_transform(X)

test = [1,2,3,4,5,6,7,8,9]
test[:4]


# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, 
                                                    random_state = 101)
# ML Models
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

# Accuracy dictionary
accuracy = {
  "RandomForestClassifier" : 0,
  "LinearDiscriminantAnalysis" : 0,
  'KNeighborsClassifier' : 0,
  'GaussianNB' : 0,
  'DecisionTreeClassifier' : 0,
  'SVC' : 0 
}

# Training with no preprocessing
start = time.process_time()
classifier, model_name = model(0)
print(time.process_time() - start)
prediction = classifier.predict(x_test)
acc = accuracy_score(y_test, prediction)
print(acc)

feat_importances = pd.Series(classifier.feature_importances_, index= X.columns)
feat_importances.nlargest(13).plot(kind='barh')


X_feat = X[[10, 7, 1, 6]]
x_train_feat, x_test_feat, y_train_feat, y_test_feat = train_test_split(X_feat, y, test_size = 0.30, 
                                                    random_state = 101)
start = time.process_time()
classifier, model_name = model(0, x_train=x_train_feat, y_train=y_train_feat)
print(time.process_time() - start)
prediction = classifier.predict(x_test_feat)
acc = accuracy_score(y_test_feat, prediction)
print(acc)

# DL Models

y_train = np_utils.to_categorical(y_train, 2)
y_test = np_utils.to_categorical(y_test, 2)


model = Sequential()
model.add(Dense(4547, input_dim=13, activation='relu'))
model.add(Dense(500, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

optimizer = adam(0.001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.fit(x_train, y_train, epochs=15, batch_size=10)

_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))


















