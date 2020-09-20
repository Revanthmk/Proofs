import numpy as np
import pandas as pd
import sns
import os
import re
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

for dirname, _, filenames in os.walk('/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

train_data = pd.read_csv("/input/titanic/train.csv")
train_data.head()

test_data = pd.read_csv("/kaggle/input/titanic/test.csv")
test_data.head()

women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)

men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)

train_data[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# checking for outliers
features = train_data[['Age', 'SibSp', 'Parch', 'Fare']].columns
for i in features:
    sns.boxplot(x="Survived",y=i,data=train_data)
    plt.title(i+" by "+"Survived")
    plt.show()

# Drop outliers
train_data = train_data.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)
train_data = train_data.drop(train_data[train_data['Fare'] > 500].index).reset_index(drop=True)

# Adding all the changes
df = pd.concat((train.loc[:,'Pclass':'Embarked'], test.loc[:,'Pclass':'Embarked'])).reset_index(drop=True)

# Correlation matrix
survived = train.drop(train[train['Survived'] != 1].index)
not_survived = train.drop(train[train['Survived'] != 0].index)
basic_analysis(survived,not_survived)

# Checking feature distribution
def basic_details(df):
    b = pd.DataFrame()
    b['Missing value, %'] = round(df.isnull().sum()/df.shape[0]*100)
    b['N unique value'] = df.nunique()
    b['dtype'] = df.dtypes
    return b
basic_details(df)

pd.DataFrame(df.groupby('Pclass')['Age'].describe())
# Dropping name
df = df.drop(['Name'], axis=1)
# Converting Sex into number
df["Sex"][df["Sex"] == "male"] = 0
df["Sex"][df["Sex"] == "female"] = 1
df["Sex"] = df["Sex"].astype(int)

df['Age'] = df.groupby(['Pclass'])['Age'].transform(lambda x: x.fillna(x.median()))
df["Age"] = df["Age"].astype(int)

df['Age_cat'] = pd.qcut(df['Age'],q=[0, .16, .33, .49, .66, .83, 1], labels=False, precision=1)

# Discretizing Fare 
df['Fare'] = df.groupby("Pclass")['Fare'].transform(lambda x: x.fillna(x.median())) 

df['Zero_Fare'] = df['Fare'].map(lambda x: 1 if x == 0 else (0))


def fare_category(fr): 
    if fr <= 7.91:
        return 1
    elif fr <= 14.454 and fr > 7.91:
        return 2
    elif fr <= 31 and fr > 14.454:
        return 3
    return 4

df['Fare_cat'] = df['Fare'].apply(fare_category) 


df["Embarked"] = df["Embarked"].fillna("C")
df["Embarked"][df["Embarked"] == "S"] = 1
df["Embarked"][df["Embarked"] == "C"] = 2
df["Embarked"][df["Embarked"] == "Q"] = 3
df["Embarked"] = df["Embarked"].astype(int)

df['Cabin'] = df['Cabin'].fillna('U')
df['Cabin'] = df['Cabin'].map(lambda x: re.compile("([a-zA-Z]+)").search(x).group())
cabin_category = {'A':9, 'B':8, 'C':7, 'D':6, 'E':5, 'F':4, 'G':3, 'T':2, 'U':1}
df['Cabin'] = df['Cabin'].map(cabin_category)


dummy_col=['Title', 'Sex',  'Age_cat', 'SibSp', 'Parch', 'Fare_cat', 'Embarked', 'Pclass', 'FamilySize_cat']
dummy = pd.get_dummies(df[dummy_col], columns=dummy_col, drop_first=False)
df = pd.concat([dummy, df], axis = 1)

df.shape

# Preparing for X_train and y_train
X_train = df[:train.shape[0]]
X_test_fin = df[train.shape[0]:]
y = train.Survived
X_train['Y'] = y
df = X_train
df.head(20) ## DF for Model training

X = df.drop('Y', axis=1)
y = df.Y

# Train Test Split
x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=10)

d_train = xgb.DMatrix(x_train, label=y_train)
d_valid = xgb.DMatrix(x_valid, label=y_valid)
d_test = xgb.DMatrix(X_test_fin)

params = {
        'objective':'binary:logistic',
        'max_depth':4,
        'learning_rate':0.03,
        'eval_metric':'auc',
        'min_child_weight':1,
        'subsample':0.7,
        'colsample_bytree':0.4,
        'seed':29,
        'reg_lambda':2.79,
        'reg_alpha':0.1,
        'gamma':0,
        'scale_pos_weight':1,
        'n_estimators': 600,
        'nthread':-1
}

watchlist = [(d_train, 'train'), (d_valid, 'valid')]
nrounds=10000  
model = xgb.train(params, d_train, nrounds, watchlist, early_stopping_rounds=350, 
                           maximize=True, verbose_eval=10)

fig,ax = plt.subplots(figsize=(15,20))
xgb.plot_importance(model,ax=ax,max_num_features=20,height=0.8,color='g')
#Feature Importance
plt.show()

# Trying RandomForest
y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('predicted.csv', index=False)