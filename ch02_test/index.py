import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression

cd = os.getcwd()

train = pd.read_csv(cd + '/input/titanic/train.csv')
test = pd.read_csv(cd + '/input/titanic/test.csv')
gender_submission = pd.read_csv(cd + '/input/titanic/gender_submission.csv')

data = pd.concat([train, test], sort=False)
data["Sex"].replace(["male", "female"], [0, 1], inplace=True)
data['Embarked'].fillna("S", inplace=True)
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)
data["Fare"].fillna(np.mean(data["Fare"]), inplace=True)
age_avg = data['Age'].mean()
age_std = data['Age'].std()

data['Age'].fillna(np.random.randint(age_avg - age_std, age_avg + age_std), inplace=True)

delete_columns = ['Name', 'PassengerId', 'SibSp', 'Parch', 'Ticket', 'Cabin']
data.drop(delete_columns, axis=1, inplace=True)

train = data[:len(train)]
test = data[len(train):]

y_train = train['Survived']
x_train = train.drop('Survived', axis=1)
x_test = test.drop('Survived', axis=1)

clf = LogisticRegression(penalty='l2', solver='sag', random_state=0)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

print(y_pred)
