import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

test=pd.read_csv('test.csv')
train=pd.read_csv('train.csv')

#y=train['Survived'].reset_index(drop=True)
#y=train.iloc[:,[1]].values
y=train.Survived

train_features = train.drop(['Survived'], axis=1)
test_features = test
features = pd.concat([train_features, test_features]).reset_index(drop=True)

# since the feature Name,ticket, Cabin as no contribution on the probability to survive, we can drop both the
# feature

features= features.drop(['Name'],axis=1)
features= features.drop(['Ticket'],axis=1)
features= features.drop(['Cabin'],axis=1)


features.isna().sum()

from sklearn.preprocessing import Imputer
imp = Imputer()
features.iloc[:, [3,6]] = imp.fit_transform(features.iloc[:,[3,6]].values)

features['Embarked'] = features['Embarked'].fillna('S')

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
features['Sex']=le.fit_transform(features['Sex'])
features['Embarked']=le.fit_transform(features['Embarked'])

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
features = one.fit_transform(features).toarray()
features = pd.DataFrame(list(features))

from sklearn.preprocessing import StandardScaler
sd =StandardScaler()
features = sd.fit_transform(features)
features= pd.DataFrame(list(features))

# dividing the training and test sets
train = features.iloc[:len(y),:].values
test = features.iloc[len(y):,:].values

#Selecting the best classification model:

# Logistic Regression

from sklearn.linear_model import LogisticRegression
lin_reg = LogisticRegression()
lin_reg.fit(train,y)

lin_reg.score(train,y)

y_pred = lin_reg.predict(train)

test_y = lin_reg.predict(test)
lin_reg.score(test,test_y)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred)

from sklearn.metrics import precision_score, recall_score, f1_score
precision_score(y, y_pred)
recall_score(y, y_pred)
f1_score(y, y_pred)

# KNN ( K nearest Neighbours)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(train, y)
y_pred = knn.predict(train)

knn.score(train,y)

y_test = knn.predict(test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred)

from sklearn.metrics import precision_score, recall_score, f1_score
precision_score(y, y_pred)
recall_score(y, y_pred)
f1_score(y, y_pred)

#Gaussian Naive
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(train, y)

y_pred = nb.predict(train)

nb.score(train,y)

y_test = nb.predict(test)

nb.score(test,y_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, y_pred)

from sklearn.metrics import precision_score, recall_score, f1_score
precision_score(y, y_pred)
recall_score(y, y_pred)
f1_score(y, y_pred)

#Decision Tree

from sklearn.tree import DecisionTreeClassifier
dtf = DecisionTreeClassifier(criterion = 'entropy', max_depth = 11)
dtf.fit(train,y)
dtf.score(train,y)

y_pred = dtf.predict(test)
dtf.score(test,y_pred)

# support Vector Machine
from sklearn.svm import SVC
svm = SVC()
svm.fit(train,y)
svm.score(train,y)
y_test = svm.predict(test)

svm.score(test,y_test)



