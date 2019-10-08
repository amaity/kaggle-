'''
Some more feature reduction - milking it for all it's worth.
http://rasbt.github.io/mlxtend/user_guide/classifier/StackingClassifier/
https://stackoverflow.com/questions/54120935/my-stackingcvclassifier-has-lower-accuracy-than-base-classifiers-yet-does-very-w
https://www.kaggle.com/kwabenantim/forest-cover-stacking-multiple-classifiers
https://www.analyticsindiamag.com/7-types-classification-algorithms/
'''
import os
import numpy as np 
import pandas as pd 

#reading the files
train = pd.read_csv("train.csv", index_col='Id')
test = pd.read_csv("test.csv", index_col='Id')
#print(train.columns)

##PREPROCESSING----------------------------------------------

#dropping Soil_Type7 and Soil_Type15
train = train.drop(['Soil_Type7', 'Soil_Type15'], axis = 1)
test = test.drop(['Soil_Type7', 'Soil_Type15'], axis = 1)
test_id = test.index.copy()

#prepare data for training the model
X = train.drop(['Cover_Type'], axis = 1)
y = train.Cover_Type

#reducing Soil_Type cols to single col 
X = X.iloc[:, :14].join(X.iloc[:, 14:].dot(range(1,39)).to_frame('Soil_Type1'))
test = test.iloc[:, :14].join(test.iloc[:, 14:].dot(range(1,39)).to_frame('Soil_Type1'))
#print(X.columns)
#reducing Wilderness_Area to single col 
X = X.iloc[:,:10].join(X.iloc[:,10:-1].dot(range(1,5)).to_frame('Wilderness_Area1')).join(X.iloc[:,-1])
test = test.iloc[:,:10].join(test.iloc[:,10:-1].dot(range(1,5)).to_frame('Wilderness_Area1')).join(test.iloc[:,-1])
print(X.columns)

#horizontal and vertical distance to hydrology can be easily combined
cols = ['Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology']
X['Distance_to_hydrology'] = X[cols].apply(np.linalg.norm, axis=1)
X = X.drop(cols, axis = 1)
test['Distance_to_hydrology'] = test[cols].apply(np.linalg.norm, axis=1)
test = test.drop(cols, axis = 1)

#another shot in the dark - convert like colour tuples to grayscale
cols = ['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']
weights = pd.Series([0.299, 0.587, 0.114], index=cols)
X['Hillshade'] = (X[cols]*weights).sum(1)
X = X.drop(cols, axis = 1)
test['Hillshade'] = (test[cols]*weights).sum(1)
test = test.drop(cols, axis=1)

""" #split data
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y) """

##--------------------------------------------------------------------

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mlxtend.classifier import StackingCVClassifier
from sklearn.model_selection import cross_val_score
import warnings
warnings.simplefilter('ignore')

RANDOM_SEED = 12
np.random.seed(RANDOM_SEED)
clf1 = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                             metric_params=None, n_jobs=1, n_neighbors=5, p=2, 
                             weights='uniform')
clf2 = RandomForestClassifier(n_estimators=75, random_state=RANDOM_SEED)
clf3 = LinearDiscriminantAnalysis()
lr = LogisticRegression()
sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf3], 
                            meta_classifier=lr)

print('3-fold cross validation:\n')

for clf, label in zip([clf1, clf2, clf3, sclf], 
                      ['KNN', 
                       'Random Forest',
                       'LinearDiscriminantAnalysis',
                       'StackingClassifier']):

    scores = cross_val_score(clf, X.values, y.values,
                            cv=5, scoring='accuracy')

    print("Accuracy: %0.2f (+/- %0.2f) [%s]" 
          % (scores.mean(), scores.std(), label))

sclf = sclf.fit(X.values, y.values)

test_preds = sclf.predict(test)
output = pd.DataFrame({'Id': test_id, 'Cover_Type': test_preds})
#output.to_csv('submission.csv', index=False)
