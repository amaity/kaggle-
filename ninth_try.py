import os, pickle
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC

#reading the files
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
#print(train.columns)

y = train.Cover_Type
test_id = test['Id']

import os, pickle

if os.path.isfile("/X.pickle"):
    with open( "X.pickle", "rb" ) as fh1:
        X = pickle.load(fh1)
    with open('test.pickle', 'rb') as fh2:
        test = pickle.load(fh2)
else:
    #dropping Soil_Type7 and Soil_Type15
    train = train.drop(['Id','Soil_Type7', 'Soil_Type15'], axis = 1)
    test = test.drop(['Id','Soil_Type7', 'Soil_Type15'], axis = 1)

    #prepare data for training the model
    X = train.drop(['Cover_Type'], axis = 1)

    #reducing Soil_Type cols to single col 
    X = X.iloc[:, :14].join(X.iloc[:, 14:].dot(range(1,39)).to_frame('Soil_Type1'))
    test = test.iloc[:, :14].join(test.iloc[:, 14:].dot(range(1,39)).to_frame('Soil_Type1'))
    #print(X.columns)
    #reducing Wilderness_Area to single col 
    X = X.iloc[:,:10].join(X.iloc[:,10:-1].dot(range(1,5)).to_frame('Wilderness_Area1')).join(X.iloc[:,-1])
    test = test.iloc[:,:10].join(test.iloc[:,10:-1].dot(range(1,5)).to_frame('Wilderness_Area1')).join(test.iloc[:,-1])

    #pickling data for quick access
    with open('X.pickle', 'wb') as fh1:
        pickle.dump(X, fh1)
    with open('test.pickle', 'wb') as fh2:
        pickle.dump(test, fh2)

print(X.columns)

#split data
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

from sklearn.model_selection import GridSearchCV 

# defining parameter range 
param_grid = [  {'kernel': ['linear'], 'C': [0.1, 1, 10, 100]},
                {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}]  
scores = ['precision', 'recall']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(), param_grid, cv=5, verbose=3, scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_val, clf.predict(test)
    print(classification_report(y_true, y_pred))
    print()


""" grid = GridSearchCV(SVC(), param_grid, refit = True, cv=5, verbose = 3)
# fitting the model for grid search
grid.fit(X_train, y_train)
# print best parameter after tuning 
print('Best parameters: ',grid.best_params_)
# print how our model looks after hyper-parameter tuning 
print('Best estimator: ',grid.best_estimator_)
grid_predictions = grid.predict(X_val) 

# print classification report 
print(classification_report(y_val, grid_predictions)) """
