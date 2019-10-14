'''
Some more feature reduction - milking it for all it's worth.
https://rstudio-pubs-static.s3.amazonaws.com/160297_f7bcb8d140b74bd19b758eb328344908.html
https://medium.com/all-things-ai/in-depth-parameter-tuning-for-random-forest-d67bb7e920d
https://medium.com/@plog397/auc-roc-curve-scoring-function-for-multi-class-classification-9822871a6659
'''
import os, pickle
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error

#reading the files
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
#print(train.columns)

#some basic data characteristics
print('Train data shape: ', train.shape)
#print('Train dtypes: ', train.dtypes)
a = (train[train.columns[15:-1]]==1).sum() 
b = (test[test.columns[15:]]==1).sum() 
#print(pd.concat([a.rename('train'),b.rename('test')], axis=1))
#Seems like cols Soil_Type7 and Soil_Type15 can be dropped without much affecting accuracy

c = (train[train.columns[11:15]]==1).sum() 
d = (test[test.columns[11:15]]==1).sum() 
#print(pd.concat([c.rename('train'),d.rename('test')], axis=1))
#The distribution of Wilderness Area appears to be ok.
y = train.Cover_Type
test_id = test['Id']
##----------------------------------------------------------------------
if os.path.isfile("X.pickle"):
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
    print(X.columns)

    #horizontal and vertical distance to hydrology can be easily combined
    cols = ['Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology']
    X['Distance_to_hydrology'] = X[cols].apply(np.linalg.norm, axis=1)
    X = X.drop(cols, axis = 1)
    test['Distance_to_hydrology'] = test[cols].apply(np.linalg.norm, axis=1)
    test = test.drop(cols, axis = 1)

    #shot in the dark - convert like colour tuples to grayscale
    cols = ['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']
    weights = pd.Series([0.299, 0.587, 0.114], index=cols)
    X['Hillshade'] = (X[cols]*weights).sum(1)
    X = X.drop(cols, axis = 1)
    test['Hillshade'] = (test[cols]*weights).sum(1)
    test = test.drop(cols, axis=1)

    #pickling data for quick access
    with open('X.pickle', 'wb') as fh1:
        pickle.dump(X, fh1)
    with open('test.pickle', 'wb') as fh2:
        pickle.dump(test, fh2)
##------------------------------------------------------------------
#split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y)
##------------------------------------------------------------------
#https://stackoverflow.com/questions/31159157/different-result-with-roc-auc-score-and-auc
#https://www.kaggle.com/hadend/tuning-random-forest-parameters
#https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score
def multiclass_roc_auc_score(test, pred, average='macro'):
    lb = LabelBinarizer()
    lb.fit(test)
    test = lb.transform(test)
    pred = lb.transform(pred)
    return roc_auc_score(test, pred, average=average)


#parameter tuning
import matplotlib.pyplot as plt

def plotAUCScore(df_xtr, df_ytr, df_xva, df_yva, parameter, param_vals):
    train_results = []
    test_results = []
    for param in param_vals:
        clf = rf(param)
        clf.fit(df_xtr, df_ytr)
        xpred = clf.predict(df_xtr)
        train_scores = multiclass_roc_auc_score(df_ytr, xpred)
        train_results.append(train_scores)
        ypred = clf.predict(df_xva)
        test_scores = multiclass_roc_auc_score(df_yva, ypred)
        test_results.append(test_scores)
    for line, label in zip([train_results, test_results],['Train AUC', 'Test AUC']):
        plt.plot(param_vals, line, label=label)
    plt.legend()
    plt.ylabel('AUC score')
    plt.xlabel(parameter)
    plt.show()

""" def rf(i): return RandomForestClassifier(n_estimators=i)
plotAUCScore(X_train, y_train, X_val, y_val, 'n_estimators',
             [1, 2, 4, 8, 16, 32, 64, 100, 200])

def rf(i): return RandomForestClassifier(max_depth=i)
plotAUCScore(X_train, y_train, X_val, y_val, 'max_depth',
             np.linspace(1, 32, 32, endpoint=True))

def rf(i): return RandomForestClassifier(min_samples_split=i)
plotAUCScore(X_train, y_train, X_val, y_val, 'min_samples_split',
             np.linspace(0.1, 1.0, 10, endpoint=True))

def rf(i): return RandomForestClassifier(min_samples_leaf=i)
plotAUCScore(X_train, y_train, X_val, y_val, 'min_samples_leaf',
             np.linspace(0.1, 0.5, 5, endpoint=True))

def rf(i): return RandomForestClassifier(max_features=i)
plotAUCScore(X_train, y_train, X_val, y_val, 'max_features',
             list(range(1,X_train.shape[1])) ) """

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train,y_train)
y_pred = model.predict(X_val)
val_mae = mean_absolute_error(y_val,y_pred)
print('Sixth try mae of RFClassifier with base parameters: ', val_mae)
print('Random search','-'*20)
#preparing model
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 200, num = 11)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, 
                    n_iter = 100, cv = 3, verbose=0, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)
print('rf_random.best_params_:',rf_random.best_params_)

def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    return accuracy

print('Base model','-'*20)
base_model = model
base_model.fit(X_train, y_train)
base_accuracy = evaluate(base_model, X_val, y_val)
print('Random search estimator','-'*20)
best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random,  X_val, y_val)

from sklearn.model_selection import GridSearchCV
model = RandomForestClassifier()
params_rf = {
    'bootstrap': [False],
    'n_estimators': [160, 180, 200],
    'max_depth': [40, 50, 70],
    'min_samples_leaf': [1, 2],
    'min_samples_split': [2, 4]
    }
rf_gs = GridSearchCV(model, params_rf, cv=3)
rf_gs.fit(X_train,y_train)
#print('Best score: %0.3f' % rf_gs.best_score_)
#print('Best parameters set: ')
#best_parameters = rf_gs.best_estimator_.get_params()
#for param_name in sorted(params_rf.keys()):
#    print ('\t%s: %r' % (param_name, best_parameters[param_name]) )
#get the error rate
print('Grid search estimator:','-'*20)
print('grid_search.best_params_: ', rf_gs.best_params_)
best_grid = rf_gs.best_estimator_
grid_accuracy = evaluate(best_grid, X_val, y_val)

y_pred = rf_gs.predict(X_val)
val_mae = mean_absolute_error(y_val,y_pred)
print('Sixth try mae with RS RFClassifier: ', val_mae)

test_preds = rf_gs.predict(test)
output = pd.DataFrame({'Id': test_id, 'Cover_Type': test_preds.astype(int)})
#output.to_csv('submission.csv', index=False)
