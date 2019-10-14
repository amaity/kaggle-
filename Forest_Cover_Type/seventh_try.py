'''
Looking for a more intuitive method of parameter tuning.
https://rstudio-pubs-static.s3.amazonaws.com/160297_f7bcb8d140b74bd19b758eb328344908.html
https://medium.com/all-things-ai/in-depth-parameter-tuning-for-random-forest-d67bb7e920d
https://medium.com/@plog397/auc-roc-curve-scoring-function-for-multi-class-classification-9822871a6659
https://www.dlology.com/blog/simple-guide-on-how-to-generate-roc-plot-for-keras-classifier/
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

""" from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_val = sc.transform(X_val) """

param_grid = {"n_estimators":  [int(x) for x in np.linspace(start = 10, stop = 200, num = 11)],
              "max_depth": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None],
              "min_samples_split": np.linspace(0.1, 1.0, 10, endpoint=True), #np.arange(1,150,1),
              "min_samples_leaf": np.linspace(0.1, 0.5, 5, endpoint=True),  #np.arange(1,60,1),
              "max_leaf_nodes": np.arange(2,60,1),
              "min_weight_fraction_leaf": np.arange(0.1,0.4, 0.1)}

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score
def multiclass_roc_auc_score(test, pred, average='micro'):
    lb = LabelBinarizer()
    lb.fit(test)
    test = lb.transform(test)
    pred = lb.transform(pred)
    return roc_auc_score(test, pred, average=average)


#parameter tuning
import matplotlib.pyplot as plt
from collections import defaultdict
import warnings
clf = RandomForestClassifier(random_state=1)

def evaluate_param(clf, param_grid, metric, metric_abv):
    data = []
    for parameter, values in dict.items(param_grid):
        for value in values:
            d = {parameter:value}
            warnings.filterwarnings('ignore') 
            clf = RandomForestClassifier(**d)
            clf.fit(X_train, y_train)
            x_pred = clf.predict(X_train)
            train_score = metric(y_train, x_pred)
            y_pred = clf.predict(X_val)
            test_score = metric(y_val, y_pred)
            data.append({'Parameter':parameter, 'Param_value':value, 
            'Train_'+metric_abv:train_score, 'Test_'+metric_abv:test_score})
    df = pd.DataFrame(data)
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10,5))
    for (parameter, group), ax in zip(df.groupby(df.Parameter), axes.flatten()):
        group.plot(x='Param_value', y=(['Train_'+metric_abv,'Test_'+metric_abv]),
        kind='line', ax=ax, title=parameter)
        ax.set_xlabel('')
    plt.tight_layout()
    plt.show()

#evaluate_param(clf, param_grid, multiclass_roc_auc_score, 'AUC')
#evaluate_param(clf, param_grid, mean_absolute_error, 'MAE')

from sklearn.model_selection import GridSearchCV


""" def evaluate_gs(param_grid, metric, metric_abv):
    data = []
    for parameter, values in dict.items(param_grid):
        for value in values:
            d = {parameter:value}
            warnings.filterwarnings('ignore') 
            gs = GridSearchCV(clf, param_grid = {parameter: num_range})
            gs.fit(X_train, y_train)
            x_pred = gs.predict(X_train)
            train_score = metric(y_train, x_pred)
            y_pred = gs.predict(X_val)
            test_score = metric(y_val, y_pred)
            data.append({'Parameter':parameter, 'Param_value':value, 
            'Train_'+metric_abv:train_score, 'Test_'+metric_abv:test_score})
    df = pd.DataFrame(data)
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(10,5))
    for (parameter, group), ax in zip(df.groupby(df.Parameter), axes.flatten()):
        group.plot(x='Param_value', y=(['Train_'+metric_abv,'Test_'+metric_abv]),
        kind='line', ax=ax, title=parameter)
        ax.set_xlabel('')
    plt.tight_layout()
    plt.show() """


param_grid2 = {"n_estimators": [10,18,21],
                'max_leaf_nodes': [150,None],
                'max_depth': [None],
                'min_samples_split': [2, 5], 
                'min_samples_leaf': [1, 2],
              "max_features": ['auto','sqrt'],
              "bootstrap": [True, False]}

from operator import itemgetter

""" # Utility function to report best scores
def report(grid_scores, n_top):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.4f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(clf, param_grid=param_grid2)
grid_search.fit(X_train, y_train)
report(grid_search.grid_scores_, 4) """

""" grid_search = GridSearchCV(clf, param_grid=param_grid2, cv=8,
                            scoring='accuracy')
gs_result = grid_search.fit(X_train, y_train)
print(gs_result.best_params_)
best_clf = RandomForestClassifier(gs_result.best_params_)
y_pred = gs_result.predict(X_val)
val_mae = mean_absolute_error(y_val, y_pred)
print('Seventh try mae: ', val_mae) """

""" from sklearn.model_selection import RandomizedSearchCV
clf_random = RandomizedSearchCV(estimator = clf, param_distributions = param_grid2, 
                                n_iter = 100, cv = 3, verbose=0, random_state=42)
clf_random.fit(X_train, y_train)
print('Random search best parameters:',clf_random.best_params_)
best_random = clf_random.best_estimator_
y_pred = best_random.predict(X_val)
val_mae = mean_absolute_error(y_val,y_pred)
print('Seventh try mae: ', val_mae) """

""" def plotAUCScore(df_xtr, df_ytr, df_xva, df_yva, parameter, param_vals):
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
    plt.show() """


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


 
grid = GridSearchCV(clf, param_grid2, refit = True, cv=5, verbose = 3)
grid.fit(X_train, y_train)
# print best parameter after tuning 
print('Best parameters: ',grid.best_params_)
# print how our model looks after hyper-parameter tuning 
print('Best estimator: ',grid.best_estimator_)
grid_predictions = grid.predict(X_val) 

from sklearn.metrics import classification_report
print(classification_report(y_val, grid_predictions))