'''
Some more feature reduction - milking it for all it's worth.
https://rstudio-pubs-static.s3.amazonaws.com/160297_f7bcb8d140b74bd19b758eb328344908.html
https://medium.com/all-things-ai/in-depth-parameter-tuning-for-random-forest-d67bb7e920d
https://medium.com/@plog397/auc-roc-curve-scoring-function-for-multi-class-classification-9822871a6659
'''
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

##----------------------------------------------------------------------
#dropping Soil_Type7 and Soil_Type15
train = train.drop(['Id','Soil_Type7', 'Soil_Type15'], axis = 1)
test_id = test['Id']
test = test.drop(['Id','Soil_Type7', 'Soil_Type15'], axis = 1)

#prepare data for training the model
X = train.drop(['Cover_Type'], axis = 1)
y = train.Cover_Type

def reduceToColIndex(df, col_name, head, n_cols, end):
    df_ = df.iloc[:, :head].join(df.iloc[:,head:] \
                        .dot(range(1,n_cols)).to_frame(col_name)) \
                        .join(df.iloc[:,end])
    return df_

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

""" #shot in the dark - perhaps the one that is nearer (fire-point/roadway) has more influence 
cols = ['Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Fire_Points']
X['Nearest_firept_roadway'] = X[cols].max(axis=1)
X = X.drop(cols, axis = 1)
test['Nearest_firept_roadway'] = test[cols].max(axis=1)
test = test.drop(cols, axis = 1) """

#another shot in the dark - convert like colour tuples to grayscale
cols = ['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']
weights = pd.Series([0.299, 0.587, 0.114], index=cols)
X['Hillshade'] = (X[cols]*weights).sum(1)
X = X.drop(cols, axis = 1)
test['Hillshade'] = (test[cols]*weights).sum(1)
test = test.drop(cols, axis=1)

##------------------------------------------------------------------
#split data
X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                      test_size=0.2, stratify=y)
##------------------------------------------------------------------

#preparing model
from sklearn.model_selection import GridSearchCV
model = RandomForestClassifier()
params_rf = {'n_estimators': [10, 50, 75]}
rf_gs = GridSearchCV(model, params_rf, cv=5)
rf_gs.fit(X_train,y_train)
print('Best score: %0.3f' % rf_gs.best_score_)
print('Best parameters set: ')
best_parameters = rf_gs.best_estimator_.get_params()
for param_name in sorted(params_rf.keys()):
    print ('\t%s: %r' % (param_name, best_parameters[param_name]) )
#get the error rate
y_pred = rf_gs.predict(X_val)
val_mae = mean_absolute_error(y_val,y_pred)
print('Fifth try mae with RFClassifier: ', val_mae)
#Fifth try mae with RFClassifier(added dist to hydrology):  0.32275132275132273
#Fifth try mae with RFClassifier(added min of firept/roadway):  0.4451058201058201
#Fifth try mae with RFClassifier(added max of firept/roadway):  0.4341931216931217
#Fifth try mae with RFClassifier(reduced hillshade tuple):  0.3148148148148148
#Fifth try mae with RFClassifier(corrected soil_type and wilderness area reduction):  0.3253968253968254

parameters = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': np.linspace(1, 32, 8, endpoint=True),
    'min_samples_split': np.linspace(0.1, 1.0, 5, endpoint=True),
    'min_samples_leaf': np.linspace(0.1, 0.5, 5, endpoint=True),
    'max_features': list(range(1,X_train.shape[1]))
    }
grid_search = GridSearchCV(model, parameters, scoring='accuracy')
grid_search.fit(X_train, y_train)
print('Best score: %0.3f' % grid_search.best_score_ )
print('Best parameters set: ')
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print ('\t%s: %r' % (param_name, best_parameters[param_name]) )

predictions = grid_search.predict(X_val)
print (classification_report(y_test, predictions) )


""" test_preds = rf_gs.predict(test)
output = pd.DataFrame({'Id': test_id, 'Cover_Type': test_preds.astype(int)})
output.to_csv('submission.csv', index=False) """
