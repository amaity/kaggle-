'''
Hi! Trying to reduce the number of features using plain pandas
'''
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error

#reading the files
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
print(train.columns)

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

#dropping Soil_Type7 and Soil_Type15
train = train.drop(['Id','Soil_Type7', 'Soil_Type15'], axis = 1)
test_id = test['Id']
test = test.drop(['Id','Soil_Type7', 'Soil_Type15'], axis = 1)

#prepare data for training the model
X = train.drop(['Cover_Type'], axis = 1)
y = train.Cover_Type

#function for compressing the feature cols
def compressCols(df, col_name, start, end):
    df[col_name] = df[df.columns[start:end]].astype(str).apply(''.join,1).apply(int, base=2)
    df.drop(df.iloc[:,start:end], axis=1, inplace=True)
    return df

""" #compressing Soil_Type cols
train = compressCols(train, 'Soil_Type', 15, -2)
test = compressCols(test, 'Soil_Type', 15, -1)
print(test.columns) """

def reduceToColIndex(df, col_name, head, mid, end):
    df_ = df.iloc[:, :head].join(df.iloc[:,head:end] \
                          .dot(range(1,mid)).to_frame(col_name)) \
                          .join(df.iloc[:,end])
    return df_

#reducing Soil_Type cols to col index
X = reduceToColIndex(X, 'Soil_Type1', 14, 38, -1)
test = reduceToColIndex(test, 'Soil_Type1', 14, 38, -1)

#split data
X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                      test_size=0.2, stratify=y)

#preparing model
from sklearn.model_selection import GridSearchCV
model = RandomForestClassifier()
params_rf = {'n_estimators': [10, 50, 75]}
rf_gs = GridSearchCV(model, params_rf, cv=8)
rf_gs.fit(X_train,y_train)

#get the error rate
val_predictions = rf_gs.predict(X_val)
val_mae = mean_absolute_error(y_val,val_predictions)
print('Fourth try mae with RFClassifier(w/o cross-val): ', val_mae)
#Fourth try mae with RFClassifier(w/o cross-val):  0.381283068783


test_preds = rf_gs.predict(test)
output = pd.DataFrame({'Id': test_id, 'Cover_Type': test_preds.astype(int)})
output.to_csv('submission.csv', index=False)
