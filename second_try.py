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

#some basic data characteristics
#print('Shape: ', test.shape)
#print('Train dtypes: ', train.dtypes)
#print('Num of zeros: ', (test[test.columns[15:-1]]==0).sum() )

#brute force method to compress 40 columns of Soil_Type into one column
inp = train.iloc[:,15:-2].to_string(header=False, index=False, index_names=False).split('\n')
vals = [int(''.join(ele.split()),2) for ele in inp]
train['Soil_Type'] = vals
train.drop(train.iloc[:,15:-2], axis=1, inplace=True)
inp2 = train.iloc[:,11:15].to_string(header=False, index=False, index_names=False).split('\n')
vals2 = [int(''.join(ele.split()),2) for ele in inp2]
train['Wilderness_Area'] = vals2
train.drop(train.iloc[:,11:15], axis=1, inplace=True)
#train['Soil_Type'] = (train['Soil_Type']-train['Soil_Type'].min())/(train['Soil_Type'].max()-train['Soil_Type'].min())
#print(train.head())

#prepare data for training the model
X = train.drop(['Cover_Type'], axis = 1)
y = train.Cover_Type

#split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

#preparing model
forest_model = RandomForestClassifier(n_estimators=10)
forest_model.fit(X_train,y_train)

#get the error rate
val_predictions = forest_model.predict(X_val)
val_mae = mean_absolute_error(y_val,val_predictions)
print('Second try mae with RFClassifier: ', val_mae)
#Second try mae with RFClassifier:  0.3472222222222222
#Second try mae with RFClassifier:  0.373015873015873 (without normalising Soil_Type column)
#Second try mae with RFClassifier:  0.3869047619047619 (with norm of Soil....)
#Second try mae with RFClassifier:  0.34953703703703703

#ensureing test data has the same structure at the train data
test['Soil_Type'] = test[test.columns[15:-1]].astype(str).apply(''.join,1).apply(int, base=2)
test.drop(test.iloc[:,15:-1], axis=1, inplace=True)
test['Wilderness_Area'] = test[test.columns[11:15]].apply(lambda x: int(''.join(x.astype(str)),2), axis=1)
test.drop(test.iloc[:,11:15], axis=1, inplace=True)
#test['Soil_Type'] = (test['Soil_Type']-test['Soil_Type'].min())/(test['Soil_Type'].max()-test['Soil_Type'].min())
#print(test.head() )

#making the predictions
test_preds = forest_model.predict(test)
output = pd.DataFrame({'Id': test.Id, 'Cover_Type': test_preds.astype(int)})
output.to_csv('submission.csv', index=False)
