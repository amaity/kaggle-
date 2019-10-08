'''
Hi! Still trying to reduce the number of features using plain pandas
'''
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error

#reading the files
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#some basic data characteristics
#print('Shape: ', test.shape)
#print('Train columns: ', train.columns)
#print('Test columns: ', test.columns )

#using position based indexing to compress 40 columns of Soil_Type into one column
train = train.iloc[:, :15].join(train.iloc[:,15:-2] \
                          .dot(range(1,40)).to_frame('Soil_Type1')) \
                          .join(train.iloc[:,-1])
#print(train.columns)
#print(train.shape)

train = train.iloc[:, :11].join(train.iloc[:,11:-2] \
                          .dot(range(1,5)).to_frame('Wilderness_Area1')) \
                          .join(train.iloc[:,-2:])
#print(train.columns)

#prepare data for training the model
X = train.drop(['Cover_Type'], axis = 1)
y = train.Cover_Type

#split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

#preparing model
#model = RandomForestRegressor(n_estimators=10, random_state=0) #DecisionTreeRegressor(random_state=0)
model = RandomForestClassifier(n_estimators=10)
model.fit(X_train,y_train)

#get the error rate
val_predictions = model.predict(X_val)
val_mae = mean_absolute_error(y_val,val_predictions)
print('Third try mae with RFClassifier(w/o cross-val): ', val_mae)
#Third try mae (w/o cross-val):  0.5211640211640212
#Third try mae with DeciTree(w/o cross-val):  0.5380291005291006
#Third try mae with RandForest(w/o cross-val):  0.5768849206349206
#Third try mae with RFClassifier(w/o cross-val):  0.415013227513
#This attempt did slightly better than the previous try. Will cross-validation help? Let's see.

""" from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score

my_pipeline = Pipeline(steps=[
    ('preprocessor', SimpleImputer()),
    ('model', RandomForestRegressor(n_estimators=10, random_state=0))
])

scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')
print("Third try, Average MAE score with cv: ", scores.mean()) """


#ensureing test data has the same structure at the train data
test = test.iloc[:, :15].join(test.iloc[:,15:-1].dot(range(1,40)).to_frame('Soil_Type1')) 
#print(test.columns)
#print(test.shape)
test = test.iloc[:, :11].join(test.iloc[:,11:-1] \
                          .dot(range(1,5)).to_frame('Wilderness_Area1')) \
                          .join(test.iloc[:,-1])
#print(test.columns)

#making the predictions
#my_pipeline.fit(X, y)
#test_preds = my_pipeline.predict(test)
test_preds = model.predict(test)
output = pd.DataFrame({'Id': test.Id, 'Cover_Type': test_preds.astype(int)})
output.to_csv('submission.csv', index=False)

