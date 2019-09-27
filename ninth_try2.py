import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
import time
SEED = 123

#reading the files
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
#print(train.columns)

y = train.Cover_Type
test_id = test['Id']

#dropping Ids
train = train.drop(['Id'], axis = 1)
test = test.drop(['Id'], axis = 1)

#prepare data for training the model
X = train.drop(['Cover_Type'], axis = 1)
print(X.columns[(X < 0).any()])

clf = RandomForestClassifier(random_state=SEED)
clf = clf.fit(X,y)

features = pd.DataFrame({'Features': X.columns, 
                         'Importances': clf.feature_importances_})
features.sort_values(by=['Importances'], axis='index', ascending=False, inplace=True)
plt.figure(figsize=(12,4))
sns.barplot(x='Features', y='Importances', data=features)
plt.xticks(rotation='vertical')
#plt.show()

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_val)
print('-'*20)
print('Acc score before preprocessing: ',accuracy_score(y_val, y_pred))

#PREPROCESS-------------------------------------------------------------------

def preprocess(df):
    #horizontal and vertical distance to hydrology can be easily combined
    cols = ['Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology']
    df['Distance_to_hydrology'] = df[cols].apply(np.linalg.norm, axis=1)
    
    #adding a few combinations of distance features to help enhance the classification
    cols = ['Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points',
            'Horizontal_Distance_To_Hydrology']
    df['distance_mean'] = df[cols].mean(axis=1)
    df['distance_sum'] = df[cols].sum(axis=1)
    df['distance_dif_road_fire'] = df[cols[0]] - df[cols[1]]
    df['distance_dif_hydro_road'] = df[cols[2]] - df[cols[0]]
    df['distance_dif_hydro_fire'] = df[cols[2]] - df[cols[1]]
    
    #taking some factors influencing the amount of radiation
    df['Cosine_of_slope'] = np.cos(np.radians(df['Slope']) )
    #X['Diff_azimuth_aspect_9am'] = np.cos(np.radians(123.29-X['Aspect']))
    #X['Diff_azimuth_aspect_12noon'] = np.cos(np.radians(181.65-X['Aspect']))
    #X['Diff_azimuth_aspect_3pm'] = np.cos(np.radians(238.56-X['Aspect']))

    #sum of Hillshades
    shades = ['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']
    #df['Sum_of_shades'] = df[shades].sum(1)
    weights = pd.Series([0.299, 0.587, 0.114], index=cols)
    df['Hillshade'] = (df[shades]*weights).sum(1)

    df['Elevation_VDH'] = df['Elevation'] - df['Vertical_Distance_To_Hydrology']
    return df

X = preprocess(X)
test = preprocess(test)
print(X.columns)

#print(X.loc[(X==0).any(axis=1)].columns)
#------------------------------------------------------------------------------

#X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)
#clf.fit(X_train, y_train)
#y_pred = clf.predict(X_val)
#print('Acc score after preprocessing: ',accuracy_score(y_val, y_pred))
#print('-'*20)

from mlxtend.classifier import StackingCVClassifier
import warnings
warnings.simplefilter('ignore')

clf1 = KNeighborsClassifier(n_neighbors=1)
clf2 = GaussianNB()
lr = LogisticRegression()

# Starting from v0.16.0, StackingCVRegressor supports
# `random_state` to get deterministic result.
sclf = StackingCVClassifier(classifiers=[clf, clf1, clf2],
                            meta_classifier=lr)

print('3-fold cross validation:\n')

for clf, label in zip([clf, clf1, clf2, sclf], 
                      ['Random Forest', 
                       'KNN', 
                       'Naive Bayes',
                       'StackingClassifier']):

    scores = cross_val_score(clf, X.values, y.values, cv=3, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" 
          % (scores.mean(), scores.std(), label))