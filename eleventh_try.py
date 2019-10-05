import os, pickle
import numpy as np 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

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

clf = RandomForestClassifier(random_state=1)
clf = clf.fit(X,y)

def plotImpFeatures(X):
    features = pd.DataFrame({'Features': X.columns, 
                         'Importances': clf.feature_importances_})
    features.sort_values(by=['Importances'], axis='index', ascending=False, inplace=True)
    plt.figure(figsize=(12,4))
    sns.barplot(x='Features', y='Importances', data=features)
    plt.xticks(rotation='vertical')
    plt.show()

#plotImpFeatures(X)
#------------------------------------------------------------------------------
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

    #sum of Hillshades
    shades = ['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']
    #df['Sum_of_shades'] = df[shades].sum(1)
    #df['Mean_of_shades'] = df[shades].mean(1)
    weights = pd.Series([0.299, 0.587, 0.114], index=cols)
    df['Hillshade'] = (df[shades]*weights).sum(1)

    df['Elevation_VDH'] = df['Elevation'] - df['Vertical_Distance_To_Hydrology']
    df['Elev_VDH_Sum'] = df['Elevation'] + df['Vertical_Distance_To_Hydrology']
    #df['ElevationSq'] = df['Elevation']**2
    #df['ElevationLog'] = np.log1p(df['Elevation'])
    return df

X = preprocess(X)
test = preprocess(test)
#------------------------------------------------------------------------------
def drop_unimportant(df):
    df_ = df.copy()
    n_rows = df_.shape[0]
    hi_freq_cols = []
    for col in X.columns:
        mode_frequency = 100.0 * df_[col].value_counts().iat[0] / n_rows 
        if mode_frequency > 99.0:
            hi_freq_cols.append(col)
    df_ = df_.drop(hi_freq_cols, axis='columns')
    return df_

X = drop_unimportant(X)
feature_names = list(X.columns)
test = test[feature_names]
print(X.shape)
print(test.shape)
#------------------------------------------------------------------------------
from sklearn import model_selection
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import warnings
warnings.simplefilter('ignore')

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#------------------------------------------------------------------------------
clf1 = KNeighborsClassifier(n_neighbors=1)
clf2 = RandomForestClassifier(n_estimators=181, max_features='sqrt', random_state=1)
clf3 = ExtraTreesClassifier(n_estimators=400,max_depth=50,min_samples_split=2,random_state=1)
clf4 = LGBMClassifier(num_leaves=100,random_state=1)

#------------------------------------------------------------------------------

param_test = {    
    'n_estimators': [100, 125, 150, 180, 200, 250],
    'max_features': ['auto', 'sqrt'],      
    'max_depth' :  [None, 50, 60, 70, 80, 90, 100], 
    'min_samples_split' : [2, 5, 10],
    'min_samples_leaf' : [1, 2, 4],
    'bootstrap' : [True, False]
    }

def gridSearch(clf,test_params):
    rs = RandomizedSearchCV(estimator=clf, param_distributions=test_params, scoring='accuracy', cv=3, verbose=3)
    rs.fit(X,y)
    print('-'*20)
    print('Best parameters: ',rs.best_params_)
    print('Best score: ',rs.best_score_)
    print('-'*20)

#gridSearch(clf2, param_test)
#------------------------------------------------------------------------------

print('-'*20)
from mlxtend.classifier import EnsembleVoteClassifier
eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3, clf4], weights=[1,1,1,1])
labels = ['KNeighbors', 'Random Forest', 'Extra Trees', 'LGBM', 'Ensemble']
for clf, label in zip([clf1, clf2, clf3, clf4, eclf], labels):
    scores = model_selection.cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print("Accuracy: %0.3f (+/- %0.2f) [%s]" 
          % (scores.mean(), scores.std(), label))
print('-'*20)
