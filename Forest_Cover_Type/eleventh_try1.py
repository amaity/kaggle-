import os, pickle
import numpy as np 
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics import log_loss

import logging
#------------------------------------------------------------------------------
DATA_DIR = ""
SUBMISSION_FILE = "{0}/sample_submission.csv".format(DATA_DIR)
CACHE=False

NFOLDS = 5
SEED = 13
METRIC = log_loss

ID = 'Id'
TARGET = 'Cover_Type'

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)

np.random.seed(SEED)
#logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.WARNING)
#------------------------------------------------------------------------------
##totally ripped from here;)
##https://www.kaggle.com/justfor/ensembling-and-stacking-with-heamy
""" def addFeatures(df):
    df['HF1'] = df['Horizontal_Distance_To_Hydrology']+df['Horizontal_Distance_To_Fire_Points']
    df['HF2'] = (df['Horizontal_Distance_To_Hydrology']-df['Horizontal_Distance_To_Fire_Points'])
    df['HR1'] = (df['Horizontal_Distance_To_Hydrology']+df['Horizontal_Distance_To_Roadways'])
    df['HR2'] = (df['Horizontal_Distance_To_Hydrology']-df['Horizontal_Distance_To_Roadways'])
    df['FR1'] = (df['Horizontal_Distance_To_Fire_Points']+df['Horizontal_Distance_To_Roadways'])
    df['FR2'] = (df['Horizontal_Distance_To_Fire_Points']-df['Horizontal_Distance_To_Roadways'])
    df['EV1'] = df.Elevation+df.Vertical_Distance_To_Hydrology
    df['EV2'] = df.Elevation-df.Vertical_Distance_To_Hydrology
    df['Mean_HF1'] = df.HF1/2
    df['Mean_HF2'] = df.HF2/2
    df['Mean_HR1'] = df.HR1/2
    df['Mean_HR2'] = df.HR2/2
    df['Mean_FR1'] = df.FR1/2
    df['Mean_FR2'] = df.FR2/2
    df['Mean_EV1'] = df.EV1/2
    df['Mean_EV2'] = df.EV2/2    
    df['Elevation_Vertical'] = df['Elevation']+df['Vertical_Distance_To_Hydrology']    
    df['Neg_Elevation_Vertical'] = df['Elevation']-df['Vertical_Distance_To_Hydrology']
    
    # Given the horizontal & vertical distance to hydrology, 
    # it will be more intuitive to obtain the euclidean distance: sqrt{(verticaldistance)^2 + (horizontaldistance)^2}    
    df['slope_hyd_sqrt'] = (df['Horizontal_Distance_To_Hydrology']**2+df['Vertical_Distance_To_Hydrology']**2)**0.5
    df.slope_hyd_sqrt=df.slope_hyd_sqrt.map(lambda x: 0 if np.isinf(x) else x) # remove infinite value if any
    
    df['slope_hyd2'] = np.sqrt(df['Horizontal_Distance_To_Hydrology']**2+df['Vertical_Distance_To_Hydrology']**2)
    df.slope_hyd2=df.slope_hyd2.map(lambda x: 0 if np.isinf(x) else x) # remove infinite value if any
    
    #Mean distance to Amenities 
    df['Mean_Amenities']=(df.Horizontal_Distance_To_Fire_Points + df.Horizontal_Distance_To_Hydrology + df.Horizontal_Distance_To_Roadways) / 3 
    #Mean Distance to Fire and Water 
    df['Mean_Fire_Hyd1']=(df.Horizontal_Distance_To_Fire_Points + df.Horizontal_Distance_To_Hydrology) / 2
    df['Mean_Fire_Hyd2']=(df.Horizontal_Distance_To_Fire_Points + df.Horizontal_Distance_To_Roadways) / 2
    
    #Shadiness
    df['Shadiness_morn_noon'] = df.Hillshade_9am/(df.Hillshade_Noon+1)
    df['Shadiness_noon_3pm'] = df.Hillshade_Noon/(df.Hillshade_3pm+1)
    df['Shadiness_morn_3'] = df.Hillshade_9am/(df.Hillshade_3pm+1)
    df['Shadiness_morn_avg'] = (df.Hillshade_9am+df.Hillshade_Noon)/2
    df['Shadiness_afternoon'] = (df.Hillshade_Noon+df.Hillshade_3pm)/2
    df['Shadiness_mean_hillshade'] =  (df['Hillshade_9am']  + df['Hillshade_Noon'] + df['Hillshade_3pm'] ) / 3    
    
    # Shade Difference
    df["Hillshade-9_Noon_diff"] = df["Hillshade_9am"] - df["Hillshade_Noon"]
    df["Hillshade-noon_3pm_diff"] = df["Hillshade_Noon"] - df["Hillshade_3pm"]
    df["Hillshade-9am_3pm_diff"] = df["Hillshade_9am"] - df["Hillshade_3pm"]

    # Mountain Trees
    df["Slope*Elevation"] = df["Slope"] * df["Elevation"]
    # Only some trees can grow on steep montain
    
    ### More features
    df['Neg_HorizontalHydrology_HorizontalFire'] = (df['Horizontal_Distance_To_Hydrology']-df['Horizontal_Distance_To_Fire_Points'])
    df['Neg_HorizontalHydrology_HorizontalRoadways'] = (df['Horizontal_Distance_To_Hydrology']-df['Horizontal_Distance_To_Roadways'])
    df['Neg_HorizontalFire_Points_HorizontalRoadways'] = (df['Horizontal_Distance_To_Fire_Points']-df['Horizontal_Distance_To_Roadways'])
    
    df['MeanNeg_Mean_HorizontalHydrology_HorizontalFire'] = (df['Horizontal_Distance_To_Hydrology']-df['Horizontal_Distance_To_Fire_Points'])/2
    df['MeanNeg_HorizontalHydrology_HorizontalRoadways'] = (df['Horizontal_Distance_To_Hydrology']-df['Horizontal_Distance_To_Roadways'])/2
    df['MeanNeg_HorizontalFire_Points_HorizontalRoadways'] = (df['Horizontal_Distance_To_Fire_Points']-df['Horizontal_Distance_To_Roadways'])/2   
        
    df["Vertical_Distance_To_Hydrology"] = abs(df['Vertical_Distance_To_Hydrology'])
    
    df['Neg_Elev_Hyd'] = df.Elevation-df.Horizontal_Distance_To_Hydrology*0.2
    
    # Bin Features
    bin_defs = [
        # col name, bin size, new name
        ('Elevation', 200, 'Binned_Elevation'), # Elevation is different in train vs. test!?
        ('Aspect', 45, 'Binned_Aspect'),
        ('Slope', 6, 'Binned_Slope'),
        ('Horizontal_Distance_To_Hydrology', 140, 'Binned_Horizontal_Distance_To_Hydrology'),
        ('Horizontal_Distance_To_Roadways', 712, 'Binned_Horizontal_Distance_To_Roadways'),
        ('Hillshade_9am', 32, 'Binned_Hillshade_9am'),
        ('Hillshade_Noon', 32, 'Binned_Hillshade_Noon'),
        ('Hillshade_3pm', 32, 'Binned_Hillshade_3pm'),
        ('Horizontal_Distance_To_Fire_Points', 717, 'Binned_Horizontal_Distance_To_Fire_Points')
    ]
    
    for col_name, bin_size, new_name in bin_defs:
        df[new_name] = np.floor(df[col_name]/bin_size)
        
    print('Total number of features : %d' % (df.shape)[1])
    return df """

def addFeatures(df):
    #horizontal and vertical distance to hydrology can be easily combined
    cols = ['Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology']
    df['distance_to_hydrology'] = df[cols].apply(np.linalg.norm, axis=1)
    
    #adding a few combinations of distance features to help enhance the classification
    cols = ['Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points',
            'Horizontal_Distance_To_Hydrology']
    df['distance_mean'] = df[cols].mean(axis=1)
    df['distance_sum'] = df[cols].sum(axis=1)
    df['distance_dif_road_fire'] = df[cols[0]] - df[cols[1]]
    df['distance_dif_hydro_road'] = df[cols[2]] - df[cols[0]]
    df['distance_dif_hydro_fire'] = df[cols[2]] - df[cols[1]]
    
    #taking some factors influencing the amount of radiation
    df['cosine_of_slope'] = np.cos(np.radians(df['Slope']) )
    #X['Diff_azimuth_aspect_9am'] = np.cos(np.radians(123.29-X['Aspect']))
    #X['Diff_azimuth_aspect_12noon'] = np.cos(np.radians(181.65-X['Aspect']))
    #X['Diff_azimuth_aspect_3pm'] = np.cos(np.radians(238.56-X['Aspect']))

    #sum of Hillshades
    shades = ['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']
    #df['Sum_of_shades'] = df[shades].sum(1)
    weights = pd.Series([0.299, 0.587, 0.114], index=cols)
    df['hillshade'] = (df[shades]*weights).sum(1)

    df['elevation_vdh'] = df['Elevation'] - df['Vertical_Distance_To_Hydrology']
    print('Total number of features : %d' % (df.shape)[1])
    return df
#------------------------------------------------------------------------------
def preprocessData():
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    y_train = train[TARGET]
    
    classes = train.Cover_Type.unique()
    num_classes = len(classes)
    print("There are %i classes: %s " % (num_classes, classes))        

    train.drop([ID, TARGET], axis=1, inplace=True)
    test.drop([ID], axis=1, inplace=True)
    
    train = addFeatures(train)    
    test = addFeatures(test)    
    
    cols_to_normalize = [ 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
                       'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
                       'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 
                       'Horizontal_Distance_To_Fire_Points', 'Elevation',
                       #'Shadiness_morn_noon', 'Shadiness_noon_3pm', 'Shadiness_morn_3',
                       #'Shadiness_morn_avg', 
                       #'Shadiness_afternoon', 
                       #'Shadiness_mean_hillshade',
                       #'HF1', 'HF2', 
                       #'HR1', 'HR2', 
                       #'FR1', 'FR2',
                       'distance_to_hydrology','distance_mean','distance_sum',
                       'distance_dif_road_fire','distance_dif_hydro_road','distance_dif_hydro_fire',
                       'cosine_of_slope','hillshade','elevation_vdh',
                       ]

    #train[cols_to_normalize] = normalize(train[cols_to_normalize])
    #test[cols_to_normalize] = normalize(test[cols_to_normalize])

    # elevation was found to have very different distributions on test and training sets
    # lets just drop it for now to see if we can implememnt a more robust classifier!
    #train = train.drop('Elevation', axis=1)
    #test = test.drop('Elevation', axis=1)    
    
    x_train = train
    x_test = test

    return x_train, x_test, y_train
#------------------------------------------------------------------------------
X, test, y = preprocessData()
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
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import warnings
warnings.simplefilter('ignore')

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#------------------------------------------------------------------------------
clf1 = KNeighborsClassifier(n_neighbors=1)
clf2 = RandomForestClassifier(n_estimators=181, max_features='sqrt', random_state=1)
clf3 = ExtraTreesClassifier(n_estimators=400,max_depth=50,min_samples_split=2,random_state=1)
clf4 = LGBMClassifier(num_leaves=62,random_state=1)
clf5 = LogisticRegression(multi_class='multinomial',solver='newton-cg')

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
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print("Accuracy: %0.3f (+/- %0.2f) [%s]" 
          % (scores.mean(), scores.std(), label))
print('-'*20)
