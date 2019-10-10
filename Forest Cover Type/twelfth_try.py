import os, random
import numpy as np 
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import warnings
warnings.simplefilter('ignore')
#------------------------------------------------------------------------------
#import logging
#from heamy.dataset import Dataset
#from heamy.estimator import Classifier
#from heamy.pipeline import ModelsPipeline
#------------------------------------------------------------------------------
train = pd.read_csv("train.csv", index_col='Id')
test = pd.read_csv("test.csv", index_col='Id')
print(train.columns)
#------------------------------------------------------------------------------
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

def preprocessData(train, test):

    y_train = train['Cover_Type']
    
    classes = train.Cover_Type.unique()
    num_classes = len(classes)
    print("There are %i classes: %s " % (num_classes, classes))
    train.drop(['Cover_Type'], axis=1, inplace=True)

    train = addFeatures(train)    
    test = addFeatures(test)

    dtrn_first_ten = train.loc[:,:'Horizontal_Distance_To_Fire_Points']
    dtrn_wa_st = train.loc[:,'Wilderness_Area1':'Soil_Type40']
    dtrn_added_features = train.loc[:,'distance_to_hydrology':]
    dtrn_ = pd.concat([dtrn_first_ten,dtrn_added_features,dtrn_wa_st],axis=1)

    dtst_first_ten = test.loc[:,:'Horizontal_Distance_To_Fire_Points']
    dtst_wa_st = test.loc[:,'Wilderness_Area1':'Soil_Type40']
    dtst_added_features = test.loc[:,'distance_to_hydrology':]
    dtst_ = pd.concat([dtst_first_ten,dtst_added_features,dtst_wa_st],axis=1)
    
    # elevation was found to have very different distributions on test and training sets
    # lets just drop it for now to see if we can implememnt a more robust classifier!
    #train = train.drop('Elevation', axis=1)
    #test = test.drop('Elevation', axis=1)    

    return {'X_train': dtrn_, 'X_test': dtst_, 'y_train': y_train}  #dtrn_, dtst_, y_train
#------------------------------------------------------------------------------
pp = preprocessData(train, test)
X, test, y = pp['X_train'], pp['X_test'], pp['y_train']
#------------------------------------------------------------------------------
weights = [11.393577400361757, 1.4282825089634368, 0.6063107664752647, 1, 1.916980442614397, 
1.0945477432742674, 1.668754279754504, 1.7520168478233817, 8.207420802921982, 
0.7501841943847916, 1.9971420119714571, 2.72057743717325, 2.0, 1.575220244799055, 
2.0695773922466643, 2.536316322049836, 0.46168425088806536, 0.4420755307264942, 
10.660977569012896, 876.0230240897795, 795.52134403456]
#------------------------------------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier
clf1 = KNeighborsClassifier(n_neighbors=1,p=1)

def getWeights(X,test):
    X_copy = X.copy()
    test_copy = test.copy()

    for i in range(19):
        c = X.columns[i]
        X_copy[c] = weights[i]*X_copy[c]
        test_copy[c] = weights[i]*test_copy[c]
    for i in range(19,23):
        c = X.columns[i]
        X_copy[c] = weights[19]*X_copy[c]
        test_copy[c] = weights[19]*test_copy[c]
    for i in range(23,len(X.columns)):
        c = X.columns[i]
        X_copy[c] = weights[20]*X_copy[c]
        test_copy[c] = weights[20]*test_copy[c]
    return X_copy, test_copy
#------------------------------------------------------------------------------
X_copy, test_copy = getWeights(X, test)
#------------------------------------------------------------------------------
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from lightgbm import LGBMClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, \
    AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

##clf1 = KNeighborsClassifier(n_neighbors=1)
##BaggingClassifier(DecisionTreeClassifier(max_leaf_nodes=2000), n_estimators=250,random_state=1)
##AdaBoostClassifier(base_estimator=RandomForestClassifier(), random_state=1)
##KNeighborsClassifier(n_neighbors=1)
clf2 = RandomForestClassifier(n_estimators=300, max_features='sqrt', bootstrap=False,max_depth=60,
                              min_samples_split=2,min_samples_leaf=1,random_state=1)
clf3 = ExtraTreesClassifier(n_estimators=400,max_depth=50,min_samples_split=5,
                             min_samples_leaf=1,max_features=63,random_state=1)
clf4 = LGBMClassifier(objective='multiclass',num_class=7,learning_rate=0.2,random_state=1) #num_leaves=109,
lr = LogisticRegression(multi_class='multinomial', solver='newton-cg', random_state=1)
#------------------------------------------------------------------------------
rf_param = {    
    'n_estimators': [250, 300, 350, 400],
    'max_features': ['auto', 'sqrt'],      
    'max_depth' :  [None, 50, 60, 70, 80, 90], 
    #'min_samples_split' : [2, 5, 10],
    #'min_samples_leaf' : [1, 2, 4],
    #'bootstrap' : [True, False]
    }

et_param = {
    'n_estimators': range(100,501,100),
    'max_features': ['auto','sqrt','log2',None],
    'min_samples_leaf': [1,2,4],
    'min_samples_split': [2,5,10],
    }

lgb_param = {
    'num_leaves': range(31,151,2),
    'learning_rate': [0.005,0.001,0.01,0.2],
    'class_weight': [None, 'balanced'],
    'boosting_type': ['gbdt','dart','goss']
}

lr_param = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
}

def gridSearch(clf,test_params):
    rs = RandomizedSearchCV(estimator=clf, param_distributions=test_params, scoring='accuracy', cv=3, verbose=3)
    rs.fit(X_copy,y)
    print('-'*20)
    print('Best parameters: ',rs.best_params_)
    print('Best score: ',rs.best_score_)
    print('-'*20)

#gridSearch(lr, lr_param)
#------------------------------------------------------------------------------

from mlxtend.classifier import StackingCVClassifier
sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf3, lr],meta_classifier=clf4)

print('5-fold cross validation:\n')

for clf, label in zip([clf1, clf2, clf3, lr, sclf], 
                      ['KNN', 'Random Forest', 'Extra Trees','LogReg','StackingClf']):

    scores = model_selection.cross_val_score(clf, X_copy.values, y.values, cv=5, scoring='accuracy')
    print("Accuracy: %0.3f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
#------------------------------------------------------------------------------
#sclf.fit(X_copy.values,y.values)
#sclf_preds = sclf.predict(test_copy.values)
#print(sclf_preds[:10])
#------------------------------------------------------------------------------
