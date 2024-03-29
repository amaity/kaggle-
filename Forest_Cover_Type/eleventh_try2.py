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

    return dtrn_, dtst_, y_train
#------------------------------------------------------------------------------
X, test, y = preprocessData(train, test)
#------------------------------------------------------------------------------
#blind copy from here: https://www.kaggle.com/chrisfreiling/nearest-neighbor-kicks-ass
def getWeights(X_full, y_full):
    cols = list(X_full.columns.values)
    #starting weights
    weights = [14,1,1,1,1,1,1,1,7,1,2,2,2,2,2,3,1,1,10,800,800]
    
    best_score_ever=0
    best_ever_wts = [i for i in weights]
    lr = 0.5
    
    for step in range(100):        
        X_full_copy = X_full.copy()

        for i in range(19):
            c = X_full.columns[i]
            X_full_copy[c] *= weights[i]
        for i in range(19,23):
            c = X_full.columns[i]
            X_full_copy[c] *= weights[19]
        for i in range(23,len(X_full.columns)):
            c = X_full.columns[i]
            X_full_copy[c] *= weights[20]

        r = lr*random.random()
        
        
        # Choose a random weight to change
        train_index = random.randint(0,20)
        train_col = X_full.columns[train_index]
        
        # We will test four factors for changing the current weight.
        factors = [1-r,1,1+r, 1+2*r]        
        wts = [weights[train_index] * f for f in factors]

        best_score=0
        best_wt=-1

        for wt in wts:  
            if train_index<19:            
                X_full_copy[train_col] = wt * X_full[train_col]
            if train_index==19: 
                for i in range(19,23):
                    c = X_full.columns[i]
                    X_full_copy[c] = wt*X_full[c]
            if train_index > 19: 
                for i in range(23,len(X_full.columns)):
                    c = X_full.columns[i]
                    X_full_copy[c] = wt*X_full[c]

            model = NearestNeighbors(n_neighbors=2, algorithm='ball_tree')
            distances, indices  = model.fit(X_full_copy).kneighbors(X_full_copy)
            
            # we use the second best because the first best is itself.
            second_best = indices[:,1]
            labels = y_full.tolist()
            my_labels = [labels[i] for i in second_best]
            score = accuracy_score(labels, my_labels)        

            if score > best_score_ever :
                best_score_ever = score
                best_ever_wts = [i for i in weights]
                best_ever_wts[train_index] = wt
                print("\tnew best ever:",best_score_ever)
            if score > best_score:
                best_wt=wt
                best_score=score

        old_wt =  weights[train_index] 
        
        # Notice that I only go half-way to the new weights.  Just seemed like a good idea, but not sure.
        weights[train_index] =  (weights[train_index]+best_wt)/2
        
        print("step",step,"col",train_index,"best",round(old_wt,2),"->",round(best_wt,2), "\t",best_score) 
    return best_ever_wts, weights

#best_weights, final_weights = getWeights(X, y)
#print('Final weights: ',final_weights)
#weights = final_weights
weights = [11.393577400361757, 1.4282825089634368, 0.6063107664752647, 1, 1.916980442614397, 
1.0945477432742674, 1.668754279754504, 1.7520168478233817, 8.207420802921982, 
0.7501841943847916, 1.9971420119714571, 2.72057743717325, 2.0, 1.575220244799055, 
2.0695773922466643, 2.536316322049836, 0.46168425088806536, 0.4420755307264942, 
10.660977569012896, 876.0230240897795, 795.52134403456]
#------------------------------------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier
clf1 = KNeighborsClassifier(n_neighbors=1,p=1)

print(X.shape)
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

clf1.fit(X_copy, y)
clf1_preds = clf1.predict(test_copy)

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
clf4 = LGBMClassifier(num_leaves=109,objective='multiclass',num_class=7,
                       learning_rate=0.2,random_state=1)
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

def gridSearch(clf,test_params):
    rs = RandomizedSearchCV(estimator=clf, param_distributions=test_params, scoring='accuracy', cv=3, verbose=3)
    rs.fit(X_copy,y)
    print('-'*20)
    print('Best parameters: ',rs.best_params_)
    print('Best score: ',rs.best_score_)
    print('-'*20)

#gridSearch(clf4, lgb_param)
#------------------------------------------------------------------------------
print('-'*20)
from mlxtend.classifier import EnsembleVoteClassifier
eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3, clf4])
labels = ['KNeighbors', 'Random Forest', 'Extra Trees', 'LGBM', 'Ensemble']
for clf, label in zip([clf1, clf2, clf3, clf4, eclf], labels):
    scores = cross_val_score(clf, X_copy, y, cv=5, scoring='accuracy')
    print("Accuracy: %0.3f (+/- %0.2f) [%s]" 
          % (scores.mean(), scores.std(), label))
print('-'*20)

#------------------------------------------------------------------------------
sclf.fit(X_copy,y)
sclf_preds = sclf.predict(test_copy)
print(eclf_preds[:10])
#------------------------------------------------------------------------------
