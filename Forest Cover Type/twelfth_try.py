import os, random, pickle
import numpy as np 
import pandas as pd
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.decomposition import PCA
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
#print(train.columns)
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
    df['distance_road_fire'] = df[cols[:2]].mean(axis=1)
    df['distance_hydro_fire'] = df[cols[1:]].mean(axis=1)
    df['distance_road_hydro'] = df[[cols[0], cols[2]]].mean(axis=1)
    df['distance_sum_road_fire'] = df[cols[:2]].sum(axis=1)
    df['distance_sum_hydro_fire'] = df[cols[1:]].sum(axis=1)
    df['distance_sum_road_hydro'] = df[[cols[0], cols[2]]].sum(axis=1)
    df['distance_dif_road_fire'] = df[cols[0]] - df[cols[1]]
    df['distance_dif_hydro_road'] = df[cols[2]] - df[cols[0]]
    df['distance_dif_hydro_fire'] = df[cols[2]] - df[cols[1]]
    df['vertical_dif'] = df['Elevation'] - df['Vertical_Distance_To_Hydrology']
    df['vertical_sum'] = df['Elevation'] + df['Vertical_Distance_To_Hydrology']
    
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
    df['shade_noon_diff'] = df['Hillshade_9am'] - df['Hillshade_Noon']
    df['shade_3pm_diff'] = df['Hillshade_Noon'] - df['Hillshade_3pm']
    df['shade_all_diff'] = df['Hillshade_9am'] - df['Hillshade_3pm']
    df['shade_mean'] = df[shades].mean(axis=1)
    
    
    #classifying elevation
    """ df['proba_Spruce'] = (df['Elevation'].isin(range(2500,3700))).astype(int)
    df['proba_Lodgepole'] = (df['Elevation'].isin(range(0,3500))).astype(int)
    df['proba_Ponderosa'] = (df['Elevation'].isin(range(0,3000))).astype(int)
    df['proba_Cottonwood'] = (df['Elevation'].isin(range(1500,2600))).astype(int)
    df['proba_Aspen'] = (df['Elevation'].isin(range(1500,3600))).astype(int)
    df['proba_Douglas'] = (df['Elevation'].isin(range(500,3000))).astype(int)
    df['proba_Krummholz'] = (df['Elevation'].isin(range(2800,4000))).astype(int)
    cols = ['proba_Spruce','proba_Lodgepole','proba_Ponderosa','proba_Cottonwood',
            'proba_Aspen','proba_Douglas','proba_Krummholz']
    df[cols] = df[cols].div(df[cols].sum(1), axis=0) """

    #slope
    #df["slope_times_elevation"] = df["Slope"] * df["Elevation"]

    #binned features
    bin_defs = [
        # col name, bin size, new name
        ('Elevation', 200, 'binned_elevation'), 
        ('Aspect', 45, 'binned_aspect'),
        ('Slope', 6, 'binned_slope'),
    ]
    
    for col_name, bin_size, new_name in bin_defs:
        df[new_name] = np.floor(df[col_name]/bin_size)

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
    
    train.loc[:,:'shade_mean'] = normalize(train.loc[:,:'shade_mean'])
    test.loc[:,:'shade_mean'] = normalize(test.loc[:,:'shade_mean'])
    
    # elevation was found to have very different distributions on test and training sets
    # lets just drop it for now to see if we can implememnt a more robust classifier!
    #train = train.drop('Elevation', axis=1)
    #test = test.drop('Elevation', axis=1)    

    return {'X_train': dtrn_, 'X_test': dtst_, 'y_train': y_train}  #dtrn_, dtst_, y_train
#------------------------------------------------------------------------------
pp = preprocessData(train, test)
X, test, y = pp['X_train'], pp['X_test'], pp['y_train']
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

#X = drop_unimportant(X)
#feature_names = list(X.columns)
#test = test[feature_names]
#------------------------------------------------------------------------------
def initialWeights(list_of_weights,list_of_col_names,col_name,weight):
    indices = [i for i, el in enumerate(list_of_col_names) if col_name in el]
    for idx in indices:
        list_of_weights[idx] = weight
    return list_of_weights
#------------------------------------------------------------------------------
def getWeights(X_full, y_full):
    cols = list(X_full.columns.values)
    print(cols)
    #starting weights
    if os.path.isfile("weights.pickle"):
        with open( "weights.pickle", "rb" ) as f:
            w = pickle.load(f)
            if len(w) == X_full.shape[1]:
                weights = w
    else:
        weights = [1]*X_full.shape[1]
        weights = initialWeights(weights, cols, 'Elevation', 8)
        weights = initialWeights(weights, cols, 'Wilderness', 400)
        weights = initialWeights(weights, cols, 'Soil', 400)
    
    best_score_ever=0
    best_ever_wts = [i for i in weights]
    lr = 0.5
    
    for step in range(15):        
        X_full_copy = X_full.copy()

        for i in range(len(weights)):
            c = X_full.columns[i]
            X_full_copy[c] *= weights[i]

        r = lr*random.random()
        
        
        # Choose a random weight to change
        train_index = random.randint(0,X_full.shape[1]-1)
        train_col = X_full.columns[train_index]
        
        # We will test four factors for changing the current weight.
        factors = [1-r,1,1+r, 1+2*r]        
        wts = [weights[train_index] * f for f in factors]

        best_score=0
        best_wt=-1

        for wt in wts:  
            X_full_copy[train_col] = wt * X_full[train_col]

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

best_weights, final_weights = getWeights(X, y)
print('Final weights: ',final_weights)
weights = final_weights
with open('weights.pickle', 'wb') as fh1:
        pickle.dump(weights, fh1)
#------------------------------------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1,p=1)

def addWeights(X,test):
    X_copy = X.copy()
    test_copy = test.copy()

    for i in range(len(X.columns)):
        c = X.columns[i]
        X_copy[c] *= weights[i]
        test_copy[c] *= weights[i]
    return X_copy, test_copy
#------------------------------------------------------------------------------
X_copy, test_copy = addWeights(X, test)
#------------------------------------------------------------------------------
pca = PCA()
pca.fit(X_copy)

#Transforming x_train and x_test 
x_train_pca = pca.transform(X_copy)
#x_test_pca = pca.transform(x_test)

#Find number of components that explain predefined variance threshold
sum_variance, component_count = 0, 0
while sum_variance < 0.85:
    sum_variance += pca.explained_variance_ratio_[component_count]
    component_count += 1

print('-'*20)    
print('Number of Principal Components that explain >=85% of Variance: ', component_count)
print('Total Variance Explained by '+str(component_count)+' components:', str(sum_variance*100)+'%')
#------------------------------------------------------------------------------
pca = PCA(n_components=2)
pca.fit(X_copy)

#Transforming x_train and x_test 
x_train_pca = pca.transform(X_copy)
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
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
#import xgboost as xgb

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

##clf1 = KNeighborsClassifier(n_neighbors=1)
##BaggingClassifier(DecisionTreeClassifier(max_leaf_nodes=2000), n_estimators=250,random_state=1)
##AdaBoostClassifier(base_estimator=RandomForestClassifier(), random_state=1)
##KNeighborsClassifier(n_neighbors=1)
rnf = RandomForestClassifier(n_estimators=181, max_features='sqrt', bootstrap=False,max_depth=60,
                              min_samples_split=2,min_samples_leaf=1,random_state=1)
etr = ExtraTreesClassifier(n_estimators=500,max_features=X_copy.shape[1],min_samples_split=5,min_samples_leaf=1,random_state=1) 
lgb = LGBMClassifier(objective='multiclass',num_class=7,learning_rate=0.2,num_leaves=X_copy.shape[1],random_state=1) #num_leaves=109,
lrg = LogisticRegression(C=1000,multi_class='multinomial', solver='newton-cg', random_state=1)
mlp = MLPClassifier(activation='logistic',max_iter=500)
#xgb = xgb.XGBClassifier(objective='multi:softmax')
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
    'max_depth': [75,100,125,150]
    }

lgb_param = {
    'num_leaves': range(51,201,2),
    'learning_rate': [0.05,0.001,0.01,0.2],
    'class_weight': [None, 'balanced'],
    'boosting_type': ['gbdt','dart','goss']
}

lr_param = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
}

mlp_param = {
    'hidden_layer_sizes': [(66,66), (132,), (198,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.001, 0.05, 0.5],
    'learning_rate': ['constant','adaptive'],
}

def gridSearch(clf,test_params):
    rs = RandomizedSearchCV(estimator=clf, param_distributions=test_params, 
                                scoring='accuracy', cv=3, verbose=3)
    rs.fit(X_copy,y)
    print('-'*20)
    print('Best parameters: ',rs.best_params_)
    print('Best score: ',rs.best_score_)
    print('-'*20)

#gridSearch(mlp, mlp_param)
#------------------------------------------------------------------------------

from mlxtend.classifier import StackingCVClassifier
lr1 = LogisticRegression(C=5, random_state=1, solver='liblinear', multi_class= 'ovr')
sclf = StackingCVClassifier(classifiers=[knn,rnf,etr],meta_classifier=lgb)

print('-'*20)
print('5-fold cross validation:')

for clf, label in zip([knn,rnf,etr,sclf], 
                      ['Kneighbors','Random Forest','Extra Trees','StackingClf']):

    scores = model_selection.cross_val_score(clf, X_copy.values, y.values, cv=5, scoring='accuracy')
    print("Accuracy: %0.3f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
#------------------------------------------------------------------------------
#sclf.fit(X_copy.values,y.values)
#sclf_preds = sclf.predict(test_copy.values)
#print(sclf_preds[:10])
#------------------------------------------------------------------------------

