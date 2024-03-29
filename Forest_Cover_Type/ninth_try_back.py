import numpy as np 
import pandas as pd

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

##FEATURE_IMPORTANCES----------------------------------------------------------

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=1)
clf = clf.fit(X,y)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

features = pd.DataFrame({'Features': X.columns, 
                         'Importances': clf.feature_importances_})
plt.figure(figsize=(12,4))
sns.barplot(x='Features', y='Importances', data=features)
plt.xticks(rotation='vertical')
#plt.show()

##PREPROCESS-------------------------------------------------------------------

#horizontal and vertical distance to hydrology can be easily combined
cols = ['Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology']
X['Distance_to_hydrology'] = X[cols].apply(np.linalg.norm, axis=1)
test['Distance_to_hydrology'] = test[cols].apply(np.linalg.norm, axis=1)

#adding a few combinations of distance features to help enhance the classification
cols = ['Horizontal_Distance_To_Roadways',
        'Horizontal_Distance_To_Fire_Points',
        'Horizontal_Distance_To_Hydrology']

def addDistFeatures(df):
    df['distance_mean'] = df[cols].mean(axis=1)
    df['distance_sum'] = df[cols].sum(axis=1)
    df['distance_dif_road_fire'] = df[cols[0]] - df[cols[1]]
    df['distance_dif_hydro_road'] = df[cols[2]] - df[cols[0]]
    df['distance_dif_hydro_fire'] = df[cols[2]] - df[cols[1]]
    return df

X = addDistFeatures(X)
test = addDistFeatures(test)

#adding Hillshade
cols = ['Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm']
#weights = pd.Series([0.299, 0.587, 0.114], index=cols)
#X['Hillshade_Sum'] = X[cols].sum(1)
#test['Hillshade_Sum'] = test[cols].sum(1)

#applying vague connections to slope
#radiation is proportional to [cos(zenith)*cos(slope)+sin(zenith)*sin(slope)*cos(azimuth-aspect)]
#co-ordinates: 40°42′32″N 105°34′52″W
#https://www.esrl.noaa.gov/gmd/grad/solcalc/azel.html
#X['Cosine_of_slope'] = np.cos(X['Slope']) 
#X['Sine_of_slope'] = np.sin(X['Slope'])

#classifying elevation
#X['Prob_Spruce'] = (X['Elevation'].isin(range(2500,3700))).astype(int)
#X['Prob_Lodgepole'] = (X['Elevation'].isin(range(0,3500))).astype(int)
#X['Prob_Ponderosa'] = (X['Elevation'].isin(range(0,3000))).astype(int)
#X['Prob_Cottonwood'] = (X['Elevation'].isin(range(1500,2600))).astype(int)
#X['Prob_Aspen'] = (X['Elevation'].isin(range(1500,3600))).astype(int)
#X['Prob_Douglas'] = (X['Elevation'].isin(range(500,3000))).astype(int)
#X['Krummholz'] = (X['Elevation'].isin(range(2800,4000))).astype(int)
#cols = ['Prob_Spruce','Prob_Lodgepole','Prob_Ponderosa','Prob_Cottonwood','Prob_Aspen','Prob_Douglas','Krummholz']
#print(X[cols].head())
df_ = pd.concat([X['Elevation'],y],axis=1)
print(df_.groupby('Cover_Type').agg({'Elevation':['min','max']}) )
#X['Log_Elevation'] = np.log(X['Elevation'])

#trying a feature set based on soil description
soil_description = \
"""1 Cathedral family - Rock outcrop complex, extremely stony.
2 Vanet - Ratake families complex, very stony.
3 Haploborolis - Rock outcrop complex, rubbly.
4 Ratake family - Rock outcrop complex, rubbly.
5 Vanet family - Rock outcrop complex, rubbly.
6 Vanet - Wetmore families - Rock outcrop complex, stony.
7 Gothic family.
8 Supervisor - Limber families complex.
9 Troutville family, very stony.
10 Bullwark - Catamount families - Rock outcrop complex, rubbly.
11 Bullwark - Catamount families - Rock land complex, rubbly.
12 Legault family - Rock land complex, stony.
13 Catamount family - Rock land - Bullwark family complex, rubbly.
14 Pachic Argiborolis - Aquolis complex.
15 unspecified in the USFS Soil and ELU Survey.
16 Cryaquolis - Cryoborolis complex.
17 Gateview family - Cryaquolis complex.
18 Rogert family, very stony.
19 Typic Cryaquolis - Borohemists complex.
20 Typic Cryaquepts - Typic Cryaquolls complex.
21 Typic Cryaquolls - Leighcan family, till substratum complex.
22 Leighcan family, till substratum, extremely bouldery.
23 Leighcan family, till substratum - Typic Cryaquolls complex.
24 Leighcan family, extremely stony.
25 Leighcan family, warm, extremely stony.
26 Granile - Catamount families complex, very stony.
27 Leighcan family, warm - Rock outcrop complex, extremely stony.
28 Leighcan family - Rock outcrop complex, extremely stony.
29 Como - Legault families complex, extremely stony.
30 Como family - Rock land - Legault family complex, extremely stony.
31 Leighcan - Catamount families complex, extremely stony.
32 Catamount family - Rock outcrop - Leighcan family complex, extremely stony.
33 Leighcan - Catamount families - Rock outcrop complex, extremely stony.
34 Cryorthents - Rock land complex, extremely stony.
35 Cryumbrepts - Rock outcrop - Cryaquepts complex.
36 Bross family - Rock land - Cryumbrepts complex, extremely stony.
37 Rock outcrop - Cryumbrepts - Cryorthents complex, extremely stony.
38 Leighcan - Moran families - Cryaquolls complex, extremely stony.
39 Moran family - Cryorthents - Leighcan family complex, extremely stony.
40 Moran family - Cryorthents - Rock land complex, extremely stony.
""" 
import re
#soil_type = list(filter(None, re.split(r"[,\n\d]+", soil_description)) ) 
#soil_type = [i.strip() for i in soil_type]
#soil_type.remove('Rock outcrop complex complex')
#soil_type = list(set(soil_type))
#print(np(soil_type)) 
lines = soil_description.splitlines()
tmp = [re.split(r"[.,\-\d]+", line) for line in lines]
tmp = [[s.strip() for s in lst] for lst in tmp]
soil_type = [list(filter(None, lst)) for lst in tmp]
#soil_type = np.array([np.array(i) for i in soil_type])
#print(soil_type)
soil_class = list(set([item for sublist in soil_type for item in sublist]))
#soil_type = list(filter(bool, [i.strip() for i in tmp]))
#print(soil_class)
#https://stackoverflow.com/questions/53631460/using-numpy-isin-element-wise
#soil_type_array = (np.array(soil_class)==soil_type[...,None]).any(axis=1)
#print(soil_type)
rocky = [s for s in soil_class if "Rock" in s]
stony = [s for s in soil_class if "stony" in s]
rubbly = [s for s in soil_class if "rubbly" in s]

c = np.array([np.in1d(np.array(soil_class), el) for el in np.array(soil_type)]).astype(int)
#for el in np.array(soil_type):
#print(el, np.where(np.array(soil_class)==el[0]),(np.in1d(np.array(soil_class), el)) )
scf = pd.DataFrame(data=c, index=range(1,41), columns=soil_class)
scf['rocky'] = np.logical_or.reduce(scf[rocky], axis=1)
scf['stony'] = np.logical_or.reduce(scf[stony], axis=1)
#scf = scf[['rocky','stony','rubbly']]
stone_cols = ['rocky','stony','rubbly']
family_cols = [col for col in scf.columns if 'family' in col]
print(family_cols)
scf = scf[stone_cols + family_cols]

def transformCols(df):
    #df_s = df.loc[:,'Soil_Type1':'Soil_Type40']
    df = df.loc[:, :'Horizontal_Distance_To_Fire_Points'] \
            .join(df.loc[:,'Wilderness_Area1':'Wilderness_Area4'] \
            .dot(range(1,5)).to_frame('Wilderness_Area1')) \
            .join(df.loc[:,'Soil_Type1':'Soil_Type40'] \
            .dot(range(1,41)).to_frame('Soil_Type1')) \
            .join(df.loc[:,'Distance_to_hydrology':])
    df_c = scf.reindex(list(df['Soil_Type1'])).reset_index(drop=True)
    #df[['Wilderness_Area1','Soil_Type1']] = df[['Wilderness_Area1','Soil_Type1']].astype('category')
    df = pd.concat([df,  df_c], axis=1) #df_s,
    return df

#X = transformCols(X)
print(X.columns)
#test = transformCols(test)

families = ['Como family', 'Troutville family', 'Bullwark family complex', 
'Rogert family', 'Leighcan family', 'Vanet family', 'Moran family', 
'Cathedral family', 'Bross family', 'Gothic family', 'Ratake family', 
'Gateview family', 'Legault family', 'Leighcan family complex', 'Catamount family']
elev_range =   [(2000, 2750), #Como
                (2438, 3474), #Troutville
                (2286, 3048), #Bullwark
                (2300, 3300), #Rogert
                (2133, 3657), #Leighcan
                (2370, 2590), #Vanet
                (1980, 3350), #Moran
                (1890, 3000), #Cathedral
                (3048, 4267), #Bross
                (2200, 3200), #Gothic
                (2286, 3048), #Ratake
                (2316, 3048), #Gateview
                (2286, 3474), #Legault
                (2133, 3657), #Leighcan
                (2438, 3505)  #Catamount
                ]

 
""" {
    'Cathedral family - Rock outcrop complex, extremely stony': [Slope: ],
    'Vanet - Ratake families complex, very stony': [Slope: 5 to 40 percent], 
    'Haploborolis - Rock outcrop complex': [Slope: 5 to 40 percent],
    'Ratake family - Rock outcrop complex': [Slope: 5 to 40 percent],
    'Vanet family - Rock outcrop complex': [Slope: 5 to 40 percent],
    'Vanet - Wetmore families - Rock outcrop complex': [Slope: 5 to 40 percent],
    'Bullwark - Catamount families - Rock outcrop complex': [Slope: 40 to 150 percent],
    'Typic Cryaquepts - Typic Cryaquolls complex': [Slope: 0 to 15 percent],
    'Moran family-Lithic Cryorthents-Leighcan family complex': [Slope: 40 to 75 percent],
    'Gateview family - Cryaquolis complex': [Slope: 0 to 15 percent],
    'Gothic family': [Elevation: 2200 to 3200 meters, Slope: 2 to 60 percent],
    'Cathedral family': [Elevation: 1890 to 3000 meters, Slope: 2 to 100 percent],
    'Vanet family': [Elevations: 2370 to 2590 meters, Slope: 20 to 40 percent],
    'Ratake family': [Elevation: 2286 to 3048 meters, Slope: 2 to 60 percent],
    'Rogert family': [Elevation: 2300 to 3300 meters, Slopes: 3 to 100 percent],
    'Legault family': [Elevation: 2286 to 3474 meters, Slope: 5 to 80 percent],
    'Moran family': [Elevation: 1980 to 3350 meters, Slope: 0 to 70 percent],
    'Bross family': [Elevation: 3048 to 4267 meters, Slope: 2 to 50 percent],
    'Catamount family': [Elevation: 2438 to 3505 meters, Slope: 5 to 70 percent],
    'Troutville family': [Elevation: 2438 to 3474 meters, Slope: 2 to 60 percent],
    'Leighcan family': [Elevation: 2133 to 3657 meters, Slope:0 to 70 percent],
    'Bullwark family': [Elevation: 2286 to 3048 meters, Slope: 5 to 50 percent],
    'Bross family': [Elevation: 3048 to 4267 meters, Slope: 2 to 50 percent],
    'Como family': [Elevation: 2000 to 2750 meters, Slope: 10 to 60 percent],
    'Gateview family': [Elevation: 2316 to 3048 meters, Slope: 2 to 45 percent]
} """


##SPLIT------------------------------------------------------------------------

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)

##PLOT-------------------------------------------------------------------------

param_grid = {"n_estimators":  [int(x) for x in np.linspace(start = 10, stop = 200, num = 11)],
              "max_depth": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None],
              "min_samples_split": np.linspace(0.1, 1.0, 10, endpoint=True), #np.arange(1,150,1),
              "min_samples_leaf": np.linspace(0.1, 0.5, 5, endpoint=True),  #np.arange(1,60,1),
              "max_leaf_nodes": np.arange(2,60,1),
              "min_weight_fraction_leaf": np.arange(0.1,0.4, 0.1)}

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

def multiclass_roc_auc_score(test, pred, average='micro'):
    lb = LabelBinarizer()
    lb.fit(test)
    test = lb.transform(test)
    pred = lb.transform(pred)
    return roc_auc_score(test, pred, average=average)


#parameter tuning
from collections import defaultdict
import warnings


def evaluate_param(clf, param_grid, metric, metric_abv):
    data = []
    for parameter, values in dict.items(param_grid):
        for value in values:
            d = {parameter:value}
            warnings.filterwarnings('ignore') 
            clf = RandomForestClassifier(**d)
            clf.fit(X_train, y_train)
            x_pred = clf.predict(X_train)
            train_score = metric(y_train, x_pred)
            y_pred = clf.predict(X_val)
            test_score = metric(y_val, y_pred)
            data.append({'Parameter':parameter, 'Param_value':value, 
            'Train_'+metric_abv:train_score, 'Test_'+metric_abv:test_score})
    df = pd.DataFrame(data)
    _, axes = plt.subplots(nrows=2, ncols=3, figsize=(10,5))
    for (parameter, group), ax in zip(df.groupby(df.Parameter), axes.flatten()):
        group.plot(x='Param_value', y=(['Train_'+metric_abv,'Test_'+metric_abv]),
        kind='line', ax=ax, title=parameter)
        ax.set_xlabel('')
    plt.tight_layout()
    plt.show()

#evaluate_param(clf, param_grid, multiclass_roc_auc_score, 'AUC')
#evaluate_param(clf, param_grid, accuracy_score, 'ACC')

##TUNE-------------------------------------------------------------------------

param_grid2 = {"n_estimators": [200,300],
                #'max_leaf_nodes': [150,None],
                #'max_depth': [20,None],
                #'min_samples_split': [2, 5], 
                #'min_samples_leaf': [1, 2],
              "max_features": ['auto','sqrt'],
              "bootstrap": [True, False]
              }

""" from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(clf, param_grid2, refit = True, cv=5, verbose = 3)
grid.fit(X_train, y_train)
# print best parameter after tuning 
print('Best parameters: ',grid.best_params_)
# print how our model looks after hyper-parameter tuning 
print('Best estimator: ',grid.best_estimator_)
grid_predictions = grid.predict(X_val) 

from sklearn.metrics import classification_report
print(classification_report(y_val, grid_predictions)) """

#http://ataspinar.com/2017/05/26/classification-with-scikit-learn/
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
import time

clf_dict = {
    "Logistic Regression": LogisticRegression(),
    "Nearest Neighbors": KNeighborsClassifier(),
    "Linear SVM": SVC(),
    "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=100),
    "Decision Tree": tree.DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Neural Net": MLPClassifier(alpha = 1),
    "Naive Bayes": GaussianNB(),
}

def batch_classify(X_train, y_train, X_val, y_val, no_clf = 5, verbose = True):
    dict_models = {}
    for clf_name, clf in list(clf_dict.items())[:no_clf]:
        t_start = time.clock()
        clf.fit(X_train, y_train)
        t_end = time.clock()
        
        t_diff = t_end - t_start
        train_score = clf.score(X_train, y_train)
        val_score = clf.score(X_val, y_val)
        
        dict_models[clf_name] = {'model': clf, 'train_score': train_score, 'val_score': val_score, 'train_time': t_diff}
        if verbose:
            print("trained {c} in {f:.2f} s".format(c=clf_name, f=t_diff))
    return dict_models

def display_dict_models(dict_models, sort_by='val_score'):
    cls = [key for key in dict_models.keys()]
    test_s = [dict_models[key]['val_score'] for key in cls]
    training_s = [dict_models[key]['train_score'] for key in cls]
    training_t = [dict_models[key]['train_time'] for key in cls]
    
    df_ = pd.DataFrame(data=np.zeros(shape=(len(cls),4)), columns = ['classifier', 'train_score', 'val_score', 'train_time'])
    for ii in range(0,len(cls)):
        df_.loc[ii, 'classifier'] = cls[ii]
        df_.loc[ii, 'train_score'] = training_s[ii]
        df_.loc[ii, 'val_score'] = test_s[ii]
        df_.loc[ii, 'train_time'] = training_t[ii]
    
    print(df_.sort_values(by=sort_by, ascending=False))

dict_models = batch_classify(X_train, y_train, X_val, y_val, no_clf = 8)
display_dict_models(dict_models)