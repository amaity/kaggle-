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
#print(X.columns)
#print(X.loc[(X==0).any(axis=1)].columns)
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
#------------------------------------------------------------------------------

from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier, \
    ExtraTreesClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np
import warnings
warnings.simplefilter('ignore')
#------------------------------------------------------------------------------
clf1 = KNeighborsClassifier(n_neighbors=1)
##BaggingClassifier(DecisionTreeClassifier(max_leaf_nodes=2000), n_estimators=250,random_state=1)
##AdaBoostClassifier(base_estimator=RandomForestClassifier(), random_state=1)
##KNeighborsClassifier(n_neighbors=1)
clf2 = RandomForestClassifier(n_estimators=181, max_features='sqrt', random_state=1)
clf3 = ExtraTreesClassifier(n_estimators=400,max_depth=50,min_samples_split=2,random_state=1)
clf4 = LGBMClassifier()
#GradientBoostingClassifier(n_estimators=150,max_features='sqrt',max_depth=15,min_samples_split=200,random_state=1) ##acc:0.80
#GradientBoostingClassifier(n_estimators=100,min_samples_leaf=1,max_depth=3,min_samples_split=2,random_state=1) #acc:0.74
#clf5 = LogisticRegression(solver='newton-cg', multi_class='multinomial',random_state=1)
#------------------------------------------------------------------------------
param_test = {'n_estimators':range(50,301,50),
'max_depth':range(3,16,2), 'learning_rate':[0.05,0.01,0.1,10],
}
#subsample=0.85, n_estimators=300, min_samples_split=400, min_samples_leaf=70, max_features=13, max_depth=15, score=0.811, total=  38.8s

def randomSearch(clf,test_params):
    rs = RandomizedSearchCV(estimator=clf4, param_distributions=param_test, scoring='accuracy', cv=3, verbose=3)
    rs.fit(X,y)
    print('-'*20)
    print('Best parameters: ',rs.best_params_)
    print('Best score: ',rs.best_score_)
    print('-'*20)

#randomSearch(clf4, param_test)
#------------------------------------------------------------------------------
param_test1 = {'base_estimator__max_leaf_nodes':[500,1000,2000,5000,8000]}
param_test2 = {'max_depth':range(1,28,2), 'min_samples_split':range(10,101,5)}
param_test3 = {'min_samples_split':range(1000,2100,200), 'min_samples_leaf':range(30,71,10)}
neighbors = {'n_neighbors':list(range(1, 31, 2)), 'weights':['uniform','distance']}
param_test4 = {'base_estimator__max_depth' : [1, 2, 3, 4, 5],'max_samples' : [0.05, 0.1, 0.2, 0.5]}
param_test5 = {'n_clusters':range(1,11,1)}

def grifSearch(clf,test_params):
    gs = GridSearchCV(estimator=clf, param_grid=test_params, scoring='accuracy', cv=3, verbose=3)
    gs.fit(X,y)
    print('-'*20)
    print('Best parameters: ',gs.best_params_)
    print('Best score: ',gs.best_score_)
    print('-'*20)
#gridSearch(clf1, param_test5)
#------------------------------------------------------------------------------
print('5-fold cross validation:')
clfs = [clf1, clf2, clf3, clf4]
labels = ['KNeighbors', 'Random Forest', 'Extra Trees', 'Grad Boost']
for clf, label in zip(clfs, labels):
    scores = model_selection.cross_val_score(clf, X, y, cv=5, 
                                              scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]"
          % (scores.mean(), scores.std(), label))
print('-'*20)

from mlxtend.classifier import EnsembleVoteClassifier
eclf = EnsembleVoteClassifier(clfs=clfs, weights=[1,1,1,1])
clfs = [clf1, clf2, clf3, clf4, eclf]
labels = ['KNeighbors', 'Random Forest', 'Extra Trees', 'Grad Boost', 'Ensemble']
for clf, label in zip(clfs, labels):
    scores = model_selection.cross_val_score(clf, X, y, cv=5, 
                                              scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" 
          % (scores.mean(), scores.std(), label))
print('-'*20)
#------------------------------------------------------------------------------
