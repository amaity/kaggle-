import numpy as np 
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import time
import warnings
warnings.simplefilter('ignore')
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

def plotImpFeatures(X):
    features = pd.DataFrame({'Features': X.columns, 
                         'Importances': clf.feature_importances_})
    features.sort_values(by=['Importances'], axis='index', ascending=False, inplace=True)
    plt.figure(figsize=(12,4))
    sns.barplot(x='Features', y='Importances', data=features)
    plt.xticks(rotation='vertical')
    plt.show()

#plotImpFeatures(X)

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

X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.3)

#------------------------------------------------------------------------------
param_grid2 = {"n_estimators": [29,47,113,181],
                #'max_leaf_nodes': [150,None],
                #'max_depth': [20,None],
                #'min_samples_split': [2, 5],
                #'min_samples_leaf': [1, 2],
              "max_features": ['auto','sqrt'],
              "bootstrap": [True, False]}

def grid_search(clf, params, xtrain, ytrain, yval, cv=5):
    grid = GridSearchCV(clf, param_grid2, refit=True, cv=cv, verbose=0)
    grid.fit(X_train, y_train)
    print('Best parameters: ',grid.best_params_)
    print('Best estimator: ',grid.best_estimator_)
    grid_predictions = grid.predict(X_val) 
    print(classification_report(y_val, grid_predictions))

#grid_search(clf, param_grid2, X_train, y_train, y_val)
#------------------------------------------------------------------------------

neighbors = list(range(1, 50, 2))

def tuning_knn(params, xtrain, ytrain, cv):
    # empty list that will hold cv scores
    cv_scores = []
    # perform 10-fold cross validation
    for k in neighbors:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
        cv_scores.append(scores.mean())

    mse = [1 - x for x in cv_scores]
    # determining best k
    optimal_k = neighbors[mse.index(min(mse))]
    print("The optimal number of neighbors is {}".format(optimal_k))
    # plot misclassification error vs k
    plt.plot(neighbors, mse)
    plt.xlabel("Number of Neighbors K")
    plt.ylabel("Misclassification Error")
    plt.show()

#tuning_knn(neighbors, X_train, y_train, 5)
#------------------------------------------------------------------------------
""" params = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10],
          'p': [1,2,3,4,5]}
grid_search = GridSearchCV(KNeighborsClassifier(), params, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)
print('-'*20)
print('knn:')
print("train score - " + str(grid_search.score(X_train, y_train)))
print("test score - " + str(grid_search.score(X_val, y_val)))
print(grid_search.best_params_)
print('-'*20)

params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
grid_search = GridSearchCV(LogisticRegression(), params, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)
print('-'*20)
print('knn:')
print("train score - " + str(grid_search.score(X_train, y_train)))
print("test score - " + str(grid_search.score(X_val, y_val)))
print(grid_search.best_params_)
print('-'*20) """


rfc = RandomForestClassifier(n_estimators=181, bootstrap=False, 
                               max_features='auto', random_state=SEED)
gbc = GradientBoostingClassifier(n_estimators=500, learning_rate=0.1,
max_features='sqrt',max_depth=6,max_features=6,random_state=SEED)
lr = LogisticRegression(multi_class='multinomial', solver='newton-cg',
                        random_state=SEED)


from mlxtend.classifier import StackingCVClassifier
sclf = StackingCVClassifier(classifiers=[rfc, gbc], meta_classifier=lr)
print('-'*20)
print('3-fold cross validation:\n')

for clf, label in zip([rfc, gbc, sclf], 
                      ['rfc', 
                       'gbc',
                       'StackingClassifier']):

    scores = cross_val_score(clf, X.values, y.values, cv=3, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" 
          % (scores.mean(), scores.std(), label))
print('-'*20)


grid = GridSearchCV(estimator=gbc, 
                    param_grid={'max_features':[2,3,4,5,6,7]}, 
                    cv=3,
                    verbose=3,
                    refit=True)
grid.fit(X.values, y.values)

cv_keys = ('mean_test_score', 'std_test_score', 'params')

for r, _ in enumerate(grid.cv_results_['mean_test_score']):
    print("%0.3f +/- %0.2f %r"
          % (grid.cv_results_[cv_keys[0]][r],
             grid.cv_results_[cv_keys[1]][r] / 2.0,
             grid.cv_results_[cv_keys[2]][r]))

print('Best parameters: %s' % grid.best_params_)
print('Accuracy: %.2f' % grid.best_score_)