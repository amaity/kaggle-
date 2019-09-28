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

#X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3)
#clf.fit(X_train, y_train)
#y_pred = clf.predict(X_val)
#print('Acc score after preprocessing: ',accuracy_score(y_val, y_pred))
#print('-'*20)


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

X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.3)

""" params = {'n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10],
          'p': [1,2,3,4,5]}
grid_search = GridSearchCV(KNeighborsClassifier(), params, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)
print('-'*20)
print('knn:')
print("train score - " + str(grid_search.score(X_train, y_train)))
print("test score - " + str(grid_search.score(X_val, y_val)))
print(grid_search.best_params_)
print('-'*20) """

""" params = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }
grid_search = GridSearchCV(LogisticRegression(), params, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)
print('-'*20)
print('knn:')
print("train score - " + str(grid_search.score(X_train, y_train)))
print("test score - " + str(grid_search.score(X_val, y_val)))
print(grid_search.best_params_)
print('-'*20) """

from mlxtend.classifier import StackingCVClassifier

clf = RandomForestClassifier(n_estimators=100, bootstrap=False, random_state=SEED)
clf1 = KNeighborsClassifier(n_neighbors=1, p=1)
clf2 = GaussianNB()
clf3 = DecisionTreeClassifier(max_features='auto', random_state=SEED)
clf4 = LinearDiscriminantAnalysis()
lr = LogisticRegression(C=1, random_state=SEED)


sclf = StackingCVClassifier(classifiers=[clf, clf1, clf2, clf3],
                                         meta_classifier=lr)
print('-'*20)
print('3-fold cross validation:\n')

for clf, label in zip([clf, clf1, clf2, clf3, sclf], 
                      ['Random Forest', 
                       'KNN', 
                       'Naive Bayes',
                       'Decision Tree',
                       #'Linear Disc',
                       'StackingClassifier']):

    scores = cross_val_score(clf, X.values, y.values, cv=3, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" 
          % (scores.mean(), scores.std(), label))
print('-'*20)