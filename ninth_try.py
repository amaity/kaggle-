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

clf = RandomForestClassifier()
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
accuracy_score(y_val, y_pred)

#PREPROCESS

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

#taking some factors influencing the amount of radiation
X['Cosine_of_slope'] = np.cos(np.radians(X['Slope']) )
#X['Diff_azimuth_aspect_9am'] = np.cos(np.radians(123.29-X['Aspect']))
#X['Diff_azimuth_aspect_12noon'] = np.cos(np.radians(181.65-X['Aspect']))
#X['Diff_azimuth_aspect_3pm'] = np.cos(np.radians(238.56-X['Aspect']))

X['Elevation_VDH'] = X['Elevation'] - X['Vertical_Distance_To_Hydrology']

print(X.columns)

# Plotting mode frequencies as % of data size
#take from: https://www.kaggle.com/kwabenantim/forest-cover-feature-engineering
n_rows = X.shape[0]
mode_frequencies = [X[col].value_counts().iat[0] for col in X.columns]
mode_frequencies = 100.0 * np.asarray(mode_frequencies) / n_rows

mode_df = pd.DataFrame({'Features': X.columns, 
                        'Mode_Frequency': mode_frequencies})

mode_df.sort_values(by=['Mode_Frequency'], axis='index', ascending=True, inplace=True)

fig = plt.figure(figsize=(14, 4))
sns.barplot(x='Features', y='Mode_Frequency', data=mode_df)
plt.ylabel('Mode Frequency %')
plt.xticks(rotation='vertical')
#plt.show()

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

#SPLIT DATA--------------------------------------------------------------------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)
#------------------------------------------------------------------------------

#code from here:http://ataspinar.com/2017/05/26/classification-with-scikit-learn/
clf_dict = {
    "Logistic Regression": LogisticRegression(),
    "Nearest Neighbors": KNeighborsClassifier(),
    "Linear SVM": SVC(),
    "Gradient Boosting Classifier": GradientBoostingClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
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
        
        dict_models[clf_name] = {'model': clf, 'train_score': train_score, 
                                 'val_score': val_score, 'train_time': t_diff}
        if verbose:
            print("trained {c} in {f:.2f} s".format(c=clf_name, f=t_diff))
    return dict_models

def display_dict_models(dict_models, sort_by='val_score'):
    cls = [key for key in dict_models.keys()]
    test_s = [dict_models[key]['val_score'] for key in cls]
    training_s = [dict_models[key]['train_score'] for key in cls]
    training_t = [dict_models[key]['train_time'] for key in cls]
    
    df_ = pd.DataFrame(data=np.zeros(shape=(len(cls),4)), columns = ['classifier', 'train_score', 
                                                                     'val_score', 'train_time'])
    for ii in range(0,len(cls)):
        df_.loc[ii, 'classifier'] = cls[ii]
        df_.loc[ii, 'train_score'] = training_s[ii]
        df_.loc[ii, 'val_score'] = test_s[ii]
        df_.loc[ii, 'train_time'] = training_t[ii]
    
    print(df_.sort_values(by=sort_by, ascending=False))

dict_models = batch_classify(X_train, y_train, X_val, y_val, no_clf = 8)
display_dict_models(dict_models)

#code from here: https://www.dataquest.io/blog/introduction-to-ensembles/
SEED = 13

def get_models():
    """Generate a library of base learners."""
    nb = GaussianNB()
    svc = SVC(C=100, probability=True)
    knn = KNeighborsClassifier(n_neighbors=3)
    lr = LogisticRegression(C=100, random_state=SEED)
    nn = MLPClassifier((80, 10), early_stopping=False, random_state=SEED)
    gb = GradientBoostingClassifier(n_estimators=100, random_state=SEED)
    rf = RandomForestClassifier(n_estimators=10, max_features=3, random_state=SEED)

    models = {'svm': svc,
              'knn': knn,
              'naive bayes': nb,
              'mlp-nn': nn,
              'random forest': rf,
              'gbm': gb,
              'logistic': lr,
              }

    return models

def train_predict(model_list):
    """Fit models in list on training set and return preds"""
    P = np.zeros((y_val.shape[0], len(model_list)))
    P = pd.DataFrame(P)

    print("Fitting models.")
    cols = list()
    for i, (name, m) in enumerate(models.items()):
        print("%s..." % name, end=" ", flush=False)
        m.fit(X_train, y_train)
        P.iloc[:, i] = m.predict(X_val)
        cols.append(name)
        print("done")

    P.columns = cols
    print("Done.\n")
    return P

def score_models(P, y):
    """Score model in prediction DF"""
    print("Scoring models.")
    for m in P.columns:
        score = accuracy_score(y, P.loc[:, m])
        print("%-26s: %.3f" % (m, score))
    print("Done.\n")

models = get_models()
P = train_predict(models)
score_models(P, y_val)

base_learners = get_models()
meta_learner = GradientBoostingClassifier(
    n_estimators=1000,
    max_features=4,
    max_depth=3,
    subsample=0.5,
    learning_rate=0.005,
    random_state=SEED)

xtrain_base, xval_base, ytrain_base, yval_base = train_test_split(
    X, y, test_size=0.5, stratify=y, random_state=SEED)

def train_base_learners(base_learners, inp, out, verbose=True):
    """
    Train all base learners in the library.
    """
    if verbose: print("Fitting models.")
    for _, (name, m) in enumerate(base_learners.items()):
        if verbose: print("%s..." % name, end=" ", flush=False)
        m.fit(inp, out)
        if verbose: print("done")

train_base_learners(base_learners, xtrain_base, ytrain_base)

def predict_base_learners(pred_base_learners, inp, verbose=True):
    """
    Generate a prediction matrix.
    """
    P = np.zeros((inp.shape[0], len(pred_base_learners)))

    if verbose: print("Generating base learner predictions.")
    for i, (name, m) in enumerate(pred_base_learners.items()):
        if verbose: print("%s..." % name, end=" ", flush=False)
        p = m.predict(inp)
        P[:, i] = p
        if verbose: print("done")

    return P

P_base = predict_base_learners(base_learners, xval_base)

meta_learner.fit(P_base, yval_base)

def ensemble_predict(base_learners, meta_learner, inp, verbose=True):
    """
    Generate predictions from the ensemble.
    """
    P_pred = predict_base_learners(base_learners, inp, verbose=verbose)
    return P_pred, meta_learner.predict(P_pred)

P_pred, p = ensemble_predict(base_learners, meta_learner, X_val)
print("\nAccuracy score: %.3f" % accuracy_score(y_val, p))

#https://www.mikulskibartosz.name/fill-missing-values-using-random-forest/
