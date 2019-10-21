import numpy as np, pandas as pd, os
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize

#------------------------------------------------------------------------------
train = pd.read_csv('train.csv', index_col='Id')
print('train data shape:',train.shape)
test = pd.read_csv('test.csv', index_col='Id')
print('test data shape:',test.shape)
print(train.groupby('Cover_Type')['Elevation'].count())
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
#missing_values = test.isnull().sum()
#print('Missing values in test data: ',missing_values)
#sys.exit("Error message")
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

    train.loc[:,:'elevation_vdh'] = normalize(train.loc[:,:'elevation_vdh'])
    test.loc[:,:'elevation_vdh'] = normalize(test.loc[:,:'elevation_vdh'])
    
    # elevation was found to have very different distributions on test and training sets
    # lets just drop it for now to see if we can implememnt a more robust classifier!
    #train = train.drop('Elevation', axis=1)
    #test = test.drop('Elevation', axis=1)    

    return dtrn_, dtst_, y_train
#------------------------------------------------------------------------------
X, test, y = preprocessData(train, test)
print('X data shape:',X.shape)
print('test data shape:',test.shape)
print('y data shape:',y.shape)
#------------------------------------------------------------------------------
losses , accuracies = [], []
cols = [c for c in train.columns if c not in ['Id', 'Cover_Type']]
oof = np.zeros(len(train))
pred = np.zeros(len(test))

# BUILD MODELS OF EACH CATEGORY
for i in range(7):
    # ONLY TRAIN WITH DATA WHERE WHEEZY EQUALS I
    train2 = train[train['Cover_Type']==i+1]
    test2 = test[test['wheezy-copper-turtle-magic']==i]
    idx1 = train2.index; idx2 = test2.index
    train2.reset_index(drop=True,inplace=True)
#------------------------------------------------------------------------------
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score
def multiclass_roc_auc_score(test, pred, average='macro'):
    lb = LabelBinarizer()
    lb.fit(test)
    test = lb.transform(test)
    pred = lb.transform(pred)
    return roc_auc_score(test, pred, average=average)

#------------------------------------------------------------------------------
auc = multiclass_roc_auc_score(y,pred)
print('QDA scores CV =',round(auc,5))
#------------------------------------------------------------------------------
""" d, it, score = 0, 0, 0.1
predictions = preds
test_copy = test.copy()

while score <= 0.95:
    print('it',it)
    test_copy['Cover_Type'] = predictions
    pseudo = test_copy.sample(frac=0.02+d, random_state=1)
    pseudo = pseudo.values
    aug_train = np.vstack((X_copy.values,pseudo[:,:-1]))
    aug_y = np.concatenate((y.values,pseudo[:,-1]),axis=0)
    score, model = votingEnsemble(aug_train,aug_y)
    predictions = model.predict(test.values)
    d += 0.01
    it += 1

   
print(predictions[:10]) """

