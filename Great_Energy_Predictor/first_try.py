import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Suppress warnings 
import warnings
warnings.filterwarnings('ignore')
import gc


# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
#------------------------------------------------------------------------------
train = pd.read_csv('train.csv')
print('train:',train.shape)
test = pd.read_csv('test.csv')
print('test:',test.shape)
weather_train = pd.read_csv('weather_train.csv')
print('weather_train:',weather_train.shape)
weather_test = pd.read_csv('weather_test.csv')
print('weather_test:',weather_test.shape)
building_metadata = pd.read_csv('building_metadata.csv')
print('building_metadata:',building_metadata.shape)
sample_submission = pd.read_csv('sample_submission.csv')
print('sample_submission:',sample_submission.shape)
#------------------------------------------------------------------------------
print('train columns:\n',train.dtypes)
print('test columns:\n',train.dtypes)
print('weather_train columns:\n',weather_train.dtypes)
print('weather_test columns:\n',weather_test.dtypes)
print('building_metadata columns:\n',building_metadata.dtypes)
#------------------------------------------------------------------------------
#ripped from here:
#https://www.kaggle.com/caesarlupum/ashrae-start-here-a-gentle-introduction#Reducing-Memory-Size

def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
#------------------------------------------------------------------------------
train = reduce_mem_usage(train)
test = reduce_mem_usage(test)

weather_train = reduce_mem_usage(weather_train)
weather_test = reduce_mem_usage(weather_test)
building_metadata = reduce_mem_usage(building_metadata)
#------------------------------------------------------------------------------
train['timestamp'] = pd.to_datetime(train['timestamp'])
test['timestamp'] = pd.to_datetime(test['timestamp'])
weather_train['timestamp'] = pd.to_datetime(weather_train['timestamp'])
weather_test['timestamp'] = pd.to_datetime(weather_test['timestamp'])
#------------------------------------------------------------------------------
building_metadata['primary_use'] = building_metadata['primary_use'].astype('category')

temp_df = train_df[['building_id']]
temp_df = temp_df.merge(building_meta_df, on=['building_id'], how='left')
del temp_df['building_id']
train_df = pd.concat([train_df, temp_df], axis=1)

temp_df = test_df[['building_id']]
temp_df = temp_df.merge(building_meta_df, on=['building_id'], how='left')

del temp_df['building_id']
test_df = pd.concat([test_df, temp_df], axis=1)
del temp_df, building_meta_df