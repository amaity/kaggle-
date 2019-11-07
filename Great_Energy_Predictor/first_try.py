import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Suppress warnings 
import sys, warnings
warnings.filterwarnings('ignore')
import gc


# matplotlib and seaborn for plotting
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
#------------------------------------------------------------------------------
#import zipfile
#with zipfile.ZipFile('ashrae-energy-prediction.zip', 'r') as zipf:
#    zipf.extractall('.')
#------------------------------------------------------------------------------
#https://hackersandslackers.com/downcast-numerical-columns-python-pandas/
import os, feather

if os.path.isfile("train.feather"):
    train = read_feather('train.feather')
    test = read_feather('test.feather')
else:
    print('train data:')
    train = pd.read_csv('train.csv', 
                    dtype={'building_id':np.uint16, 'meter':np.uint8, 'meter_reading':np.float64})
    train['timestamp'] = pd.to_datetime(train['timestamp'], format="%Y %m %d %H:%M:%S")
    print(train.info(memory_usage='deep'))
    print('-'*20);print('test data:')
    test = pd.read_csv('test.csv', 
                   dtype={'row_id':np.uint16,'building_id':np.uint16,'meter':np.uint16})
    test['timestamp'] = pd.to_datetime(test['timestamp'], format="%Y %m %d %H:%M:%S")
    print(test.info(memory_usage='deep'))
    print('-'*20);print('weather_train data:')
    weather_train = pd.read_csv('weather_train.csv',dtype={'site_id':np.uint16})
    weather_train['timestamp'] = pd.to_datetime(weather_train['timestamp'],infer_datetime_format=True)
    weather_train[['air_temperature', 'cloud_coverage',
       'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',
       'wind_direction', 'wind_speed']] = weather_train[['air_temperature', 'cloud_coverage',
       'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',
       'wind_direction', 'wind_speed']].apply(pd.to_numeric,downcast='float')
    print(weather_train.info(memory_usage='deep'))
    print('-'*20);print('weather_test data:')
    weather_test = pd.read_csv('weather_test.csv',dtype={'site_id':np.uint16})
    weather_test['site_id'] = weather_test['site_id'].apply(pd.to_numeric,downcast='unsigned')
    weather_test['timestamp'] = pd.to_datetime(weather_test['timestamp'],infer_datetime_format=True)
    weather_test[['air_temperature', 'cloud_coverage',
       'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',
       'wind_direction', 'wind_speed']] = weather_test[['air_temperature', 'cloud_coverage',
       'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',
       'wind_direction', 'wind_speed']].apply(pd.to_numeric,downcast='float')
    print(weather_test.info(memory_usage='deep'))
    print('-'*20);print('building_metadata data:')
    building_metadata = pd.read_csv('building_metadata.csv')
    building_metadata['primary_use'] = building_metadata['primary_use'].astype('category')
    print(building_metadata.info(memory_usage='deep'))
    train = train.merge(building_metadata, on='building_id', how='left')
    test = test.merge(building_metadata, on='building_id', how='left')
    train = train.merge(weather_train, on=['site_id', 'timestamp'], how='left')
    test = test.merge(weather_test, on=['site_id', 'timestamp'], how='left')
    del weather_train, weather_test, building_metadata
    gc.collect()
    
    train.to_feather('train.feather')
    test.to_feather('test.feather')
#------------------------------------------------------------------------------
print(train.tail())
#------------------------------------------------------------------------------
#Frequency of primary_use
train.groupby(['primary_use']).agg({'site_id':'nunique'}).rename(columns={'site_id':'N'}) 
#------------------------------------------------------------------------------
train['square_feet'].hist(bins=32) #is this is wrong?
plt.xlabel("square_feet")
plt.ylabel("Frequency")
#------------------------------------------------------------------------------
for label, df in train.groupby(['primary_use']):
    ax = sns.kdeplot(df['square_feet'], label=label, shade=True)
ax.set(xlabel='square_feet', ylabel='density')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.) #label colouring needs fixing
#------------------------------------------------------------------------------
