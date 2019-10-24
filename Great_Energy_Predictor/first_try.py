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
print('train data:');print('-'*20)
train = pd.read_csv('train.csv', parse_dates=['timestamp'],
                    dtype={'building_id':np.uint16, 'meter':np.uint8, 'meter_reading':np.float64})
print(train.info(memory_usage='deep'))
#------------------------------------------------------------------------------
print('test data:');print('-'*20)
test = pd.read_csv('test.csv', parse_dates=['timestamp'],
                   dtype={'row_id':np.uint16,'building_id':np.uint16,'meter':np.uint16})
print(test.info(memory_usage='deep'))
#------------------------------------------------------------------------------
print('weather_train data:');print('-'*20)
weather_train = pd.read_csv('weather_train.csv',dtype={'site_id':np.uint16})
weather_train['timestamp'] = pd.to_datetime(weather_train['timestamp'],infer_datetime_format=True)
weather_train[['air_temperature', 'cloud_coverage',
       'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',
       'wind_direction', 'wind_speed']] = weather_train[['air_temperature', 'cloud_coverage',
       'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',
       'wind_direction', 'wind_speed']].apply(pd.to_numeric,downcast='float')
print(weather_train.info(memory_usage='deep'))
#------------------------------------------------------------------------------
print('weather_test data:');print('-'*20)
weather_test = pd.read_csv('weather_test.csv',dtype={'site_id':np.uint16})
weather_test['site_id'] = weather_test['site_id'].apply(pd.to_numeric,downcast='unsigned')
weather_test['timestamp'] = pd.to_datetime(weather_test['timestamp'],infer_datetime_format=True)
weather_test[['air_temperature', 'cloud_coverage',
       'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',
       'wind_direction', 'wind_speed']] = weather_test[['air_temperature', 'cloud_coverage',
       'dew_temperature', 'precip_depth_1_hr', 'sea_level_pressure',
       'wind_direction', 'wind_speed']].apply(pd.to_numeric,downcast='float')
print(weather_test.info(memory_usage='deep'))
#------------------------------------------------------------------------------
print('building_metadata data:');print('-'*20)
building_metadata = pd.read_csv('building_metadata.csv')
building_metadata['primary_use'] = building_metadata['primary_use'].astype('category')
print(building_metadata.info(memory_usage='deep'))
#------------------------------------------------------------------------------
sample_submission = pd.read_csv('sample_submission.csv')
print('sample_submission:',sample_submission.shape)
#------------------------------------------------------------------------------
plt.figure(figsize = (15,5))
train['meter_reading'].plot()
plt.show()
#------------------------------------------------------------------------------
total = train.isnull().sum().sort_values(ascending = False)
percent = (train.isnull().sum()/train.isnull().count()*100).sort_values(ascending = False)
missing_train_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_train_data.head())
#------------------------------------------------------------------------------
total = weather_train.isnull().sum().sort_values(ascending = False)
percent = (weather_train.isnull().sum()/weather_train.isnull().count()*100).sort_values(ascending = False)
missing_weather_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
print(missing_weather_data.head(9))
#------------------------------------------------------------------------------
