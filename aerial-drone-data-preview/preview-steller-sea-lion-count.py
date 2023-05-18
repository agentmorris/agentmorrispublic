#
# Code to preview metadata for the "NOAA Fisheries Steller Sea Lion Population Count" dataset:
#
# https://www.kaggle.com/competitions/noaa-fisheries-steller-sea-lion-population-count
#

#%% Imports and constants

import os
import pandas as pd

train_folder = r'g:\temp\drone-datasets\noaa-fisheries-steller-sea-lion-population-count\Train'
train_metadata_file = r'train.csv'
train_metadata_path = os.path.join(train_folder,train_metadata_file)

assert os.path.isdir(train_folder)
assert os.path.isfile(train_metadata_path)


#%% Process metadata, make sure files exist

df = pd.read_csv(train_metadata_path)

print('Loaded metadata for {} images'.format(len(df)))

n_animals = 0

for i_row,row in df.iterrows():
    train_id = row['train_id']
    image_file = os.path.join(train_folder,str(train_id) + '.jpg')
    assert os.path.isfile(image_file)
    n_animals += (row['adult_males']+row['subadult_males']+row['adult_females']+row['juveniles']+row['pups'])

print('Counts total {} animals'.format(n_animals))