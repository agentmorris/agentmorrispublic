#
# Code to preview metadata for the "NOAA Right Whale Recognition" dataset:
#
# https://www.kaggle.com/c/noaa-right-whale-recognition
#
# Annotations are individual IDs in a .csv file.
#

#%% Imports and constants

import os
import pandas as pd

image_folder = r"G:\temp\drone-datasets\noaa-right-whale-recognition\imgs\imgs"
metadata_file = r"G:\temp\drone-datasets\noaa-right-whale-recognition\train.csv"


#%% Process metadata

df = pd.read_csv(metadata_file)

unique_whales = set(df['whaleID'])
unique_images = set(df['Image'])

print('Read metadata for {} unique whales in {} unique images ({} rows)'.format(
    len(unique_whales),len(unique_images),len(df)))

image_files = os.listdir(image_folder)
image_files = [fn for fn in image_files if fn.endswith('.jpg')]

print('Total images: {}'.format(len(image_files)))


#%% Make sure all files exist

missing_files = []

# i_row = 0; row = df.iloc[i_row]
for i_row,row in df.iterrows():
    image_name = row['Image']
    whale_id = row['whaleID']
    image_path = os.path.join(image_folder,image_name)
    # assert os.path.isfile(image_path)
    if not os.path.isfile(image_path):
        missing_files.append(image_name)

print('Missing {} of {} files'.format(len(missing_files),len(df)))
    
# Only one file is missing, not sure what happened to whale 4460

