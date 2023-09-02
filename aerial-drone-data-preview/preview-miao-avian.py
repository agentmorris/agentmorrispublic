#
# Code to process metadata and copy sample images for the Miao et al avian classification dataset:
#
# https://www.sciencebase.gov/catalog/item/63bf21cdd34e92aad3cdac5a    
#

#%% Imports and constants

import os
import json

from collections import defaultdict

image_folder = r'g:\temp\avian\images'

# Filenames include 'images/'
image_folder_base = r'g:\temp\avian'
annotation_folder = r'g:\temp\avian\annotations'
metadata_files = ['test.txt','train.txt','val.txt']
labelmap_file = os.path.join(annotation_folder,'labelmap.txt')
metadata_files = [os.path.join(annotation_folder,fn) for fn in metadata_files]

output_folder = r'g:\temp'


#%% Process metadata

with open(labelmap_file,'r') as f:
    lines = f.readlines()
    
category_to_name = {}

# s = lines[0]
for s in lines:
    d = json.loads(s.replace("'",'"'))
    category_to_name[d['id']] = d['name']

category_to_count = defaultdict(int)

n_annotations = 0

image_filenames = []

# metadata_file = metadata_files[0]
for metadata_file in metadata_files:
    
    with open(metadata_file,'r') as f:
        lines = f.readlines()
    lines = [s.strip() for s in lines]
    for s in lines:
        tokens = s.split(' ')
        assert len(tokens) == 2
        fn = tokens[0]        
        fn_abs = os.path.join(image_folder_base,fn)
        image_filenames.append(fn_abs)
        assert os.path.isfile(fn_abs)
        category_id = int(tokens[1])
        # assert category_id in category_to_name
        category_to_count[category_id] = category_to_count[category_id] + 1
        n_annotations += 1
    

#%% Randomly sample N patches

import random
import shutil

n = 8
sampled_images = random.sample(image_filenames,n)

for src in sampled_images:
    dst = os.path.join(output_folder,os.path.basename(src))
    shutil.copyfile(src,dst)
