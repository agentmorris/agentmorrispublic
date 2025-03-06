"""

find_multispecies_images.py

Download images containing multiple species from the Snapshot Safari 2024 expansion dataset:
    
https://lila.science/datasets/snapshot-safari-2024-expansion/

Assumes you've already downloaded the metadata .json file from:
    
https://storage.googleapis.com/public-datasets-lila/snapshot-safari-2024-expansion/snapshot_safari_2024_metadata.zip

"""
    
#%% Imports and constants

import os
import json

from tqdm import tqdm
from collections import defaultdict

from megadetector.utils.ct_utils import sort_dictionary_by_value
from megadetector.utils.url_utils import parallel_download_urls

base_url = 'https://storage.googleapis.com/public-datasets-lila/snapshot-safari-2024-expansion/'
metadata_fn = '/mnt/g/temp/snapshot_safari_2024_metadata.json'
n_images_to_download = 200
output_folder = '/mnt/g/temp/multispecies-test-images'

# Optionally restrict to one dataset
required_prefix = 'KAR'

os.makedirs(output_folder,exist_ok=True)


#%% Read metadata

with open(metadata_fn,'r') as f:
    d = json.load(f)
    
image_id_to_annotations = defaultdict(list)


#%% Find images with multiple species

# ann = d['annotations']
for ann in tqdm(d['annotations']):
    image_id_to_annotations[ann['image_id']].append(ann)

image_id_to_count = {}
for image_id in image_id_to_annotations:
    if required_prefix is not None:
        if not image_id.startswith(required_prefix):
            continue
    image_id_to_count[image_id] = len(image_id_to_annotations[image_id])
    
# Sort in descending order by count
image_id_to_count = sort_dictionary_by_value(image_id_to_count,reverse=True)

image_ids_to_download = set(list(image_id_to_count.keys())[0:n_images_to_download])

     
#%% Find relative paths (which in practice are the image IDs, but let's be sure)

image_paths_to_download = []
for im in d['images']:
    if im['id'] in image_ids_to_download:
        image_paths_to_download.append(im['file_name'])
        
assert len(image_paths_to_download) == n_images_to_download


#%% Build a list of URLs

url_to_target_file = {}

# p = image_paths_to_download[0]
for p in image_paths_to_download:
    url = base_url + p
    output_file = os.path.join(output_folder,p)
    url_to_target_file[url] = output_file


#%% Downlopad

parallel_download_urls(url_to_target_file)
