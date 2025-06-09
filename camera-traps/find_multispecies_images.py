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
import random

from tqdm import tqdm
from collections import defaultdict

from megadetector.utils.ct_utils import sort_dictionary_by_value
from megadetector.utils.url_utils import parallel_download_urls

base_url = 'https://storage.googleapis.com/public-datasets-lila/snapshot-safari-2024-expansion/'
metadata_fn = 'g:/temp/snapshot_safari_2024_metadata.json'
n_images_to_download = 400
output_folder = 'g:/temp/multispecies-test-images'

# Optionally restrict to one dataset
required_prefix = None # 'KAR'

# Should we choose the images with the highest numbers of species, or randomly sample 
# multi-species images?
sample_by_count = False
sampling_seed = 10

flatten_output_paths = True

assert os.path.isfile(metadata_fn)
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
if sample_by_count:
    image_id_to_count = sort_dictionary_by_value(image_id_to_count,reverse=True)
    image_ids_to_download = set(list(image_id_to_count.keys())[0:n_images_to_download])
else:
    random.seed(sampling_seed)
    image_ids_with_multiple_species = [image_id for image_id in image_id_to_count if \
                                       image_id_to_count[image_id] > 1]
    image_ids_to_download = random.sample(image_ids_with_multiple_species,n_images_to_download)
     

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
    if flatten_output_paths:
        output_filename_relative = p.replace('\\','/').replace('/','#')
    else:
        output_filename_relative = p
    output_file = os.path.join(output_folder,output_filename_relative)
    url_to_target_file[url] = output_file

print('Downloading {} files'.format(len(url_to_target_file)))


#%% Download

_ = parallel_download_urls(url_to_target_file)
