########
#
# usgs-geese-copy-val-images.py
#
# Because our inference mechanics really want to operate on a *folder* of images,
# but we want to run the model on our val images, this script creates a folder with all
# our val images in it, and also samples N images that are in neither the val nor train sets
# (generally empty images).
#
########

#%% Constants and imports

import os
import json
import shutil

from tqdm import tqdm

image_folder_base = '/media/user/My Passport/2017-2019/01_JPGs'
assert os.path.isdir(image_folder_base)
assert image_folder_base[-1] != '/'

output_folder_base = os.path.expanduser('~/data/usgs-geese')
train_images_list_file = os.path.join(output_folder_base,'train_images.json')
val_images_list_file = os.path.join(output_folder_base,'val_images.json')

assert all([os.path.isfile(fn) for fn in [train_images_list_file,val_images_list_file]])

target_folder_base = os.path.join(output_folder_base,'eval_images')
target_folder_val = os.path.join(target_folder_base,'val-images')
os.makedirs(target_folder_val,exist_ok=True)

target_folder_unused = os.path.join(target_folder_base,'unused-images')
os.makedirs(target_folder_unused,exist_ok=True)

with open(val_images_list_file,'r') as f:
    val_images_relative = json.load(f)

with open(train_images_list_file,'r') as f:
    train_images_relative = json.load(f)

n_unused_images_to_sample = 1000

images_copied = set()


#%% Get the total size of all val images

import humanfriendly

total_size_bytes = 0

# fn_relative = val_images_relative[0]
for fn_relative in val_images_relative:
    fn_abs = os.path.join(image_folder_base,fn_relative)
    total_size_bytes += os.stat(fn_abs).st_size
    
print('Total val data set size: {} in {} files'.format(
    humanfriendly.format_size(total_size_bytes),
    len(val_images_relative)))


#%% Copy val images to the output folder

# fn = val_images_relative[0]
for fn in tqdm(val_images_relative):
    
    source_fn = os.path.join(image_folder_base,fn)
    assert os.path.isfile(source_fn)
    
    target_name_relative = fn.replace('/','_')
    target_fn = os.path.join(target_folder_val,target_name_relative)
    assert source_fn not in images_copied    
    images_copied.add(source_fn)
    shutil.copyfile(source_fn,target_fn)
    

#%% Find N random files that are in neither the val nor train data

import random
from md_utils import path_utils

all_images = path_utils.find_images(image_folder_base,recursive=True)

# Remove the 'out lagoon' images
all_images = [fn for fn in all_images if 'out' not in fn]

val_images_absolute_set = set([os.path.join(image_folder_base,fn) for fn in val_images_relative])
train_images_absolute_set = set([os.path.join(image_folder_base,fn) for fn in train_images_relative])

print('{} val images, {} train images'.format(len(val_images_absolute_set),len(train_images_absolute_set)))

all_unused_images = []

for fn in all_images:
    if (fn not in val_images_absolute_set) and (fn not in train_images_absolute_set):
        all_unused_images.append(fn)
        
print('{} of {} images were unused in train or val'.format(
    len(all_unused_images),len(all_images)))

random.seed(0)
unused_images_to_copy = random.sample(all_unused_images,n_unused_images_to_sample)


#%% Copy unused images to the target folder

# fn_abs = unused_images_to_copy[0]
for fn_abs in tqdm(unused_images_to_copy):
    
    assert os.path.isfile(fn_abs)
    
    fn_rel = os.path.relpath(fn_abs,image_folder_base)
    target_name_relative = fn_rel.replace('/','_')
    target_fn = os.path.join(target_folder_unused,target_name_relative)
    assert fn_abs not in images_copied
    images_copied.add(fn_abs)
    shutil.copyfile(fn_abs,target_fn)

