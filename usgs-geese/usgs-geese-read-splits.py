########
#
# usgs-geese-read-splits.py
#
# When we trained the goose detection model, I split the data 80/20 (train/val) by 
# images, broke those images into patches, and wrote the corresponding patches out to
# YOLOv5-formatted annotations.  I was lazy and did not write out an explicit list of which
# images were in train/val.
#
# This script reads the YOLOv5-formatted list of *patches* to determine which *images* are
# in train/val, and writes that out to a more durable location.
#
########

#%% Imports and constants

import os
import glob

training_folder_base = '/home/user/data/usgs-geese'
training_folder_val = os.path.join(training_folder_base,'yolo_val')
training_folder_train = os.path.join(training_folder_base,'yolo_train')

assert all([os.path.isdir(fn) for fn in (training_folder_train,training_folder_val)])

image_folder_base = '/media/user/My Passport/2017-2019/01_JPGs'
assert os.path.isdir(image_folder_base)
assert image_folder_base[-1] != '/'

output_folder_base = os.path.expanduser('~/data/usgs-geese')
train_images_list_file = os.path.join(output_folder_base,'train_images.json')
val_images_list_file = os.path.join(output_folder_base,'val_images.json')

#%% Read train and val patch names


yolo_annotation_files_train = glob.glob(training_folder_train + '/*.txt')
for fn in yolo_annotation_files_train:
    assert os.path.isfile(fn.replace('.txt','.jpg'))

yolo_annotation_files_val = glob.glob(training_folder_val + '/*.txt')
for fn in yolo_annotation_files_val:
    assert os.path.isfile(fn.replace('.txt','.jpg'))

print('Enumerated {} train patches and {} val patches'.format(
    len(yolo_annotation_files_train),len(yolo_annotation_files_val)))


#%% Enumerate image files

image_files = glob.glob(image_folder_base + '/**/*.JPG',recursive=True)
image_files_relative = [fn.replace(image_folder_base + '/','') for fn in image_files]
image_files_relative_set = set(image_files_relative)


#%% Infer image names

#
# Filenames look like this:
#
# 2017_replicate_2017-09-30_cam1_293a0019_0000_0000.jpg    
# 
# The corresponding image name is:
#     
# 2017/Replicate_2017-09-30/Cam1/293A0019.JPG
#

def yolo_txt_file_to_image_file(fn):
    
    txt_file_relative = os.path.relpath(fn,training_folder_base)
    txt_file_relative = txt_file_relative.replace('yolo_val/','').replace(
        'yolo_train/','')
    
    image_id = txt_file_relative[0:-14]
    tokens = image_id.split('_')
    assert len(tokens) == 5
    
    _ = int(tokens[0])
    
    assert tokens[1] == 'replicate'
    
    assert 'cam' in tokens[3] 
    
    # This is not always true
    # assert 'cam' in tokens[4]
    
    image_file = tokens[0] + '/' + 'Replicate_' + tokens[2] + '/' + \
        tokens[3].replace('cam','Cam') + '/' + tokens[4].upper() + '.JPG'
        
    if image_file not in image_files_relative_set:
        image_file = image_file.replace('Cam','CAM')
        
    assert image_file in image_files_relative_set
    
    return image_file

train_image_files = set()
val_image_files = set()

# fn = yolo_annotation_files_train[0]
for fn in yolo_annotation_files_train:
    train_image_files.add(yolo_txt_file_to_image_file(fn))

for fn in yolo_annotation_files_val:
    val_image_files.add(yolo_txt_file_to_image_file(fn))

print('Found {} train images and {} val images'.format(
    len(train_image_files),len(val_image_files)))


train_image_files = sorted(list(train_image_files))
val_image_files = sorted(list(val_image_files))


#%% Verify that all of the images we think should exist actually exist

for i_fn,fn in enumerate(train_image_files):
    assert fn in image_files_relative

for i_fn,fn in enumerate(val_image_files):
    assert fn in image_files_relative


#%% Write train/val splits out to file

import json

with open(train_images_list_file,'w') as f:
    json.dump(train_image_files,f,indent=1)
    
with open(val_images_list_file,'w') as f:
    json.dump(val_image_files,f,indent=1)    