#
# Code to preview metadata for the "KABR" dataset:
#
# https://dirtmaxim.github.io/kabr/
#

#%% Imports and constants

import os
import json

base_folder = r'C:\temp\KABR\KABR'
annotation_folder = os.path.join(base_folder,'annotation')
image_folder = os.path.join(base_folder,'dataset','image')

class_id_file = os.path.join(annotation_folder,'classes.json')
train_label_file = os.path.join(annotation_folder,'train.csv')
val_label_file = os.path.join(annotation_folder,'val.csv')

assert os.path.isdir(annotation_folder) and os.path.isdir(image_folder)
assert os.path.isfile(class_id_file)
assert os.path.isfile(train_label_file) and os.path.isfile(val_label_file)

output_file_annotated = r'c:\temp\kabr_sample_image_annotated.jpg'
output_file_unannotated = r'c:\temp\kabr_sample_image_unannotated.jpg'


#%% Read annotation files

with open(class_id_file,'r') as f:
    category_name_to_id = json.load(f)
    
category_id_to_name = {v: k.lower() for k, v in category_name_to_id.items()}

def parse_label_file(fn):
    
    with open(fn,'r') as f:
        lines = f.readlines()
    lines = [s.strip() for s in lines]
   
    # Columns are:
    # 
    # original_vido_id video_id frame_id path labels
    # 
    # The first column name is not a typo here, this matches the metadata.
    assert lines[0].startswith('original_vido_id')
    
    filename_to_category = {}
    
    # i_line = 1
    for i_line in range(1,len(lines)):
        line = lines[i_line]
        tokens = line.split(' ')
        assert len(tokens) == 5
        category_id = tokens[-1]
        category_id = int(category_id)
        frame_file = tokens[3]
        assert '/' in frame_file
        assert frame_file not in filename_to_category
        filename_to_category[frame_file] = category_id
           
    return filename_to_category

# fn = train_label_file
train_filename_to_category = parse_label_file(train_label_file)
val_filename_to_category = parse_label_file(val_label_file)
filename_to_category = {**train_filename_to_category,**val_filename_to_category}
assert len(filename_to_category) == len(train_filename_to_category) + len(val_filename_to_category)


#%% Enumerate images

from megadetector.utils.path_utils import find_images

image_files_relative = find_images(image_folder,return_relative_paths=True,recursive=True)

print('Enumerated {} images in {}'.format(len(image_files_relative),image_folder))


#%% Pick an image at random

import random
random.seed(0)

image_fn_relative = random.choice(image_files_relative)
image_fn_abs = os.path.join(image_folder,image_fn_relative)

# from md_utils.path_utils import open_file; open_file(image_fn_abs)

assert image_fn_relative[0] in ('Z','G')
if image_fn_relative[0] == 'G':
    species = 'giraffe'
else:
    species = 'zebra'
    
behavior_category_id = filename_to_category[image_fn_relative.replace('\\','/')]
behavior_category_name = category_id_to_name[behavior_category_id]

image_string = 'species: {}, behavior: {}'.format(species,behavior_category_name)
print(image_string)


#%% Copy unannotated image

import shutil
shutil.copyfile(image_fn_abs,output_file_unannotated)


#%% Render annotated image

from PIL import Image, ImageDraw, ImageFont

im = Image.open(image_fn_abs)
im_draw = ImageDraw.Draw(im)
font = ImageFont.truetype('arial.ttf', 20)
im_draw.text((10,10), image_string, fill=(255,255,255), font=font)
im.save(output_file_annotated)
