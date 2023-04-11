#
# Code to render sample images and count annotations in the Eikelboom et al.
# aerial image datasets.
#
# https://data.4tu.nl/articles/dataset/Improving_the_precision_and_accuracy_of_animal_population_estimates_with_aerial_image_object_detection/12713903/1
#
# Annotations are boxes in a .csv file.
#

#%% Constants and imports

import pandas as pd
import operator
import os
import shutil

import path_utils
from visualization import visualization_utils as visutils

base_folder = r'c:\drone-data\02 - eikelboom'

image_folders = ['test','train','val']

expected_species = ['zebra','elephant','giraffe']

# 
# The annotation columns are:
# 
# [filename], [x1], [x2], [y1], [y2], [species]
#
#

output_file_annotated = r'g:\temp\eikelboom_savanna_sample_image_annotated.jpg'
output_file_unannotated = r'g:\temp\eikelboom_savanna_sample_image_unannotated.jpg'


#%% List all images

# ...and map them to the folder that contains them
image_name_to_folder = {}

# image_folder = image_folders[0]
for image_folder in image_folders:
    image_files = os.listdir(os.path.join(base_folder,image_folder))
    for fn in image_files:
        assert fn.lower().endswith('.jpg') or fn.lower.endswith('.png')
        assert fn not in image_name_to_folder
        image_name_to_folder[fn] = image_folder
    
    
#%% Read the annotation file

df = pd.read_csv(os.path.join(base_folder,'annotations_images.csv'))


#%% Count annotations for each image

from collections import defaultdict
image_name_to_count = defaultdict(int)

for i_row,row in df.iterrows():
    image_name_to_count[row['FILE']] += 1


#%% Render annotations for an image that has a decent number of annotations

# Sort in descending order by value
sorted_annotations = dict(sorted(image_name_to_count.items(), 
                                 key=operator.itemgetter(1),reverse=True))

sorted_annotations = list(sorted_annotations)
image_name = sorted_annotations[1]

image_folder = image_name_to_folder[image_name]
image_full_path = os.path.join(base_folder,image_folder,image_name)
assert os.path.isfile(image_full_path)

image_annotations = []

for i_row,row in df.iterrows():
    if row['FILE'] == image_name:
        ann = {}
        ann['species'] = row['SPECIES'].lower()
        for s in ['x1','y1','x2','y2']:
            ann[s] = row[s]
        image_annotations.append(ann)

print('Found {} annotations for image {}'.format(
    len(image_annotations),image_full_path))


#%% Render boxes

detection_formatted_boxes = []

pil_im = visutils.open_image(image_full_path)
image_w = pil_im.size[0]
image_h = pil_im.size[1]

category_id_to_name = {}
category_name_to_id = {}
for i_s,s in enumerate(expected_species):
    cat_id = str(i_s)
    category_id_to_name[cat_id] = s
    category_name_to_id[s] = cat_id
    
# ann = image_annotations[0]
for ann in image_annotations:
    
    det = {}
    det['conf'] = None
    det['category'] = category_name_to_id[ann['species']]
    
    # Convert to relative x/y/w/h
    box = [ann['x1']/image_w,
           ann['y1']/image_h,
           (ann['x2']-ann['x1'])/image_w,
           (ann['y2']-ann['y1'])/image_h]           
    
    det['bbox'] = box    
    detection_formatted_boxes.append(det)
    
visutils.draw_bounding_boxes_on_file(image_full_path, output_file_annotated, detection_formatted_boxes,
                                     confidence_threshold=0.0,detector_label_map=category_id_to_name)
path_utils.open_file(output_file_annotated)

shutil.copyfile(image_full_path,output_file_unannotated)
