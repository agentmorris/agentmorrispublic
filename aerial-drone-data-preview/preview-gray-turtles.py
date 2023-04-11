#
# Code to render sample images and count annotations in the Gray et al.
# drone image dataset:
#
# https://zenodo.org/record/5004596#.ZChnr3ZBxD8
#
# Annotations are points in a .csv file.
#

#%% Constants and imports

import pandas as pd
import operator
import os
import shutil
from collections import defaultdict

import path_utils
from visualization import visualization_utils as visutils

base_folder = r'c:\drone-data\07 - gray'
annotation_file = 'turtle_image_metadata.csv'

output_file_annotated = r'g:\temp\gray_turtles_sample_image_annotated.jpg'
output_file_unannotated = r'g:\temp\gray_turtles_sample_image_unannotated.jpg'

# The location columns in the metadata are labeled "top" and "left", which implies boxes, but
# widths and heights are not provided.  Points are definitely to the top-left of the turtles,
# so it looks like maybe there were boxes at some point. These are reasonable approximations.

box_width = 100
box_height = 100


#%% List all images

image_files = path_utils.find_images(base_folder,recursive=True)
image_files_relative = [os.path.relpath(fn,base_folder) for fn in image_files]
    
print('Enumerated {} images'.format(len(image_files_relative)))


#%% Read the annotation file

df = pd.read_csv(os.path.join(base_folder,annotation_file))

print('Read {} annotations'.format(len(df)))


#%% Map images to valid annotations

image_to_annotations = defaultdict(list)

n_annotations = 0

# i_row = 0; row = df.iloc[i_row]
for i_row,row in df.iterrows():
    
    # These are false positives
    if row['label'] != 'Certain Turtle':
        continue

    n_annotations += 1
    image_file_relative = os.path.join(row['file_location'],row['filename'])
        
    ann = {}
    for s in ['label','ImageHeight','ImageWidth','top','left']:
        ann[s] = row[s]
        ann['file_name'] = image_file_relative
        
    image_to_annotations[image_file_relative].append(ann)
    
print('Found {} annotations on {} images'.format(n_annotations,len(image_to_annotations)))

      
#%% Render annotations for an image that has a decent number of annotations

image_name_to_count = {}
for image_name in image_to_annotations:
    image_name_to_count[image_name] = len(image_to_annotations[image_name])
    
# Sort in descending order by value
images_sorted_by_count = dict(sorted(image_name_to_count.items(), 
                                 key=operator.itemgetter(1),reverse=True))

images_sorted_by_count = list(images_sorted_by_count)
image_name = images_sorted_by_count[0]

image_full_path = os.path.join(base_folder,image_name)
assert os.path.isfile(image_full_path)

image_annotations = image_to_annotations[image_name]

print('Found {} annotations for image {}'.format(
    len(image_annotations),image_full_path))


#%% Render boxes

detection_formatted_boxes = []

pil_im = visutils.open_image(image_full_path)
image_w = pil_im.size[0]
image_h = pil_im.size[1]

# ann = image_annotations[0]
for ann in image_annotations:
    
    assert ann['ImageHeight'] == image_h
    assert ann['ImageWidth'] == image_w
    
    det = {}
    det['conf'] = None
    det['category'] = '0'
    
    # Convert to relative x/y/w/h
    box = [ann['left']/image_w,
           ann['top']/image_h,
           box_width/image_w,
           box_height/image_h]           
    
    det['bbox'] = box    
    detection_formatted_boxes.append(det)
    
visutils.draw_bounding_boxes_on_file(image_full_path, output_file_annotated, detection_formatted_boxes,
                                     confidence_threshold=0.0,detector_label_map=None)
path_utils.open_file(output_file_annotated)

shutil.copyfile(image_full_path,output_file_unannotated)
