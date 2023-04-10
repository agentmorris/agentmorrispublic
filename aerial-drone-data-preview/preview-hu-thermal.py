#
# Code to render sample images and count annotations in the Hu et al.
# aerial image dataset.
#
# https://data.mendeley.com/datasets/46k66mz9sz/4
#
# Annotations are boxes in a .csv file.
#

#%% Constants and imports

import pandas as pd
import os

base_folder = r'C:\drone-data\03 - hu'
annotation_file = r"00_UAV-derived Thermal Waterfowl Dataset\00_UAV-derived Waterfowl Thermal Imagery Dataset\01_Thermal Images and Ground Truth (used for detector training and testing)\02_Groundtruth Label for Positive Images\Bounding Box Label.csv"
positive_image_folder = r"00_UAV-derived Thermal Waterfowl Dataset\00_UAV-derived Waterfowl Thermal Imagery Dataset\01_Thermal Images and Ground Truth (used for detector training and testing)\01_Posistive Image"
negative_image_folder = r"00_UAV-derived Thermal Waterfowl Dataset\00_UAV-derived Waterfowl Thermal Imagery Dataset\01_Thermal Images and Ground Truth (used for detector training and testing)\03_Negative Images"

annotation_file = os.path.join(base_folder,annotation_file)
positive_image_folder = os.path.join(base_folder,positive_image_folder)
negative_image_folder = os.path.join(base_folder,negative_image_folder)

assert os.path.isfile(annotation_file)
assert os.path.isdir(positive_image_folder)
assert os.path.isdir(negative_image_folder)

# 
# The annotation columns are:
# 
# filename, x, y, w, h
#


#%% List all positive images

images_relative = os.listdir(positive_image_folder)
    
    
#%% Read the annotation file

df = pd.read_csv(annotation_file)

print('Read {} annotations'.format(len(df)))


#%% Count annotations for each image

from collections import defaultdict
image_name_to_count = defaultdict(int)

for i_row,row in df.iterrows():
    fn = row['imageFilename']
    assert fn in images_relative        
    image_name_to_count[fn] += 1


#%% Render annotations for an image that has a decent number of annotations

import operator

# Sort in descending order by value
images_sorted_by_count = dict(sorted(image_name_to_count.items(), 
                                 key=operator.itemgetter(1),reverse=True))

images_sorted_by_count = list(images_sorted_by_count)
image_name = images_sorted_by_count[2]

image_full_path = os.path.join(positive_image_folder,image_name)

assert os.path.isfile(image_full_path)

image_annotations = []

for i_row,row in df.iterrows():
    if row['imageFilename'] == image_name:
        ann = {}
        ann['x'] = row['x(column)']
        ann['y'] = row['y(row)']
        ann['w'] = row['width']
        ann['h'] = row['height']
        image_annotations.append(ann)

print('Found {} annotations for image {}'.format(
    len(image_annotations),image_full_path))


#%% Render boxes

from visualization import visualization_utils as visutils

detection_formatted_boxes = []

pil_im = visutils.open_image(image_full_path)
image_w = pil_im.size[0]
image_h = pil_im.size[1]

category_id_to_name = {'0':''}
# ann = image_annotations[0]
for ann in image_annotations:
    
    det = {}
    det['conf'] = None
    det['category'] = '0'
    
    # Convert to relative x/y/w/h
    box = [ann['x']/image_w,
           ann['y']/image_h,
           (ann['w'])/image_w,
           (ann['h'])/image_h]           
    
    det['bbox'] = box    
    detection_formatted_boxes.append(det)
    
output_file = r'g:\temp\hu_thermal_sample_image_annotated.jpg'
visutils.draw_bounding_boxes_on_file(image_full_path, output_file, detection_formatted_boxes,
                                     confidence_threshold=0.0,detector_label_map=category_id_to_name,
                                     thickness=1,expansion=0)


import shutil
shutil.copyfile(image_full_path,r'g:\temp\hu_thermal_sample_image_unannotated.tif')

import path_utils
path_utils.open_file(output_file)