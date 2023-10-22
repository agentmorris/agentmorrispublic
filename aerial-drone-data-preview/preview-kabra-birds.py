#
# Code to render sample images and annotations in the Kabra et al. birds dataset:
#
# https://ieeexplore.ieee.org/document/10069986
#
# Annotations are boxes in .csv files.
#

#%% Constants and imports

import os
import pandas as pd
import shutil
import glob
import operator
import numpy as np

from collections import defaultdict
from tqdm import tqdm

from md_visualization import visualization_utils as visutils
from md_utils import path_utils

base_folder = r'g:\temp\F21-S22-Combined-D2K-Audubon\Good annotations'
output_file_annotated = r'g:\temp\kabra_birds_sample_image_annotated.jpg'
output_file_unannotated = r'g:\temp\kabra_birds_sample_image_unannotated.png'


#%% Read and summarize annotations

relative_filename_to_annotations = defaultdict(list)
n_annotations = 0
category_name_to_id = {}

csv_files = glob.glob(base_folder + '/*.csv')

# Mapping from class IDs to category names is not strictly unique
class_id_to_category_names = defaultdict(set)
relative_filename_to_annotations = defaultdict(list)

box_widths = []

# annotation_csv_file = csv_files[0]
for annotation_csv_file in tqdm(csv_files):
    
    image_filename = annotation_csv_file.replace('.csv','.jpg')
    assert os.path.isfile(image_filename)
    
    df = pd.read_csv(annotation_csv_file)
    n_annotations += len(df)
    
    # i_row = 0; row = df.iloc[i_row]
    for i_row,row in df.iterrows():
    
        class_id = row['class_id']
        class_id_to_category_names[class_id].add(row['desc'])            
                
        ann = {}
        image_filename_relative = os.path.basename(image_filename)
        ann['file'] = image_filename_relative
        ann['class_id'] = row['class_id']
        for s in ['x','y','width','height']:
            ann[s] = row[s]

        box_widths.append(ann['width'])
        relative_filename_to_annotations[image_filename_relative].append(ann)
        
    # ...for each row in this csv file    

# ...for each csv file        

print('Read {} annotations for {} images, average width {}'.format(
    n_annotations,len(relative_filename_to_annotations),np.mean(box_widths)))

print('Categories:')
for s in category_name_to_id.keys():
    print(s)


#%% Print category names

category_names = class_id_to_category_names.values()
for names in category_names:
    for s in names:
        print(s)
        

#%% Find an image with a bunch of annotations

image_name_to_count = {}
for image_name in relative_filename_to_annotations:
    image_name_to_count[image_name] = len(relative_filename_to_annotations[image_name])
    
# Sort in descending order by value
sorted_annotations = dict(sorted(image_name_to_count.items(), 
                                 key=operator.itemgetter(1),reverse=True))

sorted_annotations = list(sorted_annotations)

# Representative of the camera trap images, which is just one dataset
# image_relative_path = sorted_annotations[192]
image_relative_path = sorted_annotations[60]

image_full_path = os.path.join(base_folder,image_relative_path)
assert os.path.isfile(image_full_path)

image_annotations = relative_filename_to_annotations[image_relative_path]

print('Found {} annotations for image {}'.format(
    len(image_annotations),image_full_path))


#%% Pick and render all annotations for one image file

pil_im = visutils.open_image(image_full_path)
image_w = pil_im.size[0]
image_h = pil_im.size[1]

detection_formatted_boxes = []

category_name_to_id = defaultdict(int)
next_category_id = 0

# ann = image_annotations[0]
for ann in image_annotations:
        
    x0 = ann['x']
    x1 = ann['width'] + x0
    y0 = ann['y']
    y1 = ann['height'] + y0
    
    det = {}
    det['conf'] = None
    category_name = next(iter(class_id_to_category_names[ann['class_id']]))
    if category_name not in category_name_to_id:
        next_category_id += 1
        category_id = next_category_id
        category_name_to_id[category_name] = category_id
    else:
        category_id = category_name_to_id[category_name]
    det['category'] = category_id
    box_w = (x1-x0)
    box_h = (y1-y0)
    assert box_h > 0
    box = [x0/image_w,
           y0/image_h,
           box_w/image_w,
           box_h/image_h]
    
    det['bbox'] = box    
    detection_formatted_boxes.append(det)
    
category_id_to_name = {}
for category_name in category_name_to_id.keys():
    category_id = category_name_to_id[category_name]
    category_id_to_name[category_id] = category_name
    
visutils.draw_bounding_boxes_on_file(image_full_path, output_file_annotated, detection_formatted_boxes,       
                                     confidence_threshold=0.0,detector_label_map=category_id_to_name,
                                     thickness=10,expansion=0)

shutil.copyfile(image_full_path,output_file_unannotated)    
path_utils.open_file(output_file_annotated)
