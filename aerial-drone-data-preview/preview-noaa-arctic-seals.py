#
# Code to render sample images and count annotations in the "NOAA Arctic Seals" dataset:
#
# https://lila.science/datasets/noaa-arctic-seals-2019/
#

#%% Constants and imports

import os
import pandas as pd
from tqdm import tqdm

from visualization import visualization_utils as visutils

annotation_csv_file = r"G:\temp\drone-datasets\noaa-arctic-seals\surv_test_kamera_detections_20210212_full_paths.csv"

file_base = r"I:\lila\noaa-kotz"

#%% Read and summarize annotations

df = pd.read_csv(annotation_csv_file)

species = list(set(df.detection_type))
species_string = ''
for s in species:
    species_string += s.lower() + ','
species_string = species_string[0:-1]

print('Species present:')
print(species_string)

print('Number of annotations:')
print(len(df))
# import clipboard; clipboard.copy(species_string)

category_id_to_name = {}
category_name_to_id = {}

for i_species in range(0,len(species)):
    category_id_to_name[i_species] = species[i_species]
    category_name_to_id[species[i_species]] = i_species


#%% Find unique RGB image files, count annotations, find average annotation size

import numpy as np
from collections import defaultdict

box_widths = []

rgb_path_to_ir_path = {}
rgb_path_to_annotation_rows = defaultdict(list)
for i_row,row in df.iterrows():
    
    rgb_path = row['rgb_image_path']
    ir_path = row['ir_image_path']
    rgb_path_to_annotation_rows[rgb_path].append(i_row)
    if isinstance(ir_path,str):
        if rgb_path in rgb_path_to_ir_path:
            assert rgb_path_to_ir_path[rgb_path] == ir_path
        else:
            rgb_path_to_ir_path[rgb_path] = ir_path

    box_width_pixels = row['rgb_right'] - row['rgb_left']
    assert box_width_pixels > 0
    box_widths.append(box_width_pixels)
    
print('Found {} annotations, average width {}'.format(len(box_widths),np.mean(box_widths)))

rgb_paths = sorted(list(rgb_path_to_ir_path.keys()))


#%% Find an image with a bunch of annotations

image_counts = set()

for i_image,rgb_path in enumerate(rgb_paths):
    annotations_rows = rgb_path_to_annotation_rows[rgb_path]
    if len(annotations_rows) > 30:
        break

print('Image {} has {} annotations'.format(i_image,len(annotations_rows)))


#%% Pick and render all annotations for one image file

i_image = 575
rgb_path = rgb_paths[i_image]
ir_path = rgb_path_to_ir_path[rgb_path]
annotations_rows = rgb_path_to_annotation_rows[rgb_path]

rgb_full_path = os.path.join(file_base,rgb_path)
assert os.path.isfile(rgb_full_path)

pil_im = visutils.open_image(rgb_full_path)
image_w = pil_im.size[0]
image_h = pil_im.size[1]

detection_formatted_boxes = []

# i_row = annotations_rows[0]
for i_row in annotations_rows:
        
    row = df.iloc[i_row]
    
    x0 = row['rgb_left']
    x1 = row['rgb_right']
    y1 = row['rgb_top']
    y0 = row['rgb_bottom']
    
    det = {}
    det['conf'] = None
    det['category'] = category_name_to_id[row['detection_type']]
    box_w = (x1-x0)
    box_h = (y1-y0)
    assert box_h > 0
    box = [x0/image_w,
           y0/image_h,
           box_w/image_w,
           box_h/image_h]
    
    det['bbox'] = box    
    detection_formatted_boxes.append(det)
    
output_file = r'g:\temp\noaa_arctic_seals_sample_image_annotated.jpg'
visutils.draw_bounding_boxes_on_file(rgb_full_path, output_file, detection_formatted_boxes,       
                                     confidence_threshold=0.0,detector_label_map=category_id_to_name,
                                     thickness=2,expansion=0)


import shutil
shutil.copyfile(rgb_full_path,r'g:\temp\noaa_arctic_seals_sample_image_unannotated.jpg')    
    
    
