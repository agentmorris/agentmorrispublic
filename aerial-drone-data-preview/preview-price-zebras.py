#
# Code to render sample images and annotations in the Price et al. aerial zebra dataset:
#
# https://keeper.mpdl.mpg.de/d/a9822e000aff4b5391e1
#
# Annotations are boxes in Labelme .json files.
#

#%% Constants and imports

import os
import json
import shutil
import glob
import operator
import numpy as np

from collections import defaultdict
from tqdm import tqdm

from md_visualization import visualization_utils as visutils
from md_utils import path_utils

base_folder = r'G:\temp\aerial-dataset'

# Within this folder:
#
# "Round1" is manual annotations
# "Round2" is semi-automated box annotations, with manual behavior annotations
#
base_video_folder = os.path.join(base_folder,'annotated_videos')

# This folder is a standard COCO dataset
#
# In this script, we're not going to tinker with this folder.
base_detection_folder = os.path.join(base_folder,'detector_network_dataset')

# This folder is a classification dataset with fixed-size chips
#
# In this script, we're not going to tinker with this folder.
base_classification_folder = os.path.join(base_folder,'classifier_network_dataset')
                                     
output_file_annotated = r'g:\temp\price_zebras_sample_image_annotated.jpg'
output_file_unannotated = r'g:\temp\price_zebras_sample_image_unannotated.png'

assert all([os.path.isdir(fn) for fn in [base_video_folder,base_detection_folder,base_classification_folder]])


#%% Read and summarize annotations

relative_filename_to_annotations = defaultdict(list)

annotation_files = glob.glob(base_video_folder + '/**/*.json',recursive=True)
box_widths = []
n_annotations_round1 = 0
n_annotations_round2 = 0
n_bad_points = 0

category_name_to_id = {}

# annotation_file = annotation_files[0]
for annotation_file in tqdm(annotation_files):
    
    with open(annotation_file,'r') as f:
        d = json.load(f)    
    
    assert d['imagePath'].startswith('../')

    fn_abs = os.path.join(os.path.dirname(annotation_file),d['imagePath'])
    assert os.path.isfile(fn_abs)
    fn_relative = os.path.relpath(fn_abs,base_video_folder)
    
    # i_shape = 0; shape = d['shapes'][i_shape]
    for i_shape,shape in enumerate(d['shapes']):
            
        label = shape['label']
        assert '_' in label
        category_name = label.split('_')[0]
        assert shape['shape_type'] == 'rectangle'
        # assert len(shape['points']) == 2
        if len(shape['points']) != 2:
            n_bad_points += 1
            continue
        
        x0 = shape['points'][0][0]
        y0 = shape['points'][0][1]
        x1 = shape['points'][1][0]
        y1 = shape['points'][1][1]
        
        box_widths.append(x1 - x0)
        
        if category_name not in category_name_to_id:
            category_name_to_id[category_name] = len(category_name_to_id)
        
        relative_filename_to_annotations[fn_relative].append(shape)
    
        if 'round1' in annotation_file.lower():
            n_annotations_round1 += 1
        else:
            assert 'round2' in annotation_file.lower()
            n_annotations_round2 += 1
                
    # ...for each shape in this annotation file
        
# ...for each annotation file

n_annotations = n_annotations_round1 + n_annotations_round2

print('Read {} annotations for ({} round 1 (manual)), {} round 2 (semi-automated) average width {}'.format(
    n_annotations,n_annotations_round1,n_annotations_round2,np.mean(box_widths)))

print('{} total images'.format(len(relative_filename_to_annotations)))

print('{} bad annotations (rectangles with the wrong number of corner points)'.format(n_bad_points))

print('Categories:')
for s in category_name_to_id.keys():
    print(s)


#%% Find an image with a bunch of annotations

use_only_round1_images = True

image_name_to_count = {}
for image_name in relative_filename_to_annotations:
    if use_only_round1_images:
        if not 'round1' in image_name:
            continue
    image_name_to_count[image_name] = len(relative_filename_to_annotations[image_name])
    
# Sort in descending order by value
sorted_annotations = dict(sorted(image_name_to_count.items(), 
                                 key=operator.itemgetter(1),reverse=True))

sorted_annotations = list(sorted_annotations)

image_relative_path = sorted_annotations[30]

image_full_path = os.path.join(base_video_folder,image_relative_path)
assert os.path.isfile(image_full_path)

image_annotations = relative_filename_to_annotations[image_relative_path]

print('Found {} annotations for image {}'.format(
    len(image_annotations),image_full_path))


##%% Pick and render all annotations for one image file

pil_im = visutils.open_image(image_full_path)
image_w = pil_im.size[0]
image_h = pil_im.size[1]

detection_formatted_boxes = []
box_behaviors = []

# shape = image_annotations[0]
for shape in image_annotations:
        
    x0 = shape['points'][0][0]
    y0 = shape['points'][0][1]
    x1 = shape['points'][1][0]
    y1 = shape['points'][1][1]
    
    if y1 < y0:
        y0, y1 = y1, y0
    
    if x1 < x0:
        x0, x1 = x1, x0
        
    det = {}
    det['conf'] = None
    det['category'] = category_name_to_id[shape['label'].split('_')[0]]
    box_w = (x1-x0)
    box_h = (y1-y0)
    assert box_h > 0
    box = [x0/image_w,
           y0/image_h,
           box_w/image_w,
           box_h/image_h]
    
    det['bbox'] = box    
    detection_formatted_boxes.append(det)
    behavior = ''
    for flag in shape['flags'].keys():
        if shape['flags'][flag]:
            behavior += ' (' + flag + ')'
    box_behaviors.append(behavior)
    
"""
def draw_bounding_boxes_on_file(input_file, output_file, detections, confidence_threshold=0.0,
                                detector_label_map=DEFAULT_DETECTOR_LABEL_MAP,
                                thickness=DEFAULT_BOX_THICKNESS, expansion=0,
                                colormap=DEFAULT_COLORS,
                                custom_strings=None):

"""    

category_id_to_name = {v: k for k, v in category_name_to_id.items()}

visutils.draw_bounding_boxes_on_file(image_full_path, output_file_annotated, detection_formatted_boxes,       
                                     confidence_threshold=0.0,detector_label_map=category_id_to_name,
                                     thickness=8,expansion=1, colormap=['red'],
                                     custom_strings=box_behaviors)

shutil.copyfile(image_full_path,output_file_unannotated)    
path_utils.open_file(output_file_annotated)
