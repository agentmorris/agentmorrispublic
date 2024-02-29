#
# Code to render sample images and annotations in the Delplanque et al dataset of African mammals:
#
# https://dataverse.uliege.be/file.xhtml?fileId=11098&version=1.0
#
# Annotations are boxes in COCO .json files.
#

#%% Constants and imports

import os
import json
import shutil
import operator
import numpy as np

from collections import defaultdict
from tqdm import tqdm

from md_visualization import visualization_utils as visutils
from md_utils import path_utils

annotation_files = {
    'train':'groundtruth/json/big_size/train_big_size_A_B_E_K_WH_WB.json',
    'val':'groundtruth/json/big_size/val_big_size_A_B_E_K_WH_WB.json',
    'test':'groundtruth/json/big_size/test_big_size_A_B_E_K_WH_WB.json'
}

base_folder = r'g:\temp\general_dataset\general_dataset'

for split_name in annotation_files.keys():
    annotation_file = os.path.join(base_folder,annotation_files[split_name])
    assert os.path.isfile(annotation_file), 'Could not find file {}'.format(annotation_file)

output_file_annotated = r'g:\temp\delplanque_mammals_sample_image_annotated.jpg'
output_file_unannotated = r'g:\temp\delplanque_mammals_sample_image_unannotated.jpg'


#%% Read and summarize annotations

abs_filename_to_annotations = defaultdict(list)
category_name_to_id = {}
category_id_to_count = defaultdict(int)

n_annotations = 0

box_widths = []

# split_name = next(iter(annotation_files))
for split_name in annotation_files:

    annotation_file_relative = annotation_files[split_name]
    annotation_file_abs = os.path.join(base_folder,annotation_file_relative)
    image_folder = os.path.join(base_folder,split_name)
    assert os.path.isdir(image_folder)
    
    with open(annotation_file_abs,'r') as f:
        annotation_data = json.load(f)
        
    # Map all annotations in this dataset to their corresponding images
    image_id_to_annotations = defaultdict(list)
    for ann in annotation_data['annotations']:
        image_id_to_annotations[ann['image_id']].append(ann)
    
    for cat in annotation_data['categories']:
        if cat['name'] in category_name_to_id:
            assert cat['id'] == category_name_to_id[cat['name']]
        else:
            category_name_to_id[cat['name']] = cat['id']
    
    # im = annotation_data['images'][0]
    for im in tqdm(annotation_data['images']):
        
        file_name_short = im['file_name']
        fn_abs = os.path.join(image_folder,file_name_short)
        assert os.path.isfile(fn_abs)
        
        annotations_this_image = image_id_to_annotations[im['id']]
        
        # ann = annotations_this_image[0]
        for ann in annotations_this_image:
            category_id = ann['category_id']
            category_id_to_count[category_id] = category_id_to_count[category_id] + 1
            assert 'bbox' in ann
            assert len(ann['bbox']) == 4
            abs_filename_to_annotations[fn_abs].append(ann)
            n_annotations += 1       
            
            box_widths.append(ann['bbox'][3])
            
        # ...for each annotation
        
    # ...for each image

# ...for each split

print('\nRead {} annotations (average width {}) for {} images'.format(
    n_annotations,np.mean(box_widths),len(abs_filename_to_annotations)))

print('Categories:')
for s in category_name_to_id.keys():
    print(s)


#%% Find an image with a bunch of annotations

required_token = None
prohibited_tokens = None # ['test']
abs_filename_to_annotations_selected = {}

# fn = list(abs_filename_to_annotations.keys())[1000]; print(fn)
for fn in list(abs_filename_to_annotations.keys()):
    if (required_token is None or required_token in fn):
        if prohibited_tokens is not None:
            found_prohibited_token = False
            for token in prohibited_tokens:
                if token in fn:
                    found_prohibited_token = True
                    break
            if found_prohibited_token:            
                continue
        abs_filename_to_annotations_selected[fn] = \
            abs_filename_to_annotations[fn]
            
image_name_to_count = {}
for image_name in abs_filename_to_annotations_selected:
    image_name_to_count[image_name] = len(abs_filename_to_annotations[image_name])
    
# Sort in descending order by value
sorted_annotations = dict(sorted(image_name_to_count.items(), 
                                 key=operator.itemgetter(1),reverse=True))

sorted_annotations = list(sorted_annotations)

image_full_path = sorted_annotations[0]

assert os.path.isfile(image_full_path)

image_annotations = abs_filename_to_annotations[image_full_path]

print('Found {} annotations for image {}'.format(
    len(image_annotations),image_full_path))


##%% Render all annotations for one image file

pil_im = visutils.open_image(image_full_path, ignore_exif_rotation=True)
image_w = pil_im.size[0]
image_h = pil_im.size[1]

detection_formatted_boxes = []
box_behaviors = []

# ann = image_annotations[0]
for ann in image_annotations:
        
    x0 = ann['bbox'][0]
    y0 = ann['bbox'][1]
    x1 = x0 + ann['bbox'][2]
    y1 = y0 + ann['bbox'][3]
    
    if y1 < y0:
        y0, y1 = y1, y0
    
    if x1 < x0:
        x0, x1 = x1, x0
        
    det = {}
    det['conf'] = None
    det['category'] = ann['category_id']
    box_w = (x1-x0)
    box_h = (y1-y0)
    assert box_h > 0
    box = [x0/image_w,
           y0/image_h,
           box_w/image_w,
           box_h/image_h]
    
    det['bbox'] = box    
    detection_formatted_boxes.append(det)    
    
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
                                     thickness=8,expansion=1, colormap=['red'], ignore_exif_rotation=True)

shutil.copyfile(image_full_path,output_file_unannotated)    
path_utils.open_file(output_file_annotated)
