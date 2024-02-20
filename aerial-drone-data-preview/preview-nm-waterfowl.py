#
# Code to render sample images and annotations in the "UAS Imagery of Migratory Waterfowl"
# datasetL:
#
# https://lila.science/datasets/uas-imagery-of-migratory-waterfowl-at-new-mexico-wildlife-refuges/
#
# Annotations are boxes in COCO .json files.
#

#%% Constants and imports

import ast
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
    'crowdsourced':
        {'raw':"crowdsourced/20240209_dronesforducks_zooniverse_raw.json",
         'consensus':"crowdsourced/20240220_dronesforducks_zooniverse_refined.json"},
    'expert':
        {'raw':"experts/20230331_dronesforducks_raw_experts.json",
         'consensus':"experts/20230331_dronesforducks_expert_refined.json"}
}

base_folder = r'g:\temp\uas-imagery-of-migratory-waterfowl'

for annotator_pool in ('crowdsourced','expert'):
    for annotation_type in ('raw','consensus'):
        filename = os.path.join(base_folder,annotation_files[annotator_pool][annotation_type])
        assert os.path.isfile(filename), 'Could not find file {}'.format(filename)


output_file_annotated = r'g:\temp\nm_waterfowl_sample_image_annotated.jpg'
output_file_unannotated = r'g:\temp\nm_waterfowl_sample_image_unannotated.jpg'


#%% Read and summarize annotations

annotation_file_relative = annotation_files['expert']['consensus']
annotation_file_abs = os.path.join(base_folder,annotation_file_relative)
image_folder = os.path.join(os.path.dirname(annotation_file_abs),'images')
assert os.path.isdir(image_folder)

with open(annotation_file_abs,'r') as f:
    annotation_data = json.load(f)
    
abs_filename_to_annotations = defaultdict(list)
category_name_to_id = {}
category_id_to_count = defaultdict(int)

n_annotations = 0

box_widths = []

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
    
    for ann in annotations_this_image:
        category_id = ann['category_id']
        category_id_to_count[category_id] = category_id_to_count[category_id] + 1
        assert 'bbox' in ann
        if isinstance(ann['bbox'],str):
            ann['bbox'] = ast.literal_eval(ann['bbox']) 
        assert len(ann['bbox']) == 4
        abs_filename_to_annotations[fn_abs].append(ann)
        n_annotations += 1       
        
        box_widths.append(ann['bbox'][3])
        
    # ...for each annotation
    
# ...for each image
        
print('Read {} annotations (average width {}) for {} images'.format(
    n_annotations,np.mean(box_widths),len(abs_filename_to_annotations)))

print('Categories:')
for s in category_name_to_id.keys():
    print(s)


#%% Find an image with a bunch of annotations

image_name_to_count = {}
for image_name in abs_filename_to_annotations:
    image_name_to_count[image_name] = len(abs_filename_to_annotations[image_name])
    
# Sort in descending order by value
sorted_annotations = dict(sorted(image_name_to_count.items(), 
                                 key=operator.itemgetter(1),reverse=True))

sorted_annotations = list(sorted_annotations)

image_full_path = sorted_annotations[1]

assert os.path.isfile(image_full_path)

image_annotations = abs_filename_to_annotations[image_full_path]

print('Found {} annotations for image {}'.format(
    len(image_annotations),image_full_path))


#%% Pick and render all annotations for one image file

pil_im = visutils.open_image(image_full_path)
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
                                     thickness=8,expansion=1, colormap=['red'])

shutil.copyfile(image_full_path,output_file_unannotated)    
path_utils.open_file(output_file_annotated)
