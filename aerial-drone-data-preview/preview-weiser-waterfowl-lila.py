#
# Code to render sample images and annotations Weiser et al. waterfowl dataset, LILA edition:
#
# https://lila.science/datasets/izembek-lagoon-waterfowl/
#
# Annotations are boxes in a COCO .json file.
#

#%% Constants and imports

import os
import json

import numpy as np

from collections import defaultdict
from tqdm import tqdm

from megadetector.visualization import visualization_utils as vis_utils

output_file_annotated = r'g:\temp\weiser_waterfowl_lila_sample_image_annotated.jpg'
output_file_unannotated = r'g:\temp\weiser_waterfowl_lila_sample_image_unannotated.jpg'

annotation_file = r"J:\lila\izembek-lagoon-birds\izembek-lagoon-birds-metadata.json"
image_base = r"J:\lila\izembek-lagoon-birds"


#%% Read and summarize annotations

with open(annotation_file,'r') as f:
    d = json.load(f)

image_id_to_annotations = defaultdict(list)
category_name_to_id = {}
category_id_to_count = defaultdict(int)
box_widths = []

# Map all annotations in this dataset to their corresponding images

# ann = d['annotations'][0]
for ann in tqdm(d['annotations']):
    image_id_to_annotations[ann['image_id']].append(ann)

for cat in d['categories']:
    if cat['name'] in category_name_to_id:
        assert cat['id'] == category_name_to_id[cat['name']]
    else:
        category_name_to_id[cat['name']] = cat['id']

# im = d['images'][0]
for im in tqdm(d['images']):

    file_name_short = im['file_name']
    fn_abs = os.path.join(image_base,file_name_short)
    assert os.path.isfile(fn_abs)

    annotations_this_image = image_id_to_annotations[im['id']]

    for ann in annotations_this_image:
        category_id = ann['category_id']
        category_id_to_count[category_id] = category_id_to_count[category_id] + 1
        if 'bbox' in ann:
            assert len(ann['bbox']) == 4
            box_widths.append(ann['bbox'][3])

    # ...for each annotation

# ...for each image

print('\nRead {} annotations (average width {}) for {} images'.format(
    len(d['annotations']),np.mean(box_widths),len(d['images'])))

print('Categories:')
for s in category_name_to_id.keys():
    print(s)


#%% Find an image with a bunch of *different* annotations

threshold_count = 15
threshold_species = 3
selected_image_id = None

for image_id in image_id_to_annotations:

    category_to_count_this_image = defaultdict(int)
    annotations_this_image = image_id_to_annotations[image_id]
    for ann in annotations_this_image:
        if 'bbox' in ann:
            category_to_count_this_image[ann['category_id']] += 1

    species_over_threshold_this_image = 0
    for category_id in category_name_to_id.values():
        if category_id in category_to_count_this_image and \
            category_to_count_this_image[category_id] >= threshold_count:
                species_over_threshold_this_image += 1

    if species_over_threshold_this_image >= threshold_species:
        selected_image_id = image_id
        break

assert selected_image_id is not None

selected_image_filename_relative = None

for im in d['images']:
    if im['id'] == selected_image_id:
        selected_image_filename_relative = im['file_name']
        break

selected_image_filename_abs = os.path.join(image_base,selected_image_filename_relative)
assert os.path.isfile(selected_image_filename_abs)


#%% Pick and render all annotations for one image file

pil_im = vis_utils.open_image(selected_image_filename_abs)
image_w = pil_im.size[0]
image_h = pil_im.size[1]

detection_formatted_boxes = []

image_annotations = image_id_to_annotations[selected_image_id]

# ann = image_annotations[0]
for ann in image_annotations:

    if 'bbox' not in ann:
        continue

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

vis_utils.draw_bounding_boxes_on_file(selected_image_filename_abs, output_file_annotated, detection_formatted_boxes,
                                     confidence_threshold=0.0,detector_label_map=category_id_to_name,
                                     thickness=2,expansion=3) # colormap=['red'])

from megadetector.utils import path_utils
path_utils.open_file(output_file_annotated)

import shutil
shutil.copyfile(selected_image_filename_abs,output_file_unannotated)
