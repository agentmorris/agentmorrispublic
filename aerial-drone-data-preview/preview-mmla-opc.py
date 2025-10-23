#
# Code to render sample images and annotations in the mmla-opc dataset:
#
# https://huggingface.co/datasets/imageomics/mmla_opc
#
# Annotations are boxes in YOLO-formatted .txt files.
#

#%% Constants and imports

import os
import shutil
import operator
import numpy as np

from collections import defaultdict
from tqdm import tqdm

from megadetector.visualization import visualization_utils as visutils
from megadetector.utils import path_utils

base_folder = r'i:\data\wildwing\mmla_opc'
output_file_annotated = r'g:\temp\mmla_opc_sample_image_annotated.jpg'
output_file_unannotated = r'g:\temp\mmla_opc_sample_image_unannotated.jpg'

class_list_file = os.path.join(base_folder,'classes.txt')


#%% Read class names

category_id_to_name = {}
with open(class_list_file,'r') as f:
    lines = f.readlines()
    lines = [s.strip() for s in lines]
for i_line,line in enumerate(lines):
    category_id_to_name[i_line] = line

# {0: 'zebra'}


#%% Read and summarize annotations

annotation_files_relative = \
    path_utils.recursive_file_list(base_folder,return_relative_paths=True,convert_slashes=True)
annotation_files_relative = [fn for fn in annotation_files_relative if fn.endswith('.txt')]
annotation_files_relative = [fn for fn in annotation_files_relative if (not fn.endswith('classes.txt'))]

n_annotations = 0
relative_filename_to_annotations = defaultdict(list)

box_widths = []

# annotation_file_relative = annotation_files_relative[0]
for annotation_file_relative in tqdm(annotation_files_relative):

    image_filename_relative = annotation_file_relative.replace('.txt','.jpg')
    image_filename_abs = os.path.join(base_folder,image_filename_relative)
    annotation_filename_abs = os.path.join(base_folder,annotation_file_relative)

    assert os.path.isfile(image_filename_abs)
    assert os.path.isfile(annotation_filename_abs)

    pil_im = visutils.load_image(image_filename_abs)
    image_w = pil_im.size[0]
    image_h = pil_im.size[1]

    with open(annotation_filename_abs,'r') as f:
        annotation_lines = f.readlines()
    annotation_lines = [s.strip() for s in annotation_lines]

    # line = annotation_lines[0]
    for i_line,line in enumerate(annotation_lines):

        if len(line) == 0:
            continue
        n_annotations += 1

        # class x_center y_center width heigh
        tokens = line.split()
        assert len(tokens) == 5
        category_id = int(tokens[0])
        assert category_id in category_id_to_name

        x_center_norm = float(tokens[1])
        y_center_norm = float(tokens[2])
        width_norm = float(tokens[3])
        height_norm = float(tokens[4])

        x_norm = x_center_norm - (width_norm / 2.0)
        y_norm = y_center_norm - (height_norm / 2.0)

        box_widths.append(x_norm * image_w)

        ann = {}
        ann['file'] = image_filename_relative
        ann['class_id'] = category_id
        ann['x'] = x_norm * image_w
        ann['y'] = y_norm * image_h
        ann['width'] = width_norm * image_w
        ann['height'] = height_norm * image_h

        box_widths.append(ann['width'])
        relative_filename_to_annotations[image_filename_relative].append(ann)

    # ...for each annotation

# ...for each annotation file

print('Read {} annotations for {} images, average width {}'.format(
    n_annotations,len(relative_filename_to_annotations),np.mean(box_widths)))


#%% Print class distribution

class_name_to_count = defaultdict(int)

for fn in relative_filename_to_annotations:
    annotations_this_file = relative_filename_to_annotations[fn]
    for ann in annotations_this_file:
        class_id = ann['class_id']
        category_name = category_id_to_name[class_id]
        class_name_to_count[category_name] += 1

for class_name in class_name_to_count:
    print('{}: {}'.format(class_name,class_name_to_count[class_name]))

"""
zebra: 163219
"""


#%% Find an image with a bunch of annotations

image_name_to_count = {}
for image_name in relative_filename_to_annotations:
    image_name_to_count[image_name] = len(relative_filename_to_annotations[image_name])

# Sort in descending order by count
sorted_annotations = dict(sorted(image_name_to_count.items(),
                                 key=operator.itemgetter(1),reverse=True))

sorted_annotations = list(sorted_annotations)

image_relative_path = sorted_annotations[0]

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
    category_id = ann['class_id']
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

category_id_str_to_name = {}
for id in category_id_to_name:
    category_id_str_to_name[str(id)] = category_id_to_name[id]

visutils.draw_bounding_boxes_on_file(image_full_path,
                                     output_file_annotated,
                                     detection_formatted_boxes,
                                     confidence_threshold=0.0,
                                     detector_label_map=category_id_to_name,
                                     thickness=10,expansion=0)

shutil.copyfile(image_full_path,output_file_unannotated)
path_utils.open_file(output_file_annotated)
