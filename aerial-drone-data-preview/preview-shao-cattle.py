#
# Code to render sample images and boxes from the Shao et al. dataset:
#
# http://bird.nae-lab.org/cattle/
#
# Annotations are boxes in .csv files
#

#%% Constants and imports

import os
import operator
import shutil
import numpy as np
import glob

from md_visualization import visualization_utils as visutils
from md_utils import path_utils

base_folder = r"c:\drone-data\12 - shao"
assert os.path.isdir(base_folder)

output_file_annotated = r'g:\temp\shao_cattle_sample_image_annotated.jpg'
output_file_unannotated = r'g:\temp\shao_cattle_sample_image_unannotated.jpg'

dataset_to_image_folder = {'dataset1':'Dataset1','dataset2':'Dataset2'}


#%% Read and summarize annotations

annotation_files = glob.glob(base_folder + '/*.txt')

relative_path_to_annotations = {}

# Format is tab-delimited text, with a variable number of tokens per line depending
# on the number of annotations
box_columns = ['x','y','w','h','quality','id','id_confidence']
n_columns_per_box = len(box_columns)

# annotation_file = annotation_files[0]
for annotation_file in annotation_files:
    
    image_folder = None
    for dataset_name in dataset_to_image_folder.keys():
        if dataset_name in annotation_file:
            image_folder = dataset_to_image_folder[dataset_name]
            break
        
    with open(annotation_file,'r') as f:
        lines = f.readlines()
    lines = [s.strip() for s in lines]
    
    # Skip the header
    assert lines[0].startswith('image')
    lines = lines[1:]
    
    
    # s = lines[0]
    for i_line,s in enumerate(lines):
        tokens = s.split('\t')        
        
        image_relative_path = os.path.join(image_folder,tokens[0])
        assert os.path.isfile(os.path.join(base_folder,image_relative_path))
        
        n_annotation_columns = len(tokens) - 2
        n_boxes = int(tokens[1])
        
        # This came up for seven images
        if n_annotation_columns != (n_columns_per_box * n_boxes):
            print('Invalid annotations for image {}, bypassing'.format(image_relative_path))
            continue
        
        relative_path_to_annotations[image_relative_path] = []
        
        
        boxes = []
        
        # i_box = 0
        for i_box in range(0,n_boxes):
            
            box_start_col = 2+(i_box*n_columns_per_box)
            
            ann = {}
            for i_col,col_name in enumerate(box_columns):
                ann[col_name] = int(tokens[box_start_col+i_col])
            boxes.append(ann)
            assert ann['quality'] >= 0 and ann['quality'] <= 3
            
        relative_path_to_annotations[image_relative_path] = boxes

    # ...for each line
    
# ...for each annotation file
    
n_annotations = 0
box_widths = []    
for fn in relative_path_to_annotations.keys():
    n_annotations += len(relative_path_to_annotations[fn])
    for ann in relative_path_to_annotations[fn]:
        box_widths.append(ann['w'])
    
print('Found {} valid annotations on {} images (mean width {})'.format(
    n_annotations,len(relative_path_to_annotations),np.mean(box_widths)))


#%% Find an image with a bunch of annotations

image_name_to_count = {}
for image_name in relative_path_to_annotations:
    image_name_to_count[image_name] = len(relative_path_to_annotations[image_name])
    
# Sort in descending order by value
sorted_annotations = dict(sorted(image_name_to_count.items(), 
                                 key=operator.itemgetter(1),reverse=True))

sorted_annotations = list(sorted_annotations)

image_relative_path = sorted_annotations[1]

image_full_path = os.path.join(base_folder,image_relative_path)
assert os.path.isfile(image_full_path)

image_annotations = relative_path_to_annotations[image_relative_path]

print('Found {} annotations for image {}'.format(
    len(image_annotations),image_full_path))


#%% Pick and render all annotations for one image file

pil_im = visutils.open_image(image_full_path)
image_w = pil_im.size[0]
image_h = pil_im.size[1]

detection_formatted_boxes = []

# ann = image_annotations[0]
for ann in image_annotations:
        
    det = {}
    det['conf'] = None
    det['category'] = 0

    box = [ann['x']/image_w,
           ann['y']/image_h,
           ann['w']/image_w,
           ann['h']/image_h]
    
    det['bbox'] = box    
    detection_formatted_boxes.append(det)
    
visutils.draw_bounding_boxes_on_file(image_full_path, output_file_annotated, detection_formatted_boxes,       
                                     confidence_threshold=0.0,detector_label_map=None,
                                     thickness=4,expansion=0,colormap=['Red'])

shutil.copyfile(image_full_path,output_file_unannotated)    
path_utils.open_file(output_file_annotated)
