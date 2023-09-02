#
# Code to render sample images and annotations in the Koger et al. aerial zebra dataset:
#
# https://edmond.mpdl.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.EMRZGH
#
# Annotations are boxes in a COCO .json file (technically two COCO .json files).
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

base_folder = r'G:\temp\data-repo\data-repo'

annotation_file_ungulates = os.path.join(base_folder,r'kenyan-ungulates\ungulate-annotations\annotations-clean-name-pruned\annotations-clean-name-pruned.json')
annotation_file_geladas = os.path.join(base_folder,r'geladas\gelada-annotations\train_males.json')

annotation_file_to_image_folder = {
 annotation_file_ungulates:os.path.join(base_folder,r'kenyan-ungulates\ungulate-annotations'),
 annotation_file_geladas:os.path.join(base_folder,r'geladas\gelada-annotations\annotated_images')
}

output_file_annotated = r'g:\temp\koger_drones_sample_image_annotated.jpg'
output_file_unannotated = r'g:\temp\koger_drones_sample_image_unannotated.png'


#%% Read and summarize annotations

annotation_files = [annotation_file_ungulates,annotation_file_geladas]

abs_filename_to_annotations = defaultdict(list)
category_name_to_id = {}
category_id_to_count = defaultdict(int)

n_annotations = 0

box_widths = []
zebra_widths = []

# annotation_file = annotation_files[0]
for annotation_file in annotation_files:
    
    with open(annotation_file,'r') as f:
        d = json.load(f)    
    
    image_id_to_annotations = defaultdict(list)
    
    # Map all annotations in this dataset to their corresponding images
    
    # ann = d['annotations'][0]
    for ann in d['annotations']:
        image_id_to_annotations[ann['image_id']].append(ann)

    for cat in d['categories']:
        if cat['name'] in category_name_to_id:
            assert cat['id'] == category_name_to_id[cat['name']]
        else:
            category_name_to_id[cat['name']] = cat['id']
    
    # im = d['images'][0]
    for im in tqdm(d['images']):
        
        file_name_short = im['file_name']
        folder_name = annotation_file_to_image_folder[annotation_file]
        fn_abs = os.path.join(folder_name,file_name_short)
        assert os.path.isfile(fn_abs)
        
        annotations_this_image = image_id_to_annotations[im['id']]
        
        for ann in annotations_this_image:
            category_id = ann['category_id']
            category_id_to_count[category_id] = category_id_to_count[category_id] + 1
            assert 'bbox' in ann
            assert len(ann['bbox']) == 4
            abs_filename_to_annotations[fn_abs].append(ann)
            n_annotations += 1       
            
            box_widths.append(ann['bbox'][3])
            if category_id == 1:
                zebra_widths.append(ann['bbox'][3])
            
        # ...for each annotation
        
    # ...for each image
        
# ...for each annotation file

print('Read {} annotations (average width {} for everything, {} for zebras) for {} images'.format(
    n_annotations,np.mean(box_widths),np.mean(zebra_widths),len(abs_filename_to_annotations)))

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

image_full_path = sorted_annotations[0]

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
