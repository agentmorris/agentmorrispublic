#
# Code to render sample images and annotations in the Weinstein et al. global birds dataset:
#
# https://zenodo.org/record/5033174
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

from megadetector.visualization import visualization_utils as visutils
from megadetector.utils import path_utils

base_folder = r'c:\drone-data\09 - weinstein'
output_file_annotated = r'g:\temp\weinstein_birds_sample_image_annotated.jpg'
output_file_unannotated = r'g:\temp\weinstein_birds_sample_image_unannotated.png'


#%% Read and summarize annotations

relative_filename_to_annotations = defaultdict(list)
n_annotations = 0
category_name_to_id = {}

csv_files = glob.glob(base_folder + '/**/*.csv')

box_widths = []

# annotation_csv_file = csv_files[0]
for annotation_csv_file in tqdm(csv_files):
    
    df = pd.read_csv(annotation_csv_file)
    n_annotations += len(df)
    
    # E.g. "everglades"
    dataset_folder = os.path.basename(os.path.dirname(annotation_csv_file))
    
    # i_row = 0; row = df.iloc[i_row]
    for i_row,row in df.iterrows():
    
        image_filename_relative = os.path.join(dataset_folder,row['image_path'])
        label = row['label'].lower()
        
        if label not in category_name_to_id:
            category_name_to_id[label] = len(category_name_to_id)
        
        assert os.path.isfile(os.path.join(base_folder,image_filename_relative))
        
        ann = {}
        ann['file'] = image_filename_relative
        ann['label'] = label
        for s in ['xmin','ymin','xmax','ymax']:
            ann[s] = row[s]

        box_widths.append(ann['xmax'] - ann['xmin'])
        relative_filename_to_annotations[image_filename_relative].append(ann)
        
    # ...for each row in this csv file    

# ...for each csv file        

print('Read {} annotations for {} images, average width {}'.format(
    n_annotations,len(relative_filename_to_annotations),np.mean(box_widths)))

print('Categories:')
for s in category_name_to_id.keys():
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

# ann = image_annotations[0]
for ann in image_annotations:
        
    x0 = ann['xmin']
    x1 = ann['xmax']
    y0 = ann['ymin']
    y1 = ann['ymax']
    
    det = {}
    det['conf'] = None
    det['category'] = category_name_to_id[ann['label']]
    box_w = (x1-x0)
    box_h = (y1-y0)
    assert box_h > 0
    box = [x0/image_w,
           y0/image_h,
           box_w/image_w,
           box_h/image_h]
    
    det['bbox'] = box    
    detection_formatted_boxes.append(det)
    
visutils.draw_bounding_boxes_on_file(image_full_path, output_file_annotated, detection_formatted_boxes,       
                                     confidence_threshold=0.0,detector_label_map=None,
                                     thickness=2,expansion=0)

shutil.copyfile(image_full_path,output_file_unannotated)    
path_utils.open_file(output_file_annotated)
