#
# Code to render sample images and annotations in the Hayes et al. seabirds dataset:
#
# https://research.repository.duke.edu/concern/datasets/kp78gh20s?locale=en
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

base_folder = r'c:\drone-data\11 - hayes'
output_file_annotated = r'g:\temp\hayes_seabirds_sample_image_annotated.jpg'
output_file_unannotated = r'g:\temp\hayes_seabirds_sample_image_unannotated.png'

dataset_name_to_image_folder = {
    'Albatross':'Training, Validation, and Testing Labels and Tiles/Albatross_LabeledTiles',
    'Penguin':'Training, Validation, and Testing Labels and Tiles/Penguin_LabeledTiles'
}

# Annotation columns are:
#
# filename, x1, y1, x2, y2, label    


#%% Read and summarize annotations

csv_files = glob.glob(base_folder + '/**/*.csv',recursive=True)
csv_files = [fn for fn in csv_files if 'annotations' in fn]

annotations = []

category_name_to_id = {}

# annotation_csv_file = csv_files[0]
for annotation_csv_file in tqdm(csv_files):
    
    dataset_name = None
    for s in dataset_name_to_image_folder.keys():
        if s in annotation_csv_file:
            dataset_name = s
            break
    assert dataset_name is not None
    
    df = pd.read_csv(annotation_csv_file,header=None)
    
    # i_row = 0; row = df.iloc[i_row]
    for i_row,row in df.iterrows():
    
        image_name = row[0]
        x1 = row[1]
        y1 = row[2]
        x2 = row[3]
        y2 = row[4]
        label = row[5]
        
        assert x2 > x1
        assert y2 > y1
        
        if label not in category_name_to_id:
            category_name_to_id[label] = len(category_name_to_id)
        
        ann = {}
        ann['image_name'] = image_name
        ann['dataset_name'] = dataset_name
        ann['label'] = dataset_name.lower()
        ann['x'] = x1
        ann['y'] = y1
        ann['w'] = x2 - x1
        ann['h'] = y2 - y1

        annotations.append(ann)
        
    # ...for each row in this csv file    

# ...for each csv file        

object_widths = [ann['w'] for ann in annotations]

image_ids = set( [ (ann['dataset_name']+'-'+ann['image_name']) for ann in annotations ] )
print('Read {} annotations for {} images, average width {}'.format(
    len(annotations),len(image_ids),np.mean(object_widths)))

print('Categories:')
for s in category_name_to_id.keys():
    print(s)


#%% Find an image with a bunch of annotations

image_name_to_annotations = defaultdict(list)
for ann in annotations:
    image_name_to_annotations[ann['image_name']].append(ann)

image_name_to_count = {}
for image_name in image_name_to_annotations:
    image_name_to_count[image_name] = len(image_name_to_annotations[image_name])
    
# Sort in descending order by value
sorted_annotations = dict(sorted(image_name_to_count.items(), 
                                 key=operator.itemgetter(1),reverse=True))

sorted_annotations = list(sorted_annotations)

image_name = sorted_annotations[0]
image_annotations = image_name_to_annotations[image_name]
dataset_name = image_annotations[0]['dataset_name']

image_relative_path = os.path.join(dataset_name_to_image_folder[dataset_name],
                                   image_name)

image_full_path = os.path.join(base_folder,image_relative_path)
assert os.path.isfile(image_full_path)

print('Found {} annotations for image {}'.format(
    len(image_annotations),image_full_path))


#%% Pick and render all annotations for one image file

pil_im = visutils.open_image(image_full_path)
image_w = pil_im.size[0]
image_h = pil_im.size[1]

detection_formatted_boxes = []

# ann = image_annotations[0]
for ann in image_annotations:
        
    x = ann['x']
    y = ann['y']
    box_w = ann['w']
    box_h = ann['h']
    
    det = {}
    det['conf'] = None
    det['category'] = category_name_to_id[ann['label']]
    box = [x/image_w,
           y/image_h,
           box_w/image_w,
           box_h/image_h]
    
    det['bbox'] = box    
    detection_formatted_boxes.append(det)
    
visutils.draw_bounding_boxes_on_file(image_full_path, output_file_annotated, detection_formatted_boxes,       
                                     confidence_threshold=0.0,detector_label_map=None,
                                     thickness=2,expansion=0)

shutil.copyfile(image_full_path,output_file_unannotated)    
path_utils.open_file(output_file_annotated)
