#
# Code to render sample images and count annotations in the Qian et al.
# penguin dataset:
#
# https://zenodo.org/record/7702635#.ZChnoHZBxD8
#
# Annotations are points in LabelBox-formatted .json files.
#

#%% Constants and imports

import os
import glob
import json
import operator
import shutil

from md_visualization import visualization_utils as visutils
from md_utils import path_utils

base_folder = r'c:\drone-data\06 - qian'
assert os.path.isdir(base_folder)

image_folders = ['Jack','Luke','Maisie','Thomas']
image_folders = [os.path.join(base_folder,fn) for fn in image_folders]
for folder_name in image_folders:
    assert os.path.isdir(folder_name)
    
output_file_annotated = r'g:\temp\qian_penguins_sample_image_annotated.jpg'
output_file_unannotated = r'g:\temp\qian_penguins_sample_image_unannotated.png'


#%% Map image filenames to folders

image_to_folder = {}

for image_folder in image_folders:
    image_files = os.listdir(image_folder)
    image_files = [fn for fn in image_files if \
                   (fn.lower().endswith('.png') or fn.lower().endswith('.jpg'))]
    for image_file in image_files:
        assert image_file not in image_to_folder
        image_to_folder[image_file] = image_folder
        

#%% Read and summarize annotations

json_files = glob.glob(base_folder + '/*.json')

# Map filenames to a list of annotations for each file
filename_to_image = {}

# json_file = json_files[0]
for json_file in json_files:
    with open(json_file,'r') as f:
        input_annotations = json.load(f)
        
    # input_ann = input_annotations[0]
    for input_ann in input_annotations:
        
        image_filename = input_ann['External ID']
        if image_filename not in filename_to_image:
            output_im = {}
            output_im['filename'] = image_filename
            output_im['detections'] = []
            filename_to_image[image_filename] = output_im
        else:
            output_im = filename_to_image[image_filename]
        
        # Each annotation has a "Label" field, a dict with fields "objects" and "classifications"
        label = input_ann['Label']
        if len(label) > 0:            
            assert  len(label['classifications']) == 0
            objects = label['objects']
            for obj in objects:
                det = {}
                det['species'] = obj['value']
                det['box'] = obj['bbox']
                output_im['detections'].append(det)
            
    # ...for each annotation record
    
# ...for each .json file    

n_annotations = sum([len(ann['detections']) for ann in filename_to_image.values()])

print('Read {} annotations for {} images'.format(n_annotations,len(filename_to_image)))


#%% Render annotations for an image that has a decent number of annotations

choose_image_by_number_of_annotations = True

if choose_image_by_number_of_annotations:
        
    image_name_to_count = {}
    for fn in filename_to_image:
        image_name_to_count[fn] = len(filename_to_image[fn]['detections'])
    
    # Sort in descending order by value
    images_sorted_by_count = dict(sorted(image_name_to_count.items(), 
                                     key=operator.itemgetter(1),reverse=True))
    
    images_sorted_by_count = list(images_sorted_by_count)
    image_name = images_sorted_by_count[299]
    
else:
    
    image_name = 'Bas2019_04d_010~247_rg.chop.x4032.y3136.png'
    
folder_name = image_to_folder[image_name]
image_full_path = os.path.join(folder_name,image_name)
assert os.path.isfile(image_full_path)

image_annotations = filename_to_image[image_name]

print('Found {} annotations for image {}'.format(
    len(image_annotations['detections']),image_full_path))


#%% Render boxes

pil_im = visutils.open_image(image_full_path)
image_w = pil_im.size[0]
image_h = pil_im.size[1]

category_name_to_id = {}

# ann = image_annotations['detections'][0]
for ann in image_annotations['detections']:
    if ann['species'] not in category_name_to_id:
        category_name_to_id[ann['species']] = str(len(category_name_to_id))

category_id_to_name = {}

for s in category_name_to_id:
    category_id_to_name[category_name_to_id[s]] = s

detection_formatted_boxes = []

# ann = image_annotations[0]
for ann in image_annotations['detections']:
    
    det = {}
    det['conf'] = None
    det['category'] = category_name_to_id[ann['species']]
    
    # Convert to relative x/y/w/h
    box = [ann['box']['left']/image_w,
           ann['box']['top']/image_h,
           (ann['box']['width'])/image_w,
           (ann['box']['height'])/image_h]           
    
    det['bbox'] = box    
    detection_formatted_boxes.append(det)
    
visutils.draw_bounding_boxes_on_file(image_full_path, output_file_annotated, detection_formatted_boxes,
                                     confidence_threshold=0.0,detector_label_map=None,
                                     thickness=1,expansion=1,colormap=['red'])


shutil.copyfile(image_full_path,output_file_unannotated)
path_utils.open_file(output_file)
