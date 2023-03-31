#
# Code to render sample images and count annotations in the Conservation
# Drones dataset:
#
# https://lila.science/datasets/conservationdrones
#

#%% Constants and imports

import pandas as pd
import os

base_folder = r'G:\temp\drone-datasets\conservation-drones'

annotation_folders = {'train':'TrainReal/annotations',
                      'test':'TestReal/annotations'}

image_folders = {'train':'TrainReal/images',
                 'test':'TestReal/images'}


# 
# From the docs, the annotation columns are:
# 
# [frame_number], [object_id], [x], [y], [w], [h], [class], [species], [occlusion], [noise]
#
# From the MOT page, x and y are top-left
#


#%% Read annotation files

# Build a list of dicts with keys:
#
# dataset_name: 'train' or 'test'
# video_name: e.g. 0000000011_0000000000)
# annotations: a list of dicts with fields 'frame', 'object_id', 'top', 'left', 'w', 'h', 'class', 'species'
#

all_annotations = []

n_annotations = 0

# dataset_name = list(annotation_folders.keys())[0]
for dataset_name in annotation_folders:
    
    annotation_folder_relative = annotation_folders[dataset_name]
    annotation_folder_full_path = os.path.join(base_folder,annotation_folder_relative)
    annotation_files = os.listdir(annotation_folder_full_path)
    annotation_files = [fn for fn in annotation_files if fn.endswith('.csv')]
    annotation_files = [os.path.join(annotation_folder_full_path,fn) for fn in annotation_files]
    
    # annotation_fn = annotation_files[0]
    for annotation_fn in annotation_files:
        
        video_name = os.path.basename(annotation_fn).split('.')[0]
        df = pd.read_csv(annotation_fn,header=None)
        
        annotations_this_file = []
        
        # i_row = 0; row = df.iloc[i_row]
        for i_row,row in df.iterrows():
            
            annotation_record = {}
            annotation_record['frame_num'] = row[0]
            annotation_record['object_id'] = row[1]
            annotation_record['left'] = row[2]
            annotation_record['top'] = row[3]
            annotation_record['w'] = row[4]
            annotation_record['h'] = row[5]
            annotation_record['class'] = row[6]
            annotation_record['species'] = row[7]
            assert annotation_record['species'] in [-1,0,1,2,3,4,5,6,7,8]
            
            annotations_this_file.append(annotation_record)
        
        n_annotations += len(annotations_this_file)
        all_annotations.append(
            {'dataset_name':dataset_name,
             'video_name':video_name,
             'annotations':annotations_this_file}
            )
            
print('Read {} annotations for {} videos'.format(n_annotations,len(all_annotations)))
assert len(all_annotations) == 48


#%% Find average box width, and also the average box width for an elephant

import numpy as np

elephant_category = 1
animal_class = 0

box_widths = []
elephant_box_widths = []

# video_info = all_annotations[0]
for i_video,video_info in enumerate(all_annotations):
    for i_ann,ann in enumerate(video_info['annotations']):
        w = ann['w']
        box_widths.append(w)
        if ann['species'] == elephant_category:
            assert ann['class'] == animal_class
            elephant_box_widths.append(w)
           
print('Average box width: {}'.format(np.mean(box_widths)))
print('Average elephant box width: {}'.format(np.mean(elephant_box_widths)))


#%% Pick one frame of one video and render the corresponding annotations

i_video = 2
i_frame = 4

video_info = all_annotations[i_video]
frame_annotations = [ann for ann in video_info['annotations'] if ann['frame_num'] == i_frame]

image_folder_base = image_folders[video_info['dataset_name']]
video_image_folder = os.path.join(base_folder,image_folder_base,video_info['video_name'])
assert os.path.isdir(video_image_folder)

frame_files = os.listdir(video_image_folder)

frame_image_path = None
# E.g. 0000000067_0000000025_0000000932.jpg
# frame_fn = frame_files[0]
for frame_fn in frame_files:
    frame_number = int((frame_fn.split('.')[0]).split('_')[2])
    if frame_number == i_frame:
        frame_image_path = os.path.join(video_image_folder,frame_fn)
        
assert frame_image_path is not None

from visualization import visualization_utils as visutils

category_id_to_name = {0:'animal',1:'person'}

detection_formatted_boxes = []

pil_im = visutils.open_image(frame_image_path)
image_w = pil_im.size[0]
image_h = pil_im.size[1]

# frame_ann = frame_annotations[0]
for frame_ann in frame_annotations:
    det = {}
    det['conf'] = 1.0
    det['category'] = frame_ann['class']    
    box = [frame_ann['left']/image_w,
           frame_ann['top']/image_h,
           frame_ann['w']/image_w,
           frame_ann['h']/image_h]
    
    det['bbox'] = box    
    detection_formatted_boxes.append(det)
    
output_file = r'g:\temp\conservation_drones_sample_image.jpg'
visutils.draw_bounding_boxes_on_file(frame_image_path, output_file, detection_formatted_boxes,       
                                     confidence_threshold=0.0)


    

            
            
        
