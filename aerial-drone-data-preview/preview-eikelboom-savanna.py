#
# Code to render sample images and count annotations in the Eikelboom et al.
# aerial image datasets.
#
# https://data.4tu.nl/articles/dataset/Improving_the_precision_and_accuracy_of_animal_population_estimates_with_aerial_image_object_detection/12713903/1
#
# Annotations are boxes in a .csv file.
#

#%% Constants and imports

import pandas as pd
import operator
import os
import shutil

from megadetector.utils import path_utils
from megadetector.visualization import visualization_utils as visutils

base_folder = r'g:\temp\drone-data\02 - eikelboom'

image_folders = ['test','train','val']

expected_species = ['zebra','elephant','giraffe']

output_file_annotated = r'g:\temp\eikelboom_savanna_sample_image_annotated.jpg'
output_file_unannotated = r'g:\temp\eikelboom_savanna_sample_image_unannotated.jpg'
crop_path = r'g:\temp\eikelboom_savanna_sample_image_crop.jpg'

assert os.path.isdir(base_folder)

#
# The annotation columns are:
#
# [filename], [x1], [x2], [y1], [y2], [species]
#
#


#%% List all images

# ...and map them to the folder that contains them
image_name_to_folder = {}

# image_folder = image_folders[0]
for image_folder in image_folders:
    image_files = os.listdir(os.path.join(base_folder,image_folder))
    for fn in image_files:
        assert fn.lower().endswith('.jpg') or fn.lower.endswith('.png')
        assert fn not in image_name_to_folder
        image_name_to_folder[fn] = image_folder

print('Enumerated {} images'.format(len(image_name_to_folder)))


#%% Read the annotation file

df = pd.read_csv(os.path.join(base_folder,'annotations_images.csv'))

print('Read {} annotation rows'.format(len(df)))


#%% Count annotations for each image

from collections import defaultdict
image_name_to_count = defaultdict(int)

for i_row,row in df.iterrows():
    image_name_to_count[row['FILE']] += 1


#%% Render annotations for an image that has a decent number of annotations

# Sort in descending order by value
sorted_annotations = dict(sorted(image_name_to_count.items(),
                                 key=operator.itemgetter(1),reverse=True))

sorted_annotations = list(sorted_annotations)
image_name = sorted_annotations[1]

image_folder = image_name_to_folder[image_name]
image_full_path = os.path.join(base_folder,image_folder,image_name)
assert os.path.isfile(image_full_path)

image_annotations = []

for i_row,row in df.iterrows():
    if row['FILE'] == image_name:
        ann = {}
        ann['species'] = row['SPECIES'].lower()
        for s in ['x1','y1','x2','y2']:
            ann[s] = row[s]
        image_annotations.append(ann)

print('Found {} annotations for image {}'.format(
    len(image_annotations),image_full_path))


#%% Render boxes

from megadetector.visualization.visualization_utils import crop_image
from megadetector.visualization.visualization_utils import exif_preserving_save

detection_formatted_boxes = []

# Normalized x/y/w/h
normalized_area_to_process = [0.525, 0.556, 0.3, 0.2]

input_image_path = image_full_path
pil_im = visutils.open_image(image_full_path)

image_w = pil_im.size[0]
image_h = pil_im.size[1]

if normalized_area_to_process is not None:
    detections = []
    detections.append({'conf':1.0, 'bbox':normalized_area_to_process, 'category':'0'})
    pil_im = crop_image(detections, pil_im, confidence_threshold=0.15, expansion=0)[0]
    input_image_path = crop_path
    exif_preserving_save(pil_im,crop_path,quality='keep')

category_id_to_name = {}
category_name_to_id = {}

for i_s,s in enumerate(expected_species):
    cat_id = str(i_s)
    category_id_to_name[cat_id] = s
    category_name_to_id[s] = cat_id

# ann = image_annotations[0]
for ann in image_annotations:

    det = {}
    det['conf'] = None
    det['category'] = category_name_to_id[ann['species']]

    # Convert to normalized x/y/w/h
    box = [ann['x1']/image_w,
           ann['y1']/image_h,
           (ann['x2']-ann['x1'])/image_w,
           (ann['y2']-ann['y1'])/image_h]

    # box = normalized_area_to_process

    # "box" represents x/y/w/h in normalized coordinates relative to the whole image.
    # If normalized_area_to_process is not None, normalize [box] accordingly.
    if normalized_area_to_process is not None:

        crop_x, crop_y, crop_w, crop_h = normalized_area_to_process
        box_x, box_y, box_w, box_h = box

        # Transform from original image coordinates to crop-relative coordinates
        box = [(box_x - crop_x) / crop_w,
            (box_y - crop_y) / crop_h,
            box_w / crop_w,
            box_h / crop_h]

    # ...if we need to normalize to a crop

    # Discard boxes outside the region

    if box[1] < 0 or box[0] > 1:
        continue

    det['bbox'] = box
    detection_formatted_boxes.append(det)

visutils.draw_bounding_boxes_on_file(input_image_path,
                                     output_file_annotated,
                                     detection_formatted_boxes,
                                     confidence_threshold=0.0,
                                     detector_label_map=category_id_to_name,
                                     thickness=10,
                                     label_font_size=70,
                                     target_size=(10000,-1))

path_utils.open_file(output_file_annotated)

shutil.copyfile(image_full_path,output_file_unannotated)
