#
# Code to render sample images and annotations in the Weiser et al. waterfowl dataset:
#
# https://alaska.usgs.gov/products/data.php?dataid=484
#
# Annotations are points in .csv files.
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
from PIL import ImageDraw

from megadetector.visualization import visualization_utils as visutils
from megadetector.utils import path_utils

base_folder = r'C:\drone-data\10 - izembek'

# For this preview, we just downloaded one of the image folders (but all of the annotations)
image_folder = 'izembekFallSurvey_20191009_JPGs_CAM3_CAM30001-CAM34644'
image_token = 'izembekFallSurvey_20191009_JPGs_annotations'

output_file_annotated = r'g:\temp\weiser_waterfowl_sample_image_annotated.jpg'
output_file_unannotated = r'g:\temp\weiser_waterfowl_sample_image_unannotated.png'


#%% Read and summarize annotations

relative_filename_to_annotations = defaultdict(list)
n_annotations = 0
category_name_to_id = {}

csv_files = glob.glob(base_folder + '/**/*.csv',recursive=True)

def isnan(v):
    if not isinstance(v,float):
        return False
    return np.isnan(v)

# annotation_csv_file = csv_files[0]
for annotation_csv_file in tqdm(csv_files):
    
    # Only look at manually verified annotations
    if 'Manually' not in annotation_csv_file:
        continue
    
    if os.stat(annotation_csv_file).st_size < 50:
        continue
        
    df = pd.read_csv(annotation_csv_file)
    
    # i_row = 0; row = df.iloc[i_row]
    for i_row,row in df.iterrows():
    
        if 'SpeciesCategory' in row:
            label = row['SpeciesCategory']
        else:
            label = row['Category']
            
        if isnan(label) or len(label) == 0:
            continue
        
        # assert row['ConfidenceLevel'] >= 0
        label = label.lower()
        x = row['X']
        y = row['Y']
                        
        if label not in category_name_to_id:
            category_name_to_id[label] = len(category_name_to_id)
        
        ann = {}
        ann['label'] = label
        ann['x'] = x
        ann['y'] = y
        
        image_name = os.path.relpath(annotation_csv_file,base_folder)
        relative_filename_to_annotations[image_name].append(ann)
        n_annotations += 1
        
    # ...for each row in this csv file    

# ...for each csv file        

print('Read {} annotations for {} images'.format(
    n_annotations,len(relative_filename_to_annotations)))

print('Categories:')
for s in category_name_to_id.keys():
    print(s)


#%% Find an image with a bunch of annotations

csv_file_to_count = {}
for csv_file_relative in relative_filename_to_annotations:
    
    # We only downloaded a subset of images
    if image_token not in csv_file_relative:
        continue
    
    csv_file_to_count[csv_file_relative] = len(relative_filename_to_annotations[csv_file_relative])
    
# Sort in descending order by value
sorted_annotations = dict(sorted(csv_file_to_count.items(), 
                                 key=operator.itemgetter(1),reverse=True))

sorted_annotations = list(sorted_annotations)

available_images = os.listdir(os.path.join(base_folder,image_folder))
available_image_names = set([os.path.splitext(s)[0] for s in available_images])

selected_image_name = None
selected_csv_relative_path = None
# E.g. "izembekFallSurvey_20191009_JPGs_annotations\\Manually_corrected_annotations\\Cam3\\CAM31151_i0107.csv"
for i_csv_file,csv_relative_path in enumerate(sorted_annotations):

    # E.g. CAM31151
    image_name = os.path.basename(csv_relative_path).split('.')[0].split('_')[0]

    if image_name in available_image_names:
        selected_image_name = image_name
        selected_csv_relative_path = csv_relative_path
        break
    
image_full_path = os.path.join(base_folder,image_folder,selected_image_name + '.JPG')
assert os.path.isfile(image_full_path)

image_annotations = relative_filename_to_annotations[selected_csv_relative_path]

print('Found {} annotations for image {}'.format(
    len(image_annotations),image_full_path))


#%% Pick and render all annotations for one image file

pil_im = visutils.open_image(image_full_path)
print('Opened image with size: {}'.format(str(pil_im.size)))

draw = ImageDraw.Draw(pil_im)

ann_radius = 70

# ann = image_annotations[0]
for ann in image_annotations:
    
    x = ann['x']
    y = ann['y']
    label = ann['label']
    
    x0 = x - ann_radius
    y0 = y - ann_radius
    x1 = x + ann_radius
    y1 = y + ann_radius

    draw.ellipse((x0,y0,x1,y1),fill=None,outline=(255,0,0),width=3)
    # draw.ellipse((x0,y0,x1,y1),fill=(255,0,0),outline=None)
        
pil_im.save(output_file_annotated,quality=60)
shutil.copyfile(image_full_path,output_file_unannotated)
path_utils.open_file(output_file_annotated)
