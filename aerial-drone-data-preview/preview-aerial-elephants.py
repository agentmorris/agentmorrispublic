#
# Code to render sample images and points in the Aerial Elephants Dataset:
#
# https://zenodo.org/record/3234780
#
# Annotations are points in a .csv file.
#

#%% Constants and imports

import pandas as pd
import operator
import os
import shutil

from collections import defaultdict
from PIL import ImageDraw

import path_utils
from visualization import visualization_utils as visutils

base_folder = r'c:\drone-data\08 - naude\AED'

annotation_files = ['test_elephants.csv','training_elephants.csv']

# 
# The annotation columns are:
# 
# [filename], [x], [y]
#
#

output_file_annotated = r'g:\temp\aerial_elephants_sample_image_annotated.jpg'
output_file_unannotated = r'g:\temp\aerial_elephants_sample_image_unannotated.jpg'


#%% List all images

images_relative = path_utils.find_images(base_folder,recursive=True)
images_relative = [os.path.relpath(fn,base_folder) for fn in images_relative]

# Images will be identified by basename in the .csv file, map to relative path
image_name_to_relative_path = {os.path.splitext(os.path.basename(fn))[0]:fn for fn in images_relative}


#%% Read the annotation files

annotation_dfs = []
for fn in annotation_files:    
    df = pd.read_csv(os.path.join(base_folder,fn),header=None)
    annotation_dfs.append(df)

df = pd.concat(annotation_dfs)

df = df.rename(columns={df.columns[0]:'image_id', df.columns[1]:'x', 
                df.columns[2]:'y'})

print('Read {} annotations'.format(len(df)))


#%% Collect annotations for each image

image_to_annotations = defaultdict(list)

# i_row = 0; row = df.iloc[i_row]
for i_row,row in df.iterrows():
    
    image_id = row['image_id']
    assert image_id in image_name_to_relative_path
    
    ann = {}
    ann['x'] = row['x']
    ann['y'] = row['y']
        
    image_to_annotations[image_id].append(ann)
    
    
#%% Render annotations for an image that has a decent number of annotations

image_name_to_count = {}
for image_name in image_to_annotations:
    image_name_to_count[image_name] = len(image_to_annotations[image_name])
    
# Sort in descending order by value
sorted_annotations = dict(sorted(image_name_to_count.items(), 
                                 key=operator.itemgetter(1),reverse=True))

sorted_annotations = list(sorted_annotations)
image_name = sorted_annotations[0]

image_relative_path = image_name_to_relative_path[image_name]
image_full_path = os.path.join(base_folder,image_relative_path)
assert os.path.isfile(image_full_path)

image_annotations = image_to_annotations[image_name]

print('Found {} annotations for image {}'.format(
    len(image_annotations),image_full_path))


#%% Render points

pil_im = visutils.open_image(image_full_path)
print('Opened image with size: {}'.format(str(pil_im.size)))

draw = ImageDraw.Draw(pil_im)

ann_radius = 30
for ann in image_annotations:
    
    x = ann['x']
    y = ann['y']
    
    x0 = x - ann_radius
    y0 = y - ann_radius
    x1 = x + ann_radius
    y1 = y + ann_radius

    draw.ellipse((x0,y0,x1,y1),fill=None,outline=(255,0,0),width=3)
    # draw.ellipse((x0,y0,x1,y1),fill=(255,0,0),outline=None)
        
pil_im.save(output_file_annotated,quality=60)
shutil.copyfile(image_full_path,output_file_unannotated)
path_utils.open_file(output_file_annotated)