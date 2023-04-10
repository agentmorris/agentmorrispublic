#
# Scratch notebook for training and validating a detector based on:
#
# https://lila.science/datasets/noaa-puget-sound-nearshore-fish
#

#%% Constants and imports

import os
import random

from data_management.coco_to_yolo import coco_to_yolo

base_folder = os.path.expanduser('~/data/noaa-fish')

input_image_folder = os.path.join(base_folder,'JPEGImages')
input_file = os.path.join(base_folder,'noaa_estuary_fish.json')

assert os.path.isdir(input_image_folder)
assert os.path.isfile(input_file)

yolo_all_folder = os.path.join(base_folder,'AllImagesWithAnnotations')
yolo_train_folder = os.path.join(base_folder,'train')
yolo_val_folder = os.path.join(base_folder,'val')

yolo_dataset_file = os.path.join(base_folder,'dataset.yaml')
class_file_name = 'classes.txt'

image_id_mapping_file = os.path.join(yolo_all_folder,'image_id_to_output_image_name.json')

val_location_frac = 0.2
empty_image_selection_fraction = 0.1


#%% Prepare training folders

os.makedirs(yolo_train_folder,exist_ok=True)
os.makedirs(yolo_val_folder,exist_ok=True)


#%% Convert from COCO to YOLO

coco_to_yolo(input_image_folder,yolo_all_folder,input_file,
             source_format='coco',
             overwrite_images=False,
             create_image_and_label_folders=False,
             class_file_name=class_file_name,
             allow_empty_annotations=True,
             clip_boxes=True,
             image_id_to_output_image_json_file=image_id_mapping_file
             )


#%% Load the original metadata

# ...and the file that maps image IDs in the original COCO file to YOLO images.

import json

with open(input_file,'r') as f:
    input_metadata = json.load(f)
    
with open(image_id_mapping_file,'r') as f:
    image_id_mappings = json.load(f)

assert (len(image_id_mappings) == len(input_metadata['images']))

n_images = len(image_id_mappings)

print('Loaded metadata and mappings for {} images'.format(n_images))


#%% Split locations into train and val

random_seed = 5
random.seed(random_seed)

from collections import defaultdict
location_to_images = defaultdict(list)
image_id_to_annotations = defaultdict(list)
image_id_to_image = {}

# im = input_metadata['images'][0]
for im in input_metadata['images']:
    location_to_images[im['location']].append(im)
    image_id_to_image[im['id']] = im

# ann = input_metadata['annotations'][0]
for ann in input_metadata['annotations']:
    if 'bbox' in ann:
        assert len(ann['bbox']) == 4
        im = image_id_to_image[ann['image_id']]
        image_id_to_annotations[ann['image_id']].append(ann)        
        
print('Found {} unique locations'.format(len(location_to_images)))

n_locations = len(location_to_images)

n_val_locations = int(val_location_frac*n_locations)
val_location_ids = random.choices(list(location_to_images.keys()),k=n_val_locations)

val_images = []
train_images = []
omitted_images = []

n_val_annotations = 0
n_train_annotations = 0

print('Counts by location:\n')
for location_id in location_to_images.keys():
    
    # Is this a val location?
    val_location = (location_id in val_location_ids)        
        
    # Count images and annotations at this location
    loc_image_count = len(location_to_images[location_id])    
    loc_annotation_count = 0
    
    empty_images_this_location = []
    non_empty_images_this_location = []
    
    for im in location_to_images[location_id]:
        
        n_annotations_this_image = len(image_id_to_annotations[im['id']])
        if n_annotations_this_image == 0:
            empty_images_this_location.append(im)
        else:
            non_empty_images_this_location.append(im)
        loc_annotation_count += n_annotations_this_image
        
    # ...for each image
    
    val_string = ''
    if location_id in val_location_ids:
        val_string = ' (val)'
    print(location_id + ': {} images ({} empty) ({} annotations){}'.format(
        loc_image_count,len(empty_images_this_location),loc_annotation_count,val_string))
    
    n_empty_images_to_select = int(empty_image_selection_fraction * len(empty_images_this_location))
    selected_empty_images_this_location = random.choices(empty_images_this_location,
                                                         k=n_empty_images_to_select)
    
    for im in empty_images_this_location:
        if im not in selected_empty_images_this_location:
            omitted_images.append(im)
            
    if val_location:
        n_val_annotations =+ loc_annotation_count
        val_images.extend(non_empty_images_this_location)
        val_images.extend(selected_empty_images_this_location)
    else:
        n_train_annotations += loc_annotation_count
        train_images.extend(non_empty_images_this_location)
        train_images.extend(selected_empty_images_this_location)
        
train_image_ids = set([im['id'] for im in train_images])
val_image_ids = set([im['id'] for im in val_images])
omitted_image_ids = set([im['id'] for im in omitted_images])

for im in input_metadata['images']:
    assert im['id'] in train_image_ids or im['id'] in val_image_ids or im['id'] in omitted_image_ids

n_images = len(input_metadata['images'])
n_omitted_images = len(omitted_images)
n_val_images = len(val_images)
n_train_images = len(train_images)

print('')
print('{} train images ({:.2f}%)'.format(n_train_images,(100*n_train_images/n_images)))
print('{} val images ({:.2f}%)'.format(n_val_images,(100*n_val_images/n_images)))
print('{} omitted images ({:.2f}%)'.format(n_omitted_images,(100*n_omitted_images/n_images)))
print('')
print('{} train annotations ({:.2f}%)'.format(n_train_annotations,
                                            (100*n_train_annotations/(n_val_annotations+n_train_annotations))))
print('{} val annotations ({:.2f}%)'.format(n_val_annotations,
                                            (100*n_val_annotations/(n_val_annotations+n_train_annotations))))


#%% Copy images and annotations into train and val folders

from tqdm import tqdm
import shutil

# im = input_metadata['images'][0]
for im in tqdm(input_metadata['images']):
    
    assert im['id'] in image_id_mappings
    assert im['id'] in train_image_ids or im['id'] in val_image_ids or im['id'] in omitted_image_ids
    
    yolo_id = image_id_mappings[im['id']]
    yolo_image_filename_relative = yolo_id + '.jpg'
    yolo_txt_filename_relative = yolo_id + '.txt'
    
    source_image = os.path.join(yolo_all_folder,yolo_image_filename_relative)
    source_txt = os.path.join(yolo_all_folder,yolo_txt_filename_relative)
    
    assert os.path.isfile(source_image)
    
    if im['id'] in val_image_ids:
        dest_folder = yolo_val_folder
    elif im['id'] in train_image_ids:
        dest_folder = yolo_train_folder
    else:
        assert im['id'] in omitted_image_ids
        continue
        
    dest_image = os.path.join(dest_folder,yolo_image_filename_relative)
    if not os.path.isfile(dest_image):
        shutil.copy(source_image,dest_image)
    
    dest_txt = os.path.join(dest_folder,yolo_txt_filename_relative)
    if os.path.isfile(source_txt) and not os.path.isfile(dest_txt):
        shutil.copy(source_txt,dest_txt)

# ...for each image


#%% Confirm that the val set hasn't changed between runs

val_file = os.path.join(base_folder,'val.json')
with open(val_file,'r') as f:
    val_images = json.load(f)
val_files_previous = [im['file_name'] for im in val_images['images']]
val_files_current = os.listdir(yolo_val_folder)
val_files_current = [fn for fn in val_files_current if fn.lower().endswith('.jpg')]
assert len(val_files_current) == len(val_files_previous)
assert set(val_files_current) == set(val_files_previous)


#%% Generate the YOLO training dataset file

# Read class names
class_file_path = os.path.join(yolo_all_folder,class_file_name)
with open(class_file_path,'r') as f:
    class_lines = f.readlines()
class_lines = [s.strip() for s in class_lines]    
class_lines = [s for s in class_lines if len(s) > 0]

# Write dataset.yaml
with open(yolo_dataset_file,'w') as f:
    
    yolo_train_folder_relative = os.path.relpath(yolo_train_folder,base_folder)
    yolo_val_folder_relative = os.path.relpath(yolo_val_folder,base_folder)
    
    f.write('# Train/val sets\n')
    f.write('path: {}\n'.format(base_folder))
    f.write('train: {}\n'.format(yolo_train_folder_relative))
    f.write('val: {}\n'.format(yolo_val_folder_relative))
    
    f.write('\n')
    
    f.write('# Classes\n')
    f.write('names:\n')
    for i_class,class_name in enumerate(class_lines):
        f.write('  {}: {}\n'.format(i_class,class_name))


#%% Train YOLOv5

# Environment prep
"""
conda create --name yolov5
conda activate yolov5
conda install pip
cd ~/git
git clone https://github.com/ultralytics/yolov5 yolov5-current # clone
pip install -r requirements.txt  # install

# Because of random CUDA errors:
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
"""

#
# Train
#
# Note to self:
#
# https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results
#
# Pre-trained model sizes:
#
# https://github.com/ultralytics/yolov5/releases/tag/v6.1
#

"""
cd ~/git/yolov5-current
conda activate yolov5
LD_LIBRARY_PATH=
PYTHONPATH=
python train.py --img 1280 --batch -1 --epochs 200 --weights yolov5x6.pt --device 0,1 --project noaa-fish --name noaa-fish-yolov5x6-01-1280-200 --data "/home/user/data/noaa-fish/dataset.yaml"
"""


#%% Run the trained model using val.py

"""
cd ~/git/yolov5-current
conda activate yolov5
LD_LIBRARY_PATH=
PYTHONPATH=
"""

image_size = 1280
batch_size = 1
model_file = '/home/user/data/noaa-fish/models/noaa-fish-yolov5x6-01-1280-200-best.pt'
project_folder = '/home/user/tmp/noaa-fish/results'
dataset_file = '/home/user/data/noaa-fish/dataset.yaml'
conf_thres = 0.001
device_string = '0'

assert os.path.isfile(model_file)
assert os.path.isfile(dataset_file)

# First without augmentation...

results_folder = 'noaa-yolov5x6-val-noaug'
os.makedirs(os.path.join(project_folder,results_folder),exist_ok=True)

cmd = f'python val.py --img {image_size} --batch-size {batch_size} --weights {model_file}' \
  f' --project {project_folder} --name {results_folder} --data {dataset_file}' \
  f' --conf-thres {conf_thres} --device {device_string} --save-json --exist-ok'

# import clipboard; clipboard.copy(cmd)

# Then with augmentation...

results_folder = 'noaa-yolov5x6-val-aug'
os.makedirs(os.path.join(project_folder,results_folder),exist_ok=True)

cmd = f'python val.py --img {image_size} --batch-size {batch_size} --weights {model_file}' \
  f' --project {project_folder} --name {results_folder} --data {dataset_file}' \
  f' --conf-thres {conf_thres} --device {device_string} --save-json --augment --exist-ok'

# import clipboard; clipboard.copy(cmd)


#%% Convert results to MegaDetector format

project_folder = '/home/user/tmp/noaa-fish/results'
results_folders = ['noaa-yolov5x6-val-noaug','noaa-yolov5x6-val-aug']

results_files_md = []

from data_management.yolo_output_to_md_output import yolo_json_output_to_md_output

yolo_category_id_to_name = {1:'animal'}

# results_folder = results_folders[0]
for results_folder in results_folders:
    
    results_folder_full_path = os.path.join(project_folder,results_folder)
    assert os.path.isdir(results_folder_full_path)
    json_files = [fn for fn in os.listdir(results_folder_full_path) \
                  if fn.endswith('.json')]
    assert len(json_files) == 1
    yolo_json_file = os.path.join(results_folder_full_path,json_files[0])
    md_formatted_results_file = yolo_json_file.replace('.json','_md-format.json')
    assert md_formatted_results_file != yolo_json_file
    model_short_name = results_folder
    
    results_files_md.append(md_formatted_results_file)
    yolo_json_output_to_md_output(yolo_json_file,
                                  image_folder=yolo_val_folder,
                                  output_file=md_formatted_results_file,
                                  yolo_category_id_to_name=yolo_category_id_to_name,
                                  detector_name=model_short_name,
                                  image_id_to_relative_path=None,
                                  offset_yolo_class_ids=False)

# ...for each results folder


#%% Run the trained model using the MD pred pipeline 

"""
export PYTHONPATH=/home/user/git/cameratraps/:/home/user/git/yolov5-current:/home/user/git/ai4eutils
cd ~/git/cameratraps/detection/
conda activate yolov5

MODEL_NAME="noaa-fish-yolov5x6-01-1280-200-best"
MODEL_FILE="/home/user/data/noaa-fish/models/${MODEL_NAME}.pt"

python run_detector_batch.py ${MODEL_FILE} /home/user/data/noaa-fish/val "/home/user/data/noaa-fish/results/${MODEL_NAME}-val.json" --recursive --quiet --output_relative_filenames

python run_detector_batch.py ${MODEL_FILE} /home/user/data/noaa-fish/train "/home/user/data/noaa-fish/results/${MODEL_NAME}-train.json" --recursive --quiet --output_relative_filenames
"""


#%% Visualize raw results (from the command line)

"""
cd ~/git/cameratraps/api/batch_processing/postprocessing/
conda deactivate

MODEL_NAME="noaa-fish-yolov5x6-01-1280-200-best"

python postprocess_batch_results.py /home/user/data/noaa-fish/results/${MODEL_NAME}-val.json /home/user/data/noaa-fish/preview/${MODEL_NAME}-val --image_base_dir /home/user/data/noaa-fish/val --n_cores 10 --confidence_threshold 0.5
xdg-open /home/user/data/noaa-fish/preview/${MODEL_NAME}-val/index.html

python postprocess_batch_results.py /home/user/data/noaa-fish/results/${MODEL_NAME}-train.json /home/user/data/noaa-fish/preview/${MODEL_NAME}-train --image_base_dir /home/user/data/noaa-fish/train --n_cores 10 --confidence_threshold 0.5
xdg-open /home/user/data/noaa-fish/preview/${MODEL_NAME}-train/index.html
"""


#%% Filter out tall narrow things that aren't fish

import json

results_files_md_filtered = []

for results_file in results_files_md:
    
    results_file_out = results_file.replace('.json','-filtered.json')
    assert results_file_out != results_file

    results_files_md_filtered.append(results_file_out)
        
    assert os.path.isfile(results_file)
    
    with open(results_file,'r') as f:
        d = json.load(f)
    
    for im in tqdm(d['images']):
        
        for det in im['detections']:
            
            # Box with in normalized units
            w = det['bbox'][2]
            aspect = det['bbox'][3] / det['bbox'][2]
            
            # Is this a very tall, narrow box?
            if aspect > 5 and w < 0.03 and det['conf'] > 0:
                
                # Set it to negative confidence
                det['conf']  = -1 * det['conf']
    
    # ...for each image
    
    with open(results_file_out,'w') as f:
        json.dump(d,f,indent=1)

# ...for each results file

            
#%% Visualize results 

import path_utils

from api.batch_processing.postprocessing.postprocess_batch_results import (
    PostProcessingOptions, process_batch_results)

gt_file = '/home/user/data/noaa-fish/val.json'

# results_file = results_files_md_filtered[0]
for results_file in results_files_md_filtered: 

    assert os.path.isfile(results_file)
    
    # E.g. noaa-yolov5x6-val-noaug
    run_name = results_file.split('/')[-2]
    
    postprocessing_output_folder = os.path.join(os.path.dirname(results_file),'preview')
    
    options = PostProcessingOptions()
    options.image_base_dir = yolo_val_folder
    options.ground_truth_json_file = gt_file
    options.include_almost_detections = True
    options.num_images_to_sample = 7500
    options.confidence_threshold = 0.3
    options.almost_detection_confidence_threshold = options.confidence_threshold - 0.05
    options.separate_detections_by_category = True
    options.sample_seed = 0
    options.viz_target_width = 1280
    options.negative_classes.append('#NO_LABELS#')
    
    options.parallelize_rendering = True
    options.parallelize_rendering_n_cores = 20
    options.parallelize_rendering_with_threads = False
    
    os.makedirs(postprocessing_output_folder, exist_ok=True)
    print('Processing to {}'.format(postprocessing_output_folder))
    
    options.api_output_file = results_file
    options.output_dir = postprocessing_output_folder
    ppresults = process_batch_results(options)
    html_output_file = ppresults.output_html_file
    
    path_utils.open_file(html_output_file)
    

#%% Compare results with and without augmentation

import itertools

from api.batch_processing.postprocessing.compare_batch_results import (
    BatchComparisonOptions,PairwiseBatchComparisonOptions,compare_batch_results)

options = BatchComparisonOptions()

options.job_name = 'noaa-fish-aug-comparison'
options.output_folder = os.path.expanduser('~/tmp/noaa-fish/results/augmentation-comparison')
options.image_folder = yolo_val_folder

options.pairwise_options = []

filenames = results_files_md_filtered

detection_thresholds = [0.15,0.15]

assert len(detection_thresholds) == len(filenames)

rendering_thresholds = [(x*0.6666) for x in detection_thresholds]

# Choose all pairwise combinations of the files in [filenames]
for i, j in itertools.combinations(list(range(0,len(filenames))),2):
        
    pairwise_options = PairwiseBatchComparisonOptions()
    
    pairwise_options.results_filename_a = filenames[i]
    pairwise_options.results_filename_b = filenames[j]
    
    pairwise_options.rendering_confidence_threshold_a = rendering_thresholds[i]
    pairwise_options.rendering_confidence_threshold_b = rendering_thresholds[j]
    
    pairwise_options.detection_thresholds_a = {'animal':detection_thresholds[i]}
    pairwise_options.detection_thresholds_b = {'animal':detection_thresholds[j]}
    options.pairwise_options.append(pairwise_options)

results = compare_batch_results(options)

from path_utils import open_file # from ai4eutils
open_file(results.html_output_file)


#%% TODO: train and validate YOLOv8

"""
yolo detect train data="/home/user/data/noaa-fish/dataset.yaml" model=yolov8n.pt epochs=200 imgsz=1280
"""

# 1280 model coming out soon
#
# https://github.com/ultralytics/ultralytics/issues/338


