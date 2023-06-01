########
#
# usgs-geese-training.py
#
# This file documents the model training process, starting from where usgs-geese-training-data-prep.py
# leaves off.  Training happens at the yolov5 CLI, and the exact command line arguments are documented
# in the "Train" cell.
#
# Later cells in this file also:
#
# * Run the YOLOv5 validation scripts
# * Convert YOLOv5 val results to MD .json format
# * Use the MD visualization pipeline to visualize results
# * Use the MD inference pipeline to run the trained model
#
########

#%% TODO

"""

* Adjust hyperparameters (increase augmentation, match MDv5 parameters)

https://github.com/agentmorris/MegaDetector/blob/main/detection/detector_training/experiments/megadetector_v5_yolo/hyp_mosaic.yml

https://github.com/agentmorris/MegaDetector/tree/main/detection#training-with-yolov5

* Add hard negative patches, and/or mine for hard negative images

* Tinker with box size

* Tinker with test-time augmentation

* Try smaller YOLOv5's

* Try 1280px YOLOv8 when it's available

* Try fancier patch sampling to minimize the number of birds that are split across
  patches.

"""


#%% Train

# Tips:
#
# https://github.com/ultralytics/yolov5/wiki/Tips-for-Best-Training-Results


## Environment prep

"""
conda create --name yolov5
conda activate yolov5
conda install pip
git clone https://github.com/ultralytics/yolov5 yolov5-current
cd yolov5-current
pip install -r requirements.txt
"""

#
# I got this error:
#    
# OSError: /home/user/anaconda3/envs/yolov5/lib/python3.10/site-packages/nvidia/cublas/lib/libcublas.so.11: undefined symbol: cublasLtGetStatusString, version libcublasLt.so.11
#
# There are two ways I've found to fix this:
#
# CUDA was on my LD_LIBRARY_PATH, so this fixes it:
#
# LD_LIBRARY_PATH=
#
# Or if I do this:
# 
# pip uninstall nvidia_cublas_cu11
#
# ...when I run train.py again, it reinstalls the missing CUDA components,
# and everything is fine, but then the error comes back the *next* time I run it.
#
# So I pip uninstall again, and the circle of life continues.
#


## Training

"""
cd ~/git/yolov5-current

# I usually have an older commit of yolov5 on my PYTHONPATH, remove it.
export PYTHONPATH=
LD_LIBRARY_PATH=
conda activate yolov5

# On my 2x24GB GPU setup, a batch size of 16 failed, but 8 was safe.  Autobatch did not
# work; I got an incomprehensible error that I decided not to fix, but I'm pretty sure
# it would have come out with a batch size of 8 anyway.
BATCH_SIZE=8
IMAGE_SIZE=1280
EPOCHS=200
DATA_YAML_FILE=/home/user/data/usgs-geese/dataset.yaml

# TRAINING_RUN_NAME=usgs-geese-yolov5x-b${BATCH_SIZE}-img${IMAGE_SIZE}-e${EPOCHS}
TRAINING_RUN_NAME=usgs-geese-yolov5x-nolinks-b${BATCH_SIZE}-img${IMAGE_SIZE}-e${EPOCHS}

python train.py --img ${IMAGE_SIZE} --batch ${BATCH_SIZE} --epochs ${EPOCHS} --weights yolov5x6.pt --device 0,1 --project usgs-geese --name ${TRAINING_RUN_NAME} --data ${DATA_YAML_FILE}
"""


## Monitoring training

"""
cd ~/git/yolov5-current
conda activate yolov5
tensorboard --logdir usgs-geese
"""


## Resuming training

"""
cd ~/git/yolov5-current
conda activate yolov5
LD_LIBRARY_PATH=
export PYTHONPATH=
python train.py --resume
"""

pass


#%% Back up trained weights

"""
TRAINING_RUN_NAME="usgs-geese-yolov5x6-b8-img1280-e100"
TRAINING_OUTPUT_FOLDER="/home/user/git/yolov5-current/usgs-geese/${TRAINING_RUN_NAME}/weights"

cp ${TRAINING_OUTPUT_FOLDER}/best.pt ~/models/usgs-geese/${TRAINING_RUN_NAME}-best.pt
cp ${TRAINING_OUTPUT_FOLDER}/last.pt ~/models/usgs-geese/${TRAINING_RUN_NAME}-last.pt
"""

pass


#%% Validation with YOLOv5

import os

model_base = os.path.expanduser('~/models/usgs-geese')
training_run_names = [
    'usgs-geese-yolov5x6-b8-img1280-e125-of-200-20230401-ss',
    'usgs-geese-yolov5x6-b8-img1280-e49-of-200-20230401-dm'
]

data_folder = os.path.expanduser('~/data/usgs-geese')
image_size = 1280

# Note to self: validation batch size appears to have no impact on mAP
# (it shouldn't, but I verified that explicitly)
batch_size_val = 8

project_name = os.path.expanduser('~/tmp/usgs-geese-val')
data_file = os.path.join(data_folder,'dataset.yaml')
augment = True

assert os.path.isfile(data_file)

model_file_to_command = {}

# training_run_name = training_run_names[0]
for training_run_name in training_run_names:
    model_file_base = os.path.join(model_base,training_run_name)
    model_files = [model_file_base + s for s in ('-last.pt','-best.pt')]
    
    # model_file = model_files[0]
    for model_file in model_files:
        assert os.path.isfile(model_file)
        
        model_short_name = os.path.basename(model_file).replace('.pt','')
        cmd = 'python val.py --img {} --batch-size {} --weights {} --project {} --name {} --data {} --save-txt --save-json --save-conf --exist-ok'.format(
            image_size,batch_size_val,model_file,project_name,model_short_name,data_file)        
        if augment:
            cmd += ' --augment'
        model_file_to_command[model_file] = cmd
        
    # ...for each model
    
# ...for each training run    

for k in model_file_to_command.keys():
    # print(os.path.basename(k))
    print('')
    cmd = model_file_to_command[k]
    print(cmd + '\n')
    

"""
Results without augmentation
"""

"""
usgs-geese-yolov5x6-b8-img1280-e125-of-200-20230401-ss-last.pt

  Class     Images  Instances          P          R      mAP50   mAP50-95:
    all      11547     136014      0.591      0.562      0.513      0.287
  Brant      11547     101770      0.877      0.919      0.901       0.53
  Other      11547      21246      0.694       0.35      0.398      0.223
   Gull      11547       1594      0.481      0.561      0.422      0.204
 Canada      11547      10961      0.761      0.793      0.783      0.445
Emperor      11547        443      0.141      0.187     0.0619     0.0325
Speed: 0.5ms pre-process, 53.6ms inference, 0.9ms NMS per image at shape (8, 3, 1280, 1280)


usgs-geese-yolov5x6-b8-img1280-e125-of-200-20230401-ss-best.pt

  Class     Images  Instances          P          R      mAP50   mAP50-95:
    all      11547     136014      0.618      0.563      0.539      0.295
  Brant      11547     101770      0.861      0.927      0.908      0.526
  Other      11547      21246      0.734      0.358      0.419      0.219
   Gull      11547       1594      0.607      0.528       0.45      0.213
 Canada      11547      10961      0.766      0.853      0.844      0.479
Emperor      11547        443       0.12      0.147      0.074     0.0372
Speed: 0.5ms pre-process, 53.8ms inference, 1.1ms NMS per image at shape (8, 3, 1280, 1280)


usgs-geese-yolov5x6-b8-img1280-e49-of-200-20230401-dm-last.pt

  Class     Images  Instances          P          R      mAP50   mAP50-95:
    all      11547     136014      0.621      0.559      0.536       0.29
  Brant      11547     101770      0.865      0.925      0.908       0.53
  Other      11547      21246      0.742      0.355       0.42      0.214
   Gull      11547       1594      0.601      0.523      0.442      0.203
 Canada      11547      10961      0.776      0.848      0.839      0.467
Emperor      11547        443      0.119      0.142     0.0708     0.0356
Speed: 0.5ms pre-process, 53.5ms inference, 1.2ms NMS per image at shape (8, 3, 1280, 1280)


usgs-geese-yolov5x6-b8-img1280-e49-of-200-20230401-dm-best.pt

  Class     Images  Instances          P          R      mAP50   mAP50-95:
    all      11547     136014      0.621      0.559      0.536       0.29
  Brant      11547     101770      0.865      0.926      0.908       0.53
  Other      11547      21246      0.742      0.355       0.42      0.214
   Gull      11547       1594      0.601      0.523      0.442      0.203
 Canada      11547      10961      0.776      0.848      0.839      0.467
Emperor      11547        443      0.119      0.142     0.0708     0.0356
Speed: 0.5ms pre-process, 53.2ms inference, 1.1ms NMS per image at shape (8, 3, 1280, 1280)

"""

"""
Results with augmentation
"""

"""
usgs-geese-yolov5x6-b8-img1280-e125-of-200-20230401-ss-last.pt

Class     Images  Instances          P          R      mAP50   mAP50-95:
    all      11547     136014      0.587      0.565      0.515      0.324
  Brant      11547     101770      0.868      0.924      0.899      0.572
  Other      11547      21246      0.689      0.357      0.392      0.233
   Gull      11547       1594      0.486      0.551      0.435      0.283
 Canada      11547      10961      0.755      0.809      0.792      0.497
Emperor      11547        443      0.135      0.183     0.0579     0.0336

  
usgs-geese-yolov5x6-b8-img1280-e125-of-200-20230401-ss-best.pt

  Class     Images  Instances          P          R      mAP50   mAP50-95:
    all      11547     136014      0.601      0.563      0.535      0.324
  Brant      11547     101770      0.844      0.928      0.906      0.562
  Other      11547      21246      0.729       0.36      0.406      0.225
   Gull      11547       1594      0.553      0.528       0.44       0.28
 Canada      11547      10961      0.764      0.857      0.849      0.513
Emperor      11547        443      0.118       0.14     0.0731      0.041
Speed: 0.5ms pre-process, 118.5ms inference, 1.8ms NMS per image at shape (8, 3, 1280, 1280)


usgs-geese-yolov5x6-b8-img1280-e49-of-200-20230401-dm-last.pt

  Class     Images  Instances          P          R      mAP50   mAP50-95:
    all      11547     136014      0.607      0.561      0.531      0.326
  Brant      11547     101770      0.852      0.928      0.908      0.571
  Other      11547      21246      0.738      0.361       0.41      0.226
   Gull      11547       1594      0.564      0.529      0.423      0.274
 Canada      11547      10961      0.769      0.857      0.843       0.52
Emperor      11547        443      0.112      0.132     0.0694     0.0403
Speed: 0.5ms pre-process, 118.5ms inference, 1.7ms NMS per image at shape (8, 3, 1280, 1280)


usgs-geese-yolov5x6-b8-img1280-e49-of-200-20230401-dm-best.pt

 Class     Images  Instances          P          R      mAP50   mAP50-95:
    all      11547     136014      0.607      0.561      0.531      0.326
  Brant      11547     101770      0.852      0.928      0.908      0.571
  Other      11547      21246      0.738      0.361       0.41      0.226
   Gull      11547       1594      0.564      0.529      0.423      0.274
 Canada      11547      10961      0.769      0.857      0.843       0.52
Emperor      11547        443      0.112      0.132     0.0694     0.0403

"""

"""
cd ~/git/yolov5-current
conda activate yolov5
LD_LIBRARY_PATH=
export PYTHONPATH=
"""

"""
TRAINING_RUN_NAME="usgs-geese-yolov5x6-b8-img1280-e100"
MODEL_FILE="/home/user/models/usgs-geese/${TRAINING_RUN_NAME}-best.pt"
DATA_FOLDER="/home/user/data/usgs-geese"

python val.py --img 1280 --batch-size 8 --weights ${MODEL_FILE} --project usgs-geese --name ${TRAINING_RUN_NAME} --data ${DATA_FOLDER}/dataset.yaml 
"""
pass


#%% Convert YOLO val .json results to MD .json format

# pip install jsonpickle humanfriendly tqdm skicit-learn

import os
from data_management import yolo_output_to_md_output

import json
import glob

class_mapping_file = os.path.expanduser('~/data/usgs-geese/usgs-geese-md-class-mapping.json')
with open(class_mapping_file,'r') as f:
    category_id_to_name = json.load(f)
                        
base_folder = os.path.expanduser('~/tmp/usgs-geese-val')
run_folders = os.listdir(base_folder)
run_folders = [os.path.join(base_folder,s) for s in run_folders]
run_folders = [s for s in run_folders if os.path.isdir(s)]

image_base = os.path.expanduser('~/data/usgs-geese/yolo_val')
image_files = glob.glob(image_base + '/*.jpg')

prediction_files = []

# run_folder = run_folders[0]
for run_folder in run_folders:
    prediction_files_this_folder = glob.glob(run_folder+'/*_predictions.json')
    assert len(prediction_files_this_folder) <= 1
    if len(prediction_files_this_folder) == 1:
        prediction_files.append(prediction_files_this_folder[0])        

md_format_prediction_files = []

# prediction_file = prediction_files[0]
for prediction_file in prediction_files:

    detector_name = os.path.splitext(os.path.basename(prediction_file))[0].replace('_predictions','')
    
    # print('Converting {} to MD format'.format(prediction_file))
    output_file = prediction_file.replace('.json','_md-format.json')
    assert output_file != prediction_file
    
    yolo_output_to_md_output.yolo_json_output_to_md_output(
        yolo_json_file=prediction_file,
        image_folder=image_base,
        output_file=output_file,
        yolo_category_id_to_name=category_id_to_name,                              
        detector_name=detector_name,
        image_id_to_relative_path=None,
        offset_yolo_class_ids=False)    
    
    md_format_prediction_files.append(output_file)

# ...for each prediction file


#%% Visualize results with the MD visualization pipeline

postprocessing_output_folder = os.path.expanduser('~/tmp/usgs-geese-previews')

import path_utils

from api.batch_processing.postprocessing.postprocess_batch_results import (
    PostProcessingOptions, process_batch_results)

# prediction_file = md_format_prediction_files[0]
for prediction_file in md_format_prediction_files:
    
    assert '_md-format.json' in prediction_file
    base_task_name = os.path.basename(prediction_file).replace('_md-format.json','')

    options = PostProcessingOptions()
    options.image_base_dir = image_base
    options.include_almost_detections = True
    options.num_images_to_sample = 7500
    options.confidence_threshold = 0.15
    options.almost_detection_confidence_threshold = options.confidence_threshold - 0.05
    options.ground_truth_json_file = None
    options.separate_detections_by_category = True
    # options.sample_seed = 0
    
    options.parallelize_rendering = True
    options.parallelize_rendering_n_cores = 16
    options.parallelize_rendering_with_threads = False
    
    output_base = os.path.join(postprocessing_output_folder,
        base_task_name + '_{:.3f}'.format(options.confidence_threshold))
    
    os.makedirs(output_base, exist_ok=True)
    print('Processing to {}'.format(output_base))
    
    options.api_output_file = prediction_file
    options.output_dir = output_base
    ppresults = process_batch_results(options)
    html_output_file = ppresults.output_html_file
    path_utils.open_file(html_output_file)

# ...for each prediction file


#%%

#
# Run the MD pred pipeline 
#

"""
export PYTHONPATH=/home/user/git/MegaDetector
cd ~/git/MegaDetector/detection/
conda activate yolov5

TRAINING_RUN_NAME="usgs-geese-yolov5x6-b8-img1280-e100"
MODEL_FILE="/home/user/models/usgs-geese/${TRAINING_RUN_NAME}-best.pt"
DATA_FOLDER="/home/user/data/usgs-geese-mini-500"
RESULTS_FOLDER=${DATA_FOLDER}/results

python run_detector_batch.py ${MODEL_FILE} ${DATA_FOLDER}/yolo_val ${RESULTS_FOLDER}/${TRAINING_RUN_NAME}-val.json --recursive --quiet --output_relative_filenames --class_mapping_filename ${DATA_FOLDER}/usgs-geese-md-class-mapping.json

python run_detector_batch.py ${MODEL_FILE} ${DATA_FOLDER}/yolo_train ${RESULTS_FOLDER}/${TRAINING_RUN_NAME}-train.json --recursive --quiet --output_relative_filenames --class_mapping_filename ${DATA_FOLDER}/usgs-geese-md-class-mapping.json

"""

#
# Visualize results using the MD pipeline
#

"""
conda deactivate

cd ~/git/MegaDetector/api/batch_processing/postprocessing/

TRAINING_RUN_NAME="usgs-geese-yolov5x6-b8-img1280-e100"
DATA_FOLDER="/home/user/data/usgs-geese-mini-500"
RESULTS_FOLDER=${DATA_FOLDER}/results
PREVIEW_FOLDER=${DATA_FOLDER}/preview

python postprocess_batch_results.py ${RESULTS_FOLDER}/${TRAINING_RUN_NAME}-val.json ${PREVIEW_FOLDER}/${TRAINING_RUN_NAME}-val --image_base_dir ${DATA_FOLDER}/yolo_val --n_cores 12 --confidence_threshold 0.25 --parallelize_rendering_with_processes

python postprocess_batch_results.py ${RESULTS_FOLDER}/${TRAINING_RUN_NAME}-train.json ${PREVIEW_FOLDER}/${TRAINING_RUN_NAME}-train --image_base_dir ${DATA_FOLDER}/yolo_train --n_cores 12 --confidence_threshold 0.25 --parallelize_rendering_with_processes

xdg-open ${PREVIEW_FOLDER}/${TRAINING_RUN_NAME}-val/index.html 
xdg-open ${PREVIEW_FOLDER}/${TRAINING_RUN_NAME}-train/index.html

"""

pass
