#
# train-usgs-goose-model.py
#
# Not a lot of actual code here, just documenting model training.
#

#%% TODO

"""
* Adjust hyperparameters (increase augmentation, match MDv5 parameters)

https://github.com/microsoft/CameraTraps/blob/main/detection/detector_training/experiments/megadetector_v5_yolo/hyp_mosaic.yml

https://github.com/microsoft/CameraTraps/tree/main/detection#training-with-yolov5

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


# Environment prep
"""
conda create --name yolov5
conda activate yolov5
conda install pip
git clone https://github.com/ultralytics/yolov5 yolov5-current # clone
cd yolov5-current
pip install -r requirements.txt  # install

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
# So I pip uninstall again, and the circle of life continues.

"""

# Train
"""
# I can run this workload with both GPUs at 225W, not higher
~/limit_gpu_power
cd ~/git/yolov5-current

# I usually have an older commit of yolov5 on my PYTHONPATH, remove it.
export PYTHONPATH=
conda activate yolov5

# On my 2x24GB GPU setup, a batch size of 16 failed, but 8 was safe.  Autobatch did not
# work; I got an incomprehensible error that I decided not to fix, but I'm pretty sure
# it would have come out with a batch size of 8 anyway.
BATCH_SIZE=8
IMAGE_SIZE=1280
EPOCHS=200
DATA_YAML_FILE=/home/user/data/usgs-geese/dataset.yaml

TRAINING_RUN_NAME=usgs-geese-yolov5x6-b${BATCH_SIZE}-img${IMAGE_SIZE}-e${EPOCHS}

python train.py --img ${IMAGE_SIZE} --batch ${BATCH_SIZE} --epochs ${EPOCHS} --weights yolov5x6.pt --device 0,1 --project usgs-geese --name ${TRAINING_RUN_NAME} --data ${DATA_YAML_FILE}
"""

# Monitor training
"""
cd ~/git/yolov5-current
tensorboard --logdir usgs-geese
"""

# Resume training
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


#%% Validation

#
# Validate using YOLOv5's tools
#

"""
TRAINING_RUN_NAME="usgs-geese-yolov5x6-b8-img1280-e100"
MODEL_FILE="/home/user/models/usgs-geese/${TRAINING_RUN_NAME}-best.pt"
DATA_FOLDER="/home/user/data/usgs-geese-mini-500"

python val.py --img 1280 --batch-size 8 --weights ${MODEL_FILE} --project usgs-geese --name ${TRAINING_RUN_NAME} --data ${DATA_FOLDER}/dataset.yaml 
"""

#
# Run the MD pred pipeline 
#

"""
export PYTHONPATH=/home/user/git/cameratraps/:/home/user/git/yolov5-current:/home/user/git/ai4eutils
cd ~/git/cameratraps/detection/
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

cd ~/git/cameratraps/api/batch_processing/postprocessing/

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
