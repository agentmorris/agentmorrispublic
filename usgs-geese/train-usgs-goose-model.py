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

* Generate a bunch more hard negative patches and manually review them, especially if
  they look qualitatively different.

* Tinker with box size

* Tinker with test-time augmentation

* Possibly parallelize patch generation if I find myself running it often

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

# I don't fully understand what that's about, but this fixed it:
# 
# pip uninstall nvidia_cublas_cu11
#
# Strangely, when I run train.py again, it reinstalls the missing CUDA components,
# and everything is fine, but then the error comes back the *next* time I run it.
# So I pip uninstall again, and the circle of life continues.

"""

# Train
"""
~/limit_gpu_power
cd ~/git/yolov5-current

# I usually have an older commit of yolov5 on my PYTHONPATH, remove it.
export PYTHONPATH=
conda activate yolov5

# On my 2x24GB GPU setup, a batch size of 16 failed, but 8 was safe.  Autobatch did not
# work; I got an incomprehensible error that I decided not to fix, but I'm pretty sure
# it would have come out with a batch size of 8 anyway.
python train.py --img 1280 --batch 8 --epochs 100 --weights yolov5x6.pt --device 0,1 --project usgs-geese --name usgs-geese-yolov5x6-b8-img1280-e100 --data "/home/user/data/usgs-geese-mini-500/dataset.yaml"
"""

# Monitor training
"""
cd ~/git/yolov5-current/usgs-geese/usgs-geese-yolov5x6-autobatch-1280-300
tensorboard --logdir .
"""

pass


#%% Back up trained weights

"""
cp ~/git/yolov5-current/usgs-geese/usgs-geese-yolov5x6-autobatch-1280-300/weights/best.pt ~/models/usgs-geese/usgs-geese-yolov5x6-autobatch-1280-300-mini-2023.03.12-best.pt

cp ~/git/yolov5-current/usgs-geese/usgs-geese-yolov5x6-autobatch-1280-300/weights/last.pt ~/models/usgs-geese/usgs-geese-yolov5x6-autobatch-1280-300-mini-2023.03.12-last.pt
"""

pass


#%% Validation

#
# Val
#

"""
MODEL_FILE="/home/user/models/usgs-geese/usgs-geese-yolov5x6-autobatch-1280-300-mini-2023.03.12-best.pt"
python val.py --img 1280 --batch-size 8 --weights ${MODEL_FILE} --project usgs-geese --name usgs-geese-mini --data "/home/user/data/usgs-geese/dataset.yaml" --conf-thres 0.1
"""

#
# Run the MD pred pipeline 
#

"""
export PYTHONPATH=/home/user/git/cameratraps/:/home/user/git/yolov5-current:/home/user/git/ai4eutils
cd ~/git/cameratraps/detection/
conda activate yolov5

MODEL_NAME="usgs-geese-yolov5x6-autobatch-1280-300-mini-2023.03.12-best.pt"
MODEL_FILE="/home/user/models/usgs-geese/${MODEL_NAME}"

python run_detector_batch.py ${MODEL_FILE} "/home/user/data/usgs-geese/yolo_val" "/home/user/data/usgs-geese/results/${MODEL_NAME}-val.json" --recursive --quiet --output_relative_filenames --class_mapping_filename "/home/user/models/usgs-geese/usgs-geese-class-mapping.json"

python run_detector_batch.py ${MODEL_FILE} "/home/user/data/usgs-geese/yolo_train" "/home/user/data/usgs-geese/results/${MODEL_NAME}-train.json" --recursive --quiet --output_relative_filenames --class_mapping_filename "/home/user/models/usgs-geese/usgs-geese-class-mapping.json"

"""

#
# Visualize results using the MD pipeline
#

"""
conda deactivate

cd ~/git/cameratraps/api/batch_processing/postprocessing/

MODEL_NAME="usgs-geese-yolov5x6-autobatch-1280-300-mini-2023.03.12-best.pt"

python postprocess_batch_results.py /home/user/data/usgs-geese/results/${MODEL_NAME}-val.json /home/user/data/usgs-geese/preview/${MODEL_NAME}-val --image_base_dir /home/user/data/usgs-geese/yolo_val --n_cores 10 --confidence_threshold 0.25
xdg-open /home/user/data/usgs-geese/preview/${MODEL_NAME}-val/index.html

python postprocess_batch_results.py /home/user/data/usgs-geese/results/${MODEL_NAME}-train.json /home/user/data/usgs-geese/preview/${MODEL_NAME}-train --image_base_dir /home/user/data/usgs-geese/yolo_train --n_cores 10 --confidence_threshold 0.25
xdg-open /home/user/data/usgs-geese/preview/${MODEL_NAME}-train/index.html

"""

pass
