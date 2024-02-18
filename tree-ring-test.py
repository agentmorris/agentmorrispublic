#
# tree-ring-test.py
#
# A test script to run this tree ring detector:
#
# https://github.com/Gregor-Mendel-Institute/TRG-ImageProcessing/
#
# ...on a few sample images, with a couple of different parameter values.
#


#%% Environment setup

"""
# Get the tree ring code
mkdir ~/git
cd ~/git
git clone https://github.com/Gregor-Mendel-Institute/TRG-ImageProcessing
cd ~/git/TRG-ImageProcessing

# This is likely no longer be necessary; when I first tried this code, there were some bug fixes
# that were only on the development branch.
# git checkout development

# Download the model weights
mkdir ~/models
cd ~/models
wget https://data.swarts.gmi.oeaw.ac.at/treeringcrackscomb2_onlyring20210121T1457/mask_rcnn_treeringcrackscomb2_onlyring_0186.h5 -O ~/models/mask_rcnn_treeringcrackscomb2_onlyring_0186.h5

# Create the Python environment
cd ~/git/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/Mask_RCNN
mamba env create -f environment.yml

# Activate the Python environment
mamba activate TreeRingCNN

# Disable CUDA if you have GPU problems... I have yet to get this working with a GPU.
# export CUDA_VISIBLE_DEVICES=""

# Set up the inference environment
cd ~/git/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/Mask_RCNN/postprocessing/
MODEL_FILE=${HOME}/models/mask_rcnn_treeringcrackscomb2_onlyring_0186.h5
OUTPUT_FOLDER=${HOME}/tmp/tree-test

# Run on TRG sample data to make sure everything is configured correctly
rm -rf ~/tmp/tree-test && mkdir -p ~/tmp/tree-test
INPUT_FILE=${HOME}/git/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/Mask_RCNN/training/sample_dataset/train/1350_00041008a_0_pSX1.965402638101432_pSY1.9654116736824034.tif
python postprocessingCracksRings.py --weightRing "${MODEL_FILE}" --input "${INPUT_FILE}" --output_folder "${OUTPUT_FOLDER}" --print_detections=yes --dpi=100 --run_ID=test
"""

#%% Imports and constants

import os
import stat
from md_utils import path_utils

model_fn = os.path.expanduser('~/models/mask_rcnn_treeringcrackscomb2_onlyring_0186.h5')
assert os.path.isfile(model_fn)

output_base = os.path.expanduser('~/tmp/tree-ring-results')
os.makedirs(output_base,exist_ok=True)

output_script = os.path.join(output_base,'run_tree_ring_analysis.sh')

dpi_values = [4800]

tree_rings_image_folder_raw = os.path.expanduser('~/data/tree-rings')
tree_rings_image_folder_cropped = os.path.expanduser('~/data/tree-rings-cropped')

input_folder_mapping = {
    'raw':tree_rings_image_folder_raw,
    'cropped':tree_rings_image_folder_cropped
}


#%% Generate bash script to run all our model invocations

commands = []
image_id_to_output_folder = {}
image_id_to_original_image = {}

# input_folder_name = next(iter(input_folder_mapping))
for input_folder_name in input_folder_mapping.keys():
    
    input_folder = input_folder_mapping[input_folder_name]
    images = path_utils.find_images(input_folder,recursive=True)    
    images = [fn for fn in images if fn.lower().endswith('.tif')]

    # dpi = dpi_values[0]
    for dpi in dpi_values:

        output_folder = os.path.join(output_base,input_folder_name + '_dpi_' + str(dpi))

        # fn = images[0]
        for fn in images:
            image_id = input_folder_name + '_' + \
                os.path.relpath(fn,input_folder).replace('/','_') + '_dpi_' + str(dpi)
            output_folder = os.path.join(output_base,image_id)
            os.makedirs(output_folder,exist_ok=True)           
            
            assert image_id not in image_id_to_output_folder
            image_id_to_output_folder[image_id] = output_folder
            image_id_to_original_image[image_id] = fn
            
            cmd = 'echo Processing image {} from folder {}'.format(fn,input_folder_name)
            commands.append(cmd)
            
            cmd = 'python postprocessingCracksRings.py --weightRing "{}" --input "{}" --output_folder "{}" --print_detections=yes --dpi={} --run_ID="{}"'.format(
                model_fn,fn,output_folder,dpi,image_id)
            commands.append(cmd)
            
            commands.append('')

with open(output_script,'w') as f:
    for cmd in commands:
        f.write(cmd)
        f.write('\n')
        
st = os.stat(output_script)
os.chmod(output_script, st.st_mode | stat.S_IEXEC)


#%% Run the script

# ...manually...

"""
mamba activate TreeRingCNN
export CUDA_VISIBLE_DEVICES=""
cd  cd ~/git/TRG-ImageProcessing/CoreProcessingPipelineScripts/CNN/Mask_RCNN/postprocessing/
~/tmp/tree-ring-results/run_tree_ring_analysis.sh
"""

# Debugging notes:
    
"""
export QT_DEBUG_PLUGINS=1

# Did not help
sudo apt-get install --reinstall libxcb-xinerama0

# Fixed the issue
sudo apt install libegl1 
"""


#%% Remove empty folders

# find /home/user/tmp/tree-ring-results -empty -type d -delete


#%% Copy images to a single folder

import shutil
from tqdm import tqdm
from md_visualization.visualization_utils import resize_image

output_image_folder = os.path.join(output_base,'output-images')
output_image_folder_resized = os.path.join(output_base,'output-images-resized')
os.makedirs(output_image_folder,exist_ok=True)
os.makedirs(output_image_folder_resized,exist_ok=True)

include_original_images = False

# fn = list(fn_to_output_folder.keys())[0]
for image_id in tqdm(image_id_to_output_folder):
    output_folder = image_id_to_output_folder[image_id]
    image_id = os.path.basename(output_folder)
    output_images = path_utils.find_images(output_folder,recursive=True)
    output_images = [fn for fn in output_images if fn.endswith('.png')]
    assert len(output_images) <= 1
    if len(output_images) == 0:
        print('No output file for {}'.format(image_id))
    else:
        output_fn = os.path.join(output_image_folder,image_id + '.png')
        shutil.copyfile(output_images[0],output_fn)
        if include_original_images:
            output_original_fn = os.path.join(output_image_folder,image_id + '.original.tif')
            # shutil.copyfile(image_id_to_original_image[image_id],output_original_fn)
        output_fn_resized = os.path.join(output_image_folder_resized,image_id + '-resized.jpg')
        resize_image(output_images[0],target_width=4000,target_height=-1,
                     output_file=output_fn_resized,quality=90)
        
        