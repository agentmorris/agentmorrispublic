#
# tree-ring-test.py
#
# A test script to run this tree ring detector:
#
# https://github.com/Gregor-Mendel-Institute/TRG-ImageProcessing/
#
# ...on a few sample images, with a couple of different parameter values.
#

#%% Imports and constants

import os
import stat
import path_utils

model_fn = os.path.expanduser('~/models/mask_rcnn_treeringcrackscomb2_onlyring_0186.h5')
output_base = os.path.expanduser('~/tmp/tree-ring-results')
os.makedirs(output_base,exist_ok=True)
output_script = os.path.join(output_base,'run_tree_ring_analysis.sh')

dpi_values = [100,200]

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

for input_folder_name in input_folder_mapping.keys():
    
    input_folder = input_folder_mapping[input_folder_name]
    images = path_utils.find_images(input_folder,recursive=True)    
    images = [fn for fn in images if fn.lower().endswith('.tif')]

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
            
            cmd = 'python postprocessingCracksRings.py --weightRing "{}" --input "{}" --output_folder "{}" --print_detections=yes --dpi={} --run_ID="{}"'.format(
                model_fn,fn,output_folder,dpi,image_id)
            commands.append(cmd)

with open(output_script,'w') as f:
    for cmd in commands:
        f.write(cmd)
        f.write('\n')
        
st = os.stat(output_script)
os.chmod(output_script, st.st_mode | stat.S_IEXEC)

print('Running the model {} times'.format(len(commands)))


#%% Run the script

# ...manually...


#%% Remove empty folders

# find /home/user/tmp/tree-ring-results -empty -type d -delete


#%% Copy images to a single folder

import shutil
output_image_folder = os.path.join(output_base,'output-images')
os.makedirs(output_image_folder,exist_ok=True)

# fn = list(fn_to_output_folder.keys())[0]
for image_id in image_id_to_output_folder:
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
        output_original_fn = os.path.join(output_image_folder,image_id + '.original.tif')
        shutil.copyfile(image_id_to_original_image[image_id],output_original_fn)
        