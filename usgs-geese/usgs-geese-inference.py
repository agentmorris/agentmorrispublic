########
#
# usgs-geese-inference.py
#
# Run inference on a folder of images, by breaking each image into overlapping 
# 1280x1280 patches, running the model on each patch, and eliminating redundant
# boxes from the results.
#
# TODO:
#
# ** P0
#
# * None, all P0's are now in usgs-geese-postprocessing.py
#
# ** P1
#
# * Divide data into more chunks than devices, to support checkpointing
# * Refactor folder inference to be able to run on arbitary lists of input images
#
# ** P2
# 
# * Assess the impact of using augmentation on both accuracy and speed
#
########

#%% Constants and imports

import os
import stat
import json
import glob
import shutil
import humanfriendly

from multiprocessing.pool import ThreadPool
from multiprocessing.pool import Pool
from functools import partial
from tqdm import tqdm

import torch
from torchvision import ops

from md_utils import path_utils
from md_utils import process_utils
from md_visualization import visualization_utils as vis_utils
from data_management.yolo_output_to_md_output import yolo_json_output_to_md_output
from api.batch_processing.postprocessing import combine_api_outputs

# We will explicitly verify that images are actually this size
expected_image_width = 8688
expected_image_height = 5792

patch_size = (1280,1280)

# Absolute paths
project_dir = os.path.expanduser('~/tmp/usgs-inference')    
yolo_working_dir = os.path.expanduser('~/git/yolov5-current')
model_base = os.path.expanduser('~/models/usgs-geese')
model_file = os.path.join(model_base,
                          'usgs-geese-yolov5x6-b8-img1280-e125-of-200-20230401-ss-best.pt')

dataset_definition_file = os.path.expanduser('~/data/usgs-geese/dataset.yaml')  

assert os.path.isfile(model_file) and os.path.isdir(yolo_working_dir) and \
    os.path.isfile(dataset_definition_file)

# Derived paths
project_symlink_dir = os.path.join(project_dir,'symlink_images')
project_dataset_file_dir = os.path.join(project_dir,'dataset_files')
project_patch_dir = os.path.join(project_dir,'patches')
project_inference_script_dir = os.path.join(project_dir,'inference_scripts')
project_yolo_results_dir = os.path.join(project_dir,'yolo_results')
project_image_level_results_dir = os.path.join(project_dir,'image_level_results')
project_chunk_cache_dir = os.path.join(project_dir,'chunk_cache')
project_md_formatted_results_dir = os.path.join(project_dir,'md_formatted_results')


# Threshold used for including results in the json file during inference
default_inference_conf_thres = '0.001'

default_inference_batch_size = 8
image_size = 1280

# Right now, for debuggin, we run inference with a low confidence threshold, but after
# inference, we strip out very-low-confidence detections
post_inference_conf_thres = 0.025

n_cores_patch_generation = 16
parallelize_patch_generation_with_threads = True

force_patch_generation = False
overwrite_existing_patches = False
overwrite_md_results_files = False

devices = [0,1]

patch_jpeg_quality = 95    

default_patch_overlap = 0.1

# This isn't NMS in the usual sense of redundant model predictions; this is being
# used to de-duplicate predictions from overlapping patches.
nms_iou_threshold = 0.45

# Performing per-patch NMS is just a debugging tool, for making patch-level previews.
# There's not really a reason to  do this at the patch level when we have to do this at the
# image level anyway.
#
# The only thing we're removing when we perform NMS at the patch level is the case where nearly-identical 
# boxes are assigned to multiple classes.
do_within_patch_nms = False

# What things should we clean up at the end of the process for a folder?
cleanup_targets = ['patch_cache_file','dataset_files','symlink_images','yolo_results','inference_scripts',
                   'chunk_cache_file']
cleanup_targets.extend(['patches','patch_level_results'])
# cleanup_targets.extend(['image_level_results'])


#%% Validate class names

expected_yolo_category_id_to_name = {
    0: 'Brant',
    1: 'Other',
    2: 'Gull',
    3: 'Canada',
    4: 'Emperor'
}

def read_classes_from_yolo_dataset_file(fn):

    import re

    with open(dataset_definition_file,'r') as f:
        lines = f.readlines()
            
    to_return = {}
    pat = '\d+: .+'
    for s in lines:
        if re.search(pat,s) is not None:
            tokens = s.split(':')
            assert len(tokens) == 2
            to_return[int(tokens[0].strip())] = tokens[1].strip()
        
    return to_return

yolo_category_id_to_name = read_classes_from_yolo_dataset_file(dataset_definition_file)
assert yolo_category_id_to_name == expected_yolo_category_id_to_name

# As far as I can tell, this model does not have the class names saved, so just noting to self
# that I tried this:
#
# m = torch.load(model_file)
# print(m['model'].names)
#
#    ['0', '1', '2', '3', '4']


#%% Support functions

def get_patch_boundaries(image_size,patch_size,patch_stride=None):
    """
    Get a list of patch starting coordinates (x,y) given an image size
    and a stride.  Stride defaults to half the patch size.
    """
    
    if patch_stride is None:
        patch_stride = (round(patch_size[0]*(1.0-default_patch_overlap)),
                        round(patch_size[1]*(1.0-default_patch_overlap)))
        
    image_width = image_size[0]
    image_height = image_size[1]
        
    def add_patch_row(patch_start_positions,y_start):
        """
        Add one row to our list of patch start positions, i.e.
        loop over all columns.
        """
        
        x_start = 0; x_end = x_start + patch_size[0] - 1
        
        while(True):
            
            patch_start_positions.append([x_start,y_start])
            
            x_start += patch_stride[0]
            x_end = x_start + patch_size[0] - 1
             
            if x_end == image_width - 1:
                break
            elif x_end > (image_width - 1):
                overshoot = (x_end - image_width) + 1
                x_start -= overshoot
                x_end = x_start + patch_size[0] - 1
                patch_start_positions.append([x_start,y_start])
                break
        
        # ...for each column
        
        return patch_start_positions
        
    patch_start_positions = []
    
    y_start = 0; y_end = y_start + patch_size[1] - 1
        
    while(True):
    
        patch_start_positions = add_patch_row(patch_start_positions,y_start)
        
        y_start += patch_stride[1]
        y_end = y_start + patch_size[1] - 1
        
        if y_end == image_height - 1:
            break
        elif y_end > (image_height - 1):
            overshoot = (y_end - image_height) + 1
            y_start -= overshoot
            y_end = y_start + patch_size[1] - 1
            patch_start_positions = add_patch_row(patch_start_positions,y_start)
            break
    
    # ...for each row
    
    assert patch_start_positions[-1][0]+patch_size[0] == image_width
    assert patch_start_positions[-1][1]+patch_size[1] == image_height
    
    return patch_start_positions


def relative_path_to_image_name(rp):
    """
    Given a full path name, replace slashes and backslashes with underscores, so we can
    use the result as a filename.
    """
    
    image_name = rp.lower().replace('\\','/').replace('/','_')
    return image_name


def patch_info_to_patch_name(image_name,patch_x_min,patch_y_min):
    
    patch_name = image_name + '_' + \
        str(patch_x_min).zfill(4) + '_' + str(patch_y_min).zfill(4)
    return patch_name


def extract_patch_from_image(im,patch_xy,
                             patch_image_fn=None,patch_folder=None,image_name=None,overwrite=True):
    """
    Extracts a patch from the provided image, writing the patch out to patch_image_fn.  im
    can be a string or a PIL image.
    
    patch_xy is a length-2 tuple specifying the upper-left corner of the patch.
    
    image_name and patch_folder are only required if patch_image_fn is None.
    
    Returns a dictionary with fields xmin,xmax,ymin,ymax,patch_fn.
    """
    
    if isinstance(im,str):
        pil_im = vis_utils.open_image(im)
    else:
        pil_im = im
        
    patch_x_min = patch_xy[0]
    patch_y_min = patch_xy[1]
    patch_x_max = patch_x_min + patch_size[0] - 1
    patch_y_max = patch_y_min + patch_size[1] - 1

    # PIL represents coordinates in a way that is very hard for me to get my head
    # around, such that even though the "right" and "bottom" arguments to the crop()
    # function are inclusive... well, they're not really.
    #
    # https://pillow.readthedocs.io/en/stable/handbook/concepts.html#coordinate-system
    #
    # So we add 1 to the max values.
    patch_im = pil_im.crop((patch_x_min,patch_y_min,patch_x_max+1,patch_y_max+1))
    assert patch_im.size[0] == patch_size[0]
    assert patch_im.size[1] == patch_size[1]

    if patch_image_fn is None:
        assert patch_folder is not None,\
            "If you don't supply a patch filename to extract_patch_from_image, you need to supply a folder name"
        patch_name = patch_info_to_patch_name(image_name,patch_x_min,patch_y_min)
        patch_image_fn = os.path.join(patch_folder,patch_name + '.jpg')
    
    if os.path.isfile(patch_image_fn) and (not overwrite):
        pass
    else:        
        patch_im.save(patch_image_fn,quality=patch_jpeg_quality)
    
    patch_info = {}
    patch_info['xmin'] = patch_x_min
    patch_info['xmax'] = patch_x_max
    patch_info['ymin'] = patch_y_min
    patch_info['ymax'] = patch_y_max
    patch_info['patch_fn'] = patch_image_fn
    
    return patch_info

                             
def extract_patches_for_image(image_fn,patch_folder,image_name_base=None,overwrite=True):
    """
    Extract patches from image_fn to separate image files in patch_folder.
    
    Returns a dictionary that looks like:
        
        {
             'image_fn':'/whatever/image/you/passed/in',
             'patches':
             [
                 {
                  'xmin':x0,'ymin':y0,'xmax':x1,'ymax':y1,
                  'patch_fn':'/patch/folder/patch/image/name.jpg',
                  'image_fn':'/whatever/image/you/passed/in'
                  }
             ]
        }
            
    """
        
    os.makedirs(patch_folder,exist_ok=True)
    
    if image_name_base is None:
        image_name_base = os.path.dirname(image_fn)
        
    image_relative_path = os.path.relpath(image_fn,image_name_base)    
    image_name = relative_path_to_image_name(image_relative_path)
    
    pil_im = vis_utils.open_image(image_fn)
    assert pil_im.size[0] == expected_image_width
    assert pil_im.size[1] == expected_image_height
    
    image_width = pil_im.size[0]
    image_height = pil_im.size[1]
    image_size = (image_width,image_height)
    patch_start_positions = get_patch_boundaries(image_size,patch_size,patch_stride=None)
    
    patches = []
    
    # i_patch = 0; patch_xy = patch_start_positions[i_patch]
    for i_patch,patch_xy in enumerate(patch_start_positions):        
        patch_info = extract_patch_from_image(
            pil_im,patch_xy,
            patch_image_fn=None,patch_folder=patch_folder,
            image_name=image_name,overwrite=overwrite)
        patch_info['image_fn'] = image_fn
        patches.append(patch_info)
    
    # ...for each patch
    
    image_patch_info = {}
    image_patch_info['patches'] = patches
    image_patch_info['image_fn'] = image_fn
        
    return image_patch_info

# ...extract_patches_for_image()
    

def generate_patches_for_image(image_fn_relative,patch_folder_base,input_folder_base,overwrite=True):
    """
    Wrapper for extract_patches_for_image() that chooses a patch folder name based
    on the image name.
    
    See extract_patches_for_image for return format.
    """
    
    image_fn = os.path.join(input_folder_base,image_fn_relative)    
    assert os.path.isfile(image_fn)        
    patch_folder = os.path.join(patch_folder_base,image_fn_relative)        
    image_patch_info = extract_patches_for_image(image_fn,patch_folder,input_folder_base,overwrite=overwrite)
    return image_patch_info
    

def create_symlink_folder_for_patches(patch_files,symlink_dir):
    """
    Create a folder of symlinks pointing to patches, so we have a flat folder 
    structure that's friendly to the way YOLOv5's val.py works.  Returns a dict
    mapping patch IDs to files.
    """
    
    os.makedirs(symlink_dir,exist_ok=True)

    def safe_create_link(link_exists,link_new):
        
        if os.path.exists(link_new) or os.path.islink(link_new):
            assert os.path.islink(link_new), 'Oops, {} is a real file, not a link'.format(link_new)
            if not os.readlink(link_new) == link_exists:
                os.remove(link_new)
                os.symlink(link_exists,link_new)
        else:
            os.symlink(link_exists,link_new)
    
    patch_id_to_file = {}
    
    # i_patch = 0; patch_fn = patch_files[i_patch]
    for i_patch,patch_fn in tqdm(enumerate(patch_files),total=len(patch_files)):
        
        ext = os.path.splitext(patch_fn)[1]
        
        patch_id_string = str(i_patch).zfill(10)
        patch_id_to_file[patch_id_string] = patch_fn
        symlink_name = patch_id_string + ext
        symlink_full_path = os.path.join(symlink_dir,symlink_name)
        safe_create_link(patch_fn,symlink_full_path)
        
    # ...for each image
    
    return patch_id_to_file


def create_yolo_dataset_file(dataset_file,symlink_dir,yolo_category_id_to_name):
    """
    Create a dataset.yml file that YOLOv5's val.py can read, telling it which
    folder to run inference on.
    """
    
    category_ids = sorted(list(yolo_category_id_to_name.keys()))
    
    with open(dataset_file,'w') as f:
        f.write('path: {}\n'.format(symlink_dir))
        f.write('train: .\n')
        f.write('val: .\n')
        f.write('test: .\n')
        f.write('\n')
        f.write('nc: {}\n'.format(len(yolo_category_id_to_name)))
        f.write('\n')
        f.write('names:\n')
        for category_id in category_ids:
            assert isinstance(category_id,int)
            f.write('  {}: {}\n'.format(category_id,yolo_category_id_to_name[category_id]))
    

def run_yolo_model(project_dir,run_name,dataset_file,model_file,
                   execute=True,augment=True,device_string='0',
                   batch_size=default_inference_batch_size,
                   conf_thres=default_inference_conf_thres):
    """
    Invoke Python in a shell to run the model on an existing YOLOv5-formatted dataset.
    
    If 'execute' if false, just prepares the list of commands to run the model, but
    doesn't actually run it.
    """
    
    run_dir = os.path.join(project_dir,run_name)
    os.makedirs(run_dir,exist_ok=True)    
    
    image_size_string = str(round(image_size))
            
    cmd = 'python val.py --data "{}"'.format(dataset_file)
    cmd += ' --weights "{}"'.format(model_file)
    cmd += ' --batch-size {} --imgsz {} --conf-thres {} --task test'.format(
        batch_size,image_size_string,conf_thres)
    cmd += ' --device "{}" --save-json'.format(device_string)
    cmd += ' --project "{}" --name "{}" --exist-ok'.format(project_dir,run_name)
    
    if augment:
        cmd += ' --augment'
    
    if (execute):
        
        initial_working_dir = os.getcwd()
        os.chdir(yolo_working_dir)
        
        from ct_utils import execute_command_and_print
        cmd_result = execute_command_and_print(cmd)
        
        assert cmd_result['status'] == 0, 'Error running YOLOv5'
        
        os.chdir(initial_working_dir)
    
    return cmd

# ...run_yolo_model()


def in_place_nms(md_results, iou_thres=0.45, verbose=True):
    """
    Run torch.ops.nms in-place on MD-formatted detection results    
    """
    
    n_detections_before = 0
    n_detections_after = 0
    
    # i_image = 18; im = md_results['images'][i_image]
    for i_image,im in tqdm(enumerate(md_results['images']),total=len(md_results['images'])):
        
        if len(im['detections']) == 0:
            continue
    
        boxes = []
        scores = []
        
        n_detections_before += len(im['detections'])
        
        # det = im['detections'][0]
        for det in im['detections']:
            
            # Using x1/x2 notation rather than x0/x1 notation to be consistent
            # with the Torch documentation.
            x1 = det['bbox'][0]
            y1 = det['bbox'][1]
            x2 = det['bbox'][0] + det['bbox'][2]
            y2 = det['bbox'][1] + det['bbox'][3]
            box = [x1,y1,x2,y2]
            boxes.append(box)
            scores.append(det['conf'])

        # ...for each detection
        
        t_boxes = torch.tensor(boxes)
        t_scores = torch.tensor(scores)
        
        box_indices = ops.nms(t_boxes,t_scores,iou_thres).tolist()
        
        post_nms_detections = [im['detections'][x] for x in box_indices]
        
        assert len(post_nms_detections) <= len(im['detections'])
        
        im['detections'] = post_nms_detections
        
        n_detections_after += len(im['detections'])
        
    # ...for each image
    
    if verbose:
        print('NMS removed {} of {} detections'.format(
            n_detections_before-n_detections_after,
            n_detections_before))
        
# ...in_place_nms()


#%% The main function: run the model recursively on  folder

def run_model_on_folder(input_folder_base,recursive=True):
    """
    Run the goose detection model on all images in a folder
    """
    
    ##%% Input validation
    
    assert os.path.isdir(input_folder_base), \
        'Could not find input folder {}'.format(input_folder_base)
    
    folder_name_clean = input_folder_base.replace('\\','/').replace('/','_').replace(' ','_')
    if folder_name_clean.startswith('_'):
        folder_name_clean = folder_name_clean[1:]
    
    
    ##%% Enumerate images
    
    images_absolute = path_utils.find_images(input_folder_base,recursive=recursive)
    images_relative = [os.path.relpath(fn,input_folder_base) for fn in images_absolute]    
    

    ##%% Generate patches
    
    os.makedirs(project_chunk_cache_dir,exist_ok=True)

    # This is a .json file that includes metadata about our patches; this is only used during
    # debugging, when we want to re-start from this point but don't want to re-generate patches
    patch_cache_file = os.path.join(project_chunk_cache_dir,folder_name_clean + '_patch_info.json')
    patch_folder_for_folder = os.path.join(project_patch_dir,folder_name_clean)
                                           
    if force_patch_generation or (not os.path.isfile(patch_cache_file)):
        
        print('Generating patches for {}'.format(input_folder_base))
            
        if n_cores_patch_generation == 1:
            all_image_patch_info = []
            # image_fn_relative = images_relative[0]
            for image_fn_relative in tqdm(images_relative):
                image_patch_info = generate_patches_for_image(image_fn_relative,patch_folder_for_folder,
                                                              input_folder_base)
                all_image_patch_info.append(image_patch_info)
        else:                
            if parallelize_patch_generation_with_threads:
                pool = ThreadPool(n_cores_patch_generation)
                print('Generating patches on a pool of {} threads'.format(n_cores_patch_generation))
            else:
                pool = Pool(n_cores_patch_generation)
                print('Generating patches on a pool of {} processes'.format(n_cores_patch_generation))
    
            all_image_patch_info = list(tqdm(pool.imap(
                partial(generate_patches_for_image,
                        patch_folder_base=patch_folder_for_folder,
                        input_folder_base=input_folder_base,
                        overwrite=overwrite_existing_patches), 
                images_relative), total=len(images_relative)))
            
        all_patch_files = []
        for image_patch_info in all_image_patch_info:
            image_patch_files = [pi['patch_fn'] for pi in image_patch_info['patches']]
            all_patch_files.extend(image_patch_files)
            
        total_patch_size_bytes = 0
        for fn in tqdm(all_patch_files):
            total_patch_size_bytes += os.path.getsize(fn)
        total_patch_size_str = humanfriendly.format_size(total_patch_size_bytes)
        
        print('Generated {} patches for {} images in folder {}, taking up {}'.format(
            len(all_patch_files),len(all_image_patch_info),
            input_folder_base,total_patch_size_str))

        with open(patch_cache_file,'w') as f:
            json.dump(all_image_patch_info,f,indent=1)
            
        print('Wrote patch info to {}'.format(patch_cache_file))        
        
        del image_patch_info
        del all_patch_files
    
    else:
        
        print('Loading cached patch information from {}'.format(patch_cache_file))
        
        with open(patch_cache_file,'r') as f:
            all_image_patch_info = json.load(f)
    
    # See extract_patches_for_image for the format of all_image_patch_info
    all_patch_files = []
    for image_patch_info in all_image_patch_info:
        for patch_info in image_patch_info['patches']:
            all_patch_files.append(patch_info['patch_fn'])
            
    # Double-check that we have the right number of patches (n_images * n_patches_per_image)
    n_patches_per_image = len(get_patch_boundaries(
        (expected_image_width,expected_image_height),
        patch_size,patch_stride=None))
    assert len(all_patch_files) == n_patches_per_image * len(images_relative)
    
    
    ##%% Split patches into chunks (one per GPU), and generate symlink folder(s)
    
    def split_list(L, n):
        k, m = divmod(len(L), n)
        return list(L[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
            
    # Split patches into chunks
    n_chunks = len(devices)
    patch_chunks = split_list(all_patch_files,n_chunks)
    assert sum([len(p) for p in patch_chunks]) == len(all_patch_files)
    
    chunk_info = []
    
    folder_symlink_dir = os.path.join(project_symlink_dir,folder_name_clean)
                                      
    for i_chunk,chunk_files in enumerate(patch_chunks):
    
        chunk_symlink_dir = os.path.join(folder_symlink_dir,'chunk_{}'.format(str(i_chunk).zfill(2)))
    
        print('Generating symlinks for chunk {} in folder {}'.format(
            i_chunk,chunk_symlink_dir))
    
        chunk_patch_id_to_file = create_symlink_folder_for_patches(chunk_files,chunk_symlink_dir)
        chunk_symlinks = os.listdir(chunk_symlink_dir)
        assert len(chunk_symlinks) == len(chunk_patch_id_to_file)
        
        chunk = {'chunk_id':'chunk_{}'.format(str(i_chunk).zfill(2)),
                           'symlink_dir':chunk_symlink_dir,
                           'patch_id_to_file':chunk_patch_id_to_file}
        
        chunk_info.append(chunk)

    # ...for each chunk

    
    ##%% Generate .yaml files (to tell YOLO where the data is)
    
    folder_dataset_file_dir = os.path.join(project_dataset_file_dir,folder_name_clean)
    os.makedirs(folder_dataset_file_dir,exist_ok=True)
    
    for i_chunk,chunk in enumerate(chunk_info):
                
        chunk['dataset_file'] = os.path.join(folder_dataset_file_dir,chunk['chunk_id'] + '_dataset.yaml')
        print('Writing dataset file for chunk {} to {}'.format(i_chunk,chunk['dataset_file']))
        create_yolo_dataset_file(chunk['dataset_file'],chunk['symlink_dir'],yolo_category_id_to_name)
    
        
    ##%% Prepare commands to run the model on symlink folder(s)
    
    folder_inference_script_dir = os.path.join(project_inference_script_dir,folder_name_clean)
    os.makedirs(folder_inference_script_dir,exist_ok=True)
    
    folder_yolo_results_dir = os.path.join(project_yolo_results_dir,folder_name_clean)
    os.makedirs(folder_yolo_results_dir,exist_ok=True)
    
    for i_chunk,chunk in enumerate(chunk_info):
        
        device_string = devices[i_chunk]
        
        chunk['run_name'] = 'inference-output-' + chunk['chunk_id']        
        chunk['run_output_dir'] = os.path.join(folder_yolo_results_dir,chunk['run_name'])
        chunk['cmd'] = run_yolo_model(folder_yolo_results_dir,chunk['run_name'],chunk['dataset_file'],
                                      model_file,execute=False,
                                      device_string=device_string)
        chunk['script_name'] = os.path.join(folder_inference_script_dir,'run_chunk_{}_device_{}.sh'.format(
            str(i_chunk).zfill(2),device_string))
        with open(chunk['script_name'],'w') as f:
            f.write(chunk['cmd'])
        st = os.stat(chunk['script_name'])
        os.chmod(chunk['script_name'], st.st_mode | stat.S_IEXEC)
        print('Wrote chunk {} script to {}'.format(i_chunk,chunk['script_name']))
        
    # ...for each chunk
    
    
    ##%% Save/load chunk state for debugging (because stuff crashes)
    
    chunk_cache_file = os.path.join(project_chunk_cache_dir,folder_name_clean + '_chunk_info.json')    
    os.makedirs(project_chunk_cache_dir,exist_ok=True)
    
    if False:
        
        # Save state
        with open(chunk_cache_file,'w') as f:
            json.dump(chunk_info,f,indent=1)
    
    if False:

        # Load state
        with open(chunk_cache_file,'r') as f:
            chunk_info = json.load(f)
    

    ##%% Run inference
    
    # Changes the current working directory, making no attempt to change it back.          
    execute_inline = True    
    
    print('Inference commands:\n')
    for chunk in chunk_info:
        print('{}'.format(chunk['script_name']))
    print('')
    
    if (execute_inline):
        
        chunk_commands = [chunk['script_name'] for chunk in chunk_info]
        n_workers = len(devices)
       
        # Should we use threads (vs. processes) for parallelization?
        use_threads = True
       
        def run_chunk(cmd):
            os.environ['LD_LIBRARY_PATH']=''
            os.chdir(yolo_working_dir)
            return process_utils.execute_and_print(cmd,print_output=True)
           
        if n_workers == 1:  
         
          results = []
          for i_command,command in enumerate(chunk_commands):    
            results.append(run_chunk(command))
         
        else:
         
          if use_threads:
            print('Starting parallel thread pool with {} workers'.format(n_workers))
            pool = ThreadPool(n_workers)
          else:
            print('Starting parallel process pool with {} workers'.format(n_workers))
            pool = Pool(n_workers)
       
          results = list(pool.map(run_chunk,chunk_commands))
          
          assert all([r['status'] == 0 for r in results]), 'Error running one or more inference processes'
          
    else:
    
        print('Bypassing inline execution')
        
        
    ##%% Read and convert patch results for each chunk
    
    # We're reading patch results just to validate that they were written sensibly; we don't use the loaded
    # results directly.  We'll convert them to MD format and use that version.
    
    model_short_name = os.path.basename(model_file).replace('.pt','')    
    
    # i_chunk = 0; chunk = chunk_info[i_chunk]
    for i_chunk,chunk in enumerate(chunk_info):
        
        run_dir = chunk['run_output_dir']
        assert os.path.isdir(run_dir)
        
        json_files = glob.glob(run_dir + '/*.json')
        json_files = [fn for fn in json_files if 'md_format' not in fn]
        assert len(json_files) == 1
        
        yolo_json_file = json_files[0]
        chunk['yolo_json_file'] = yolo_json_file
        
        md_formatted_results_file = yolo_json_file.replace('.json','-md_format.json')
        chunk['md_formatted_results_file'] = md_formatted_results_file
        assert md_formatted_results_file != yolo_json_file
            
        if os.path.isfile(md_formatted_results_file) and (not overwrite_md_results_files):
            
            print('Bypassing YOLO --> MD conversion for {}, output file exists'.format(
                yolo_json_file))
            
        else:
            
            print('Reading results from {}'.format(yolo_json_file))
            
            with open(yolo_json_file,'r') as f:
                yolo_results = json.load(f)
            
            print('Read {} results for {} patches'.format(
                len(yolo_results),len(chunk['patch_id_to_file'])))
    
            # Convert patch results to MD output format
                
            patch_id_to_relative_path = {}
            
            # i_patch = 0; patch_id = next(iter(chunk['patch_id_to_file'].keys()))
            for patch_id in chunk['patch_id_to_file'].keys():
                fn = chunk['patch_id_to_file'][patch_id]
                assert patch_folder_for_folder in fn
                relative_fn = os.path.relpath(fn,patch_folder_for_folder)
                patch_id_to_relative_path[patch_id] = relative_fn
                        
            yolo_json_output_to_md_output(yolo_json_file,
                                          image_folder=patch_folder_for_folder,
                                          output_file=md_formatted_results_file,
                                          yolo_category_id_to_name=yolo_category_id_to_name,
                                          detector_name=model_short_name,
                                          image_id_to_relative_path=patch_id_to_relative_path,
                                          offset_yolo_class_ids=False)
    
        del md_formatted_results_file,run_dir
        
    # ...for each chunk
    
    
    ##%% Merge results files from each chunk into one (patch-level) results file for the folder
    
    os.makedirs(project_md_formatted_results_dir,exist_ok=True)
    md_formatted_results_files_for_chunks = [chunk['md_formatted_results_file'] for chunk in chunk_info]
    md_formatted_results_file_for_folder = os.path.join(project_md_formatted_results_dir,
                                    folder_name_clean + '.json')    
    
    _ = combine_api_outputs.combine_api_output_files(md_formatted_results_files_for_chunks,
                                                 md_formatted_results_file_for_folder,
                                                 require_uniqueness=True)
    assert os.path.isfile(md_formatted_results_file_for_folder)
    
    
    ##%% Remove low-confidence detections
    
    md_formatted_results_file_for_folder_thresholded = md_formatted_results_file_for_folder.replace(
        '.json','_threshold_{}.json'.format(post_inference_conf_thres))

    
    with open(md_formatted_results_file_for_folder,'r') as f:
        d_before_thresholding = json.load(f)
    
    n_detections_before_thresholding = 0    
    for im in d_before_thresholding['images']:
        n_detections_before_thresholding += len(im['detections'])

    from api.batch_processing.postprocessing.subset_json_detector_output import (
        subset_json_detector_output, SubsetJsonDetectorOutputOptions)

    options = SubsetJsonDetectorOutputOptions()
    options.confidence_threshold = post_inference_conf_thres
    options.overwrite_json_files = True
    
    d_after_thresholding = subset_json_detector_output(md_formatted_results_file_for_folder, 
                                                       md_formatted_results_file_for_folder_thresholded, 
                                                       options, d_before_thresholding)
    
    n_detections_after_thresholding = 0    
    for im in d_after_thresholding['images']:
        n_detections_after_thresholding += len(im['detections'])
      
    print('Thresholding reduced the total number of detections from {} to {}'.format(
        n_detections_before_thresholding,n_detections_after_thresholding))
    
    del d_before_thresholding,d_after_thresholding

    
    ##%% Optionallly perform NMS within each patch
    
    if do_within_patch_nms:
        
        print('Loading merged results file')
        
        with open(md_formatted_results_file_for_folder_thresholded,'r') as f:        
            md_results = json.load(f)
        
        print('Eliminating redundant detections')
        
        in_place_nms(md_results,iou_thres=nms_iou_threshold)
        
        patch_results_after_nms_file = md_formatted_results_file_for_folder_thresholded.replace('.json',
                                                                      '_patch-level_nms.json')
        assert patch_results_after_nms_file != md_formatted_results_file_for_folder_thresholded
        
        with open(patch_results_after_nms_file,'w') as f:
            json.dump(md_results,f,indent=1)

    else:
        
        patch_results_after_nms_file = None
        
    
    ##%% Combine all the patch results to an image-level results set
        
    patch_results_file = md_formatted_results_file_for_folder_thresholded
    
    with open(patch_results_file,'r') as f:
        all_patch_results = json.load(f)
    
    # Map absolute paths to detections; we need this because we used absolute paths
    # to map patches back to images.
    #
    # This contains patches for all images in the folder.
    patch_fn_to_results = {}
    for im in tqdm(all_patch_results['images']):
        abs_fn = os.path.join(patch_folder_for_folder,im['file'])
        patch_fn_to_results[abs_fn] = im

    md_results_image_level = {}
    md_results_image_level['info'] = all_patch_results['info']
    md_results_image_level['detection_categories'] = all_patch_results['detection_categories']
    md_results_image_level['images'] = []
    
    image_fn_to_patch_info = { x['image_fn']:x for x in all_image_patch_info }
    
    # i_image = 0; image_fn_relative = images_relative[i_image]
    for i_image,image_fn_relative in tqdm(enumerate(images_relative),total=len(images_relative)):
        
        image_fn = os.path.join(input_folder_base,image_fn_relative)
        assert os.path.isfile(image_fn)
                
        output_im = {}
        output_im['file'] = image_fn_relative
        output_im['detections'] = []
            
        pil_im = vis_utils.open_image(image_fn)
        assert pil_im.size[0] == expected_image_width
        assert pil_im.size[1] == expected_image_height
        
        image_w = pil_im.size[0]
        image_h = pil_im.size[1]
        
        image_patch_info = image_fn_to_patch_info[image_fn]
        assert image_patch_info['patches'][0]['image_fn'] == image_fn
        
        # Patches just for this image
        patch_fn_to_patch_info_this_image = {}
        
        for patch_info in image_patch_info['patches']:
            patch_fn_to_patch_info_this_image[patch_info['patch_fn']] = patch_info
                
        # For each patch
        # i_patch = 0; patch_fn = list(patch_fn_to_patch_info_this_image.keys())[i_patch]
        for i_patch,patch_fn in enumerate(patch_fn_to_patch_info_this_image.keys()):
            
            patch_results = patch_fn_to_results[patch_fn]
            patch_info = patch_fn_to_patch_info_this_image[patch_fn]
            
            # patch_results['file'] is a relative path, and a subset of patch_info['patch_fn']
            assert patch_results['file'] in patch_info['patch_fn']
            
            patch_w = (patch_info['xmax'] - patch_info['xmin']) + 1
            patch_h = (patch_info['ymax'] - patch_info['ymin']) + 1
            assert patch_w == patch_size[0]
            assert patch_h == patch_size[1]
            
            # det = patch_results['detections'][0]
            for det in patch_results['detections']:
            
                bbox_patch_relative = det['bbox']
                xmin_patch_relative = bbox_patch_relative[0]
                ymin_patch_relative = bbox_patch_relative[1]
                w_patch_relative = bbox_patch_relative[2]
                h_patch_relative = bbox_patch_relative[3]
                
                # Convert from patch-relative normalized values to image-relative absolute values
                w_pixels = w_patch_relative * patch_w
                h_pixels = h_patch_relative * patch_h
                xmin_patch_pixels = xmin_patch_relative * patch_w
                ymin_patch_pixels = ymin_patch_relative * patch_h
                xmin_image_pixels = patch_info['xmin'] + xmin_patch_pixels
                ymin_image_pixels = patch_info['ymin'] + ymin_patch_pixels
                
                # ...and now to image-relative normalized values
                w_image_normalized = w_pixels / image_w
                h_image_normalized = h_pixels / image_h
                xmin_image_normalized = xmin_image_pixels / image_w
                ymin_image_normalized = ymin_image_pixels / image_h
                
                bbox_image_normalized = [xmin_image_normalized,
                                         ymin_image_normalized,
                                         w_image_normalized,
                                         h_image_normalized]
                
                output_det = {}
                output_det['bbox'] = bbox_image_normalized
                output_det['conf'] = det['conf']
                output_det['category'] = det['category']
                
                output_im['detections'].append(output_det)
                
            # ...for each detection
            
        # ...for each patch

        md_results_image_level['images'].append(output_im)
        
    # ...for each image    
    
    os.makedirs(project_image_level_results_dir,exist_ok=True)
    
    md_results_image_level_fn = os.path.join(project_image_level_results_dir,
                                             folder_name_clean + '_md_results_image_level.json')
    print('Saving image-level results to {}'.format(md_results_image_level_fn))
          
    with open(md_results_image_level_fn,'w') as f:
        json.dump(md_results_image_level,f,indent=1)


    ##%% Perform image-level NMS
    
    in_place_nms(md_results_image_level,iou_thres=nms_iou_threshold)
    
    md_results_image_level_nms_fn = md_results_image_level_fn.replace('.json','_nms.json')
    
    print('Saving image-level results (after NMS) to {}'.format(md_results_image_level_nms_fn))
    
    with open(md_results_image_level_nms_fn,'w') as f:
        json.dump(md_results_image_level,f,indent=1)


    ##%% Clean up

    """
    For all the things we're supposed to be cleaning up, before we delete a bunch of stuff
    we worked hard to generate, make sure the files we're deleting look like what we expect.    
    """
    
    execute_cleanup = True
    
    def safe_delete(fn,verbose=True):
        
        if fn is None or len(fn) == 0:
            return
        
        try:
            if os.path.isfile(fn):
                if verbose:
                    print('Cleaning up file {}'.format(fn))
                if execute_cleanup:
                    os.remove(fn)
            elif os.path.isdir(fn):
                if verbose:
                    print('Cleaning up folder {}'.format(fn))
                if execute_cleanup:
                    shutil.rmtree(fn)
                    pass
            else:
                print('Skipping cleanup of {}, does not exist'.format(fn))
        except Exception as e:
            print('Error cleaning up {}: {}'.format(fn,str(e)))
                
    if 'patch_cache_file' in cleanup_targets:
        safe_delete(patch_cache_file)
    else:
        print('Bypassing cleanup of patch cache file')
    
    if 'chunk_cache_file' in cleanup_targets:
        safe_delete(chunk_cache_file)
    else:
        print('Bypassing cleanup of chunk cache file')
        
    if 'dataset_files' in cleanup_targets:
        if os.path.isdir(folder_dataset_file_dir):
            dataset_files = os.listdir(folder_dataset_file_dir)
            assert all([fn.endswith('.yaml') for fn in dataset_files])
            safe_delete(folder_dataset_file_dir)        
    else:
        print('Bypassing cleanup of dataset files')
        
    if 'patch_level_results' in cleanup_targets:
        safe_delete(md_formatted_results_file_for_folder)
        safe_delete(md_formatted_results_file_for_folder_thresholded)
        safe_delete(patch_results_after_nms_file)
    else:
        print('Bypassing cleanup of patch-level results')
        
    if 'inference_scripts' in cleanup_targets:
        if os.path.isdir(folder_inference_script_dir):
            inference_scripts = os.listdir(folder_inference_script_dir)
            assert all([fn.endswith('.sh') for fn in inference_scripts])
            safe_delete(folder_inference_script_dir)
    else:
        print('Bypassing cleanup of inference scripts')
    
    if 'patches' in cleanup_targets:
        if os.path.isdir(patch_folder_for_folder):
            
            # TODO: leaf-node folders should end in JPG, and all files should be .jpg
            if False:
                patches = os.listdir(patch_folder_for_folder)
                patches = [os.path.join(patch_folder_for_folder,fn) for fn in patches]
                
                # Each of these a folder with a .JPG extension, so both of the following should be true
                assert all([os.path.isdir(fn) for fn in patches])
                assert all([path_utils.is_image_file(fn) for fn in patches])        
            safe_delete(patch_folder_for_folder)
    else:
        print('Bypassing cleanup of patches')
        
    if 'symlink_images' in cleanup_targets:
        if os.path.isdir(folder_symlink_dir):
            # These are either folders called "chunk_00" or yolov5 cache files called "chunk_00.cache"
            symlink_folders_and_cache_files = os.listdir(folder_symlink_dir)
            assert all([fn.startswith('chunk') for fn in symlink_folders_and_cache_files])
            safe_delete(folder_symlink_dir)
    else:
        print('Bypassing cleanup of symlink folder')
        
    if 'yolo_results' in cleanup_targets:
        if os.path.isdir(folder_yolo_results_dir):
            yolo_results_folders = os.listdir(folder_yolo_results_dir)
            assert all([os.path.isdir(os.path.join(folder_yolo_results_dir,fn)) for fn in yolo_results_folders])
            assert all([fn.startswith('inference-output') for fn in yolo_results_folders])
            safe_delete(folder_yolo_results_dir)
    else:
        print('Bypassing cleanup of YOLO-formatted results')              
    
    # Reserving this for future use, but right now it would be silly to delete this
    if 'image_level_results' in cleanup_targets:
        safe_delete(md_results_image_level_fn)
    else:
        print('Bypassing cleanup of image-level results')
    
    
    ##%% Prepare return values
    
    to_return = {}
    to_return['md_formatted_results_file_for_folder_thresholded'] = \
        md_formatted_results_file_for_folder_thresholded
    to_return['md_results_image_level_fn'] = \
        md_results_image_level_fn
    to_return['md_results_image_level_nms_fn'] = \
        md_results_image_level_nms_fn
    
    ##%%
    
    return to_return
    
# ...run_model_on_folder()


#%% Interactive driver

if False:
    
    pass

    #%%
    
    # input_folder_base = '/media/user/My Passport/2017-2019/01_JPGs/2017/Replicate_2017-10-01/Cam1'
    # input_folder_base = '/media/user/My Passport/2022-10-09/cam3'
    # input_folder_base = '/home/user/data/usgs-test-folder'
    
    # input_folder_base = '/media/user/My Passport/2022-10-11'
    # input_folder_base = '/media/user/My Passport/2022-10-09'
    # input_folder_base = '/media/user/My Passport/2022-10-12'
    # input_folder_base = '/media/user/My Passport/2022-10-16'
    input_folder_base = '/media/user/My Passport/2022-10-17'
    
    # input_folder_base = '/home/user/data/usgs-geese/eval_images'
    
    results = run_model_on_folder(input_folder_base,recursive=True)
    

#%% Scrap

if False:

    pass
    
    #%% Time estimates
    
    # Time to process all patches for an image on a single GPU
    seconds_per_image = 25
    n_workers = 2
    seconds_per_image /= n_workers
    
    drive_base = '/media/user/My Passport'
    
    estimate_time_for_old_data = False
    
    if estimate_time_for_old_data:
        base_folder = os.path.join(drive_base,'2017-2019')
        image_folder = os.path.join(base_folder,'01_JPGs')
        image_folder_name = image_folder
        images_absolute = path_utils.find_images(image_folder,recursive=True)
    else:
        images_absolute = []
        image_folder_name = '2022 images'
        root_filenames = os.listdir(drive_base)
        for fn in root_filenames:
            if fn.startswith('2022'):
                dirname = os.path.join(drive_base,fn)
                if os.path.isdir(dirname):
                    images_absolute.extend(path_utils.find_images(dirname,recursive=True))        
    
    total_time_seconds = seconds_per_image * len(images_absolute)
    
    print('Expected time for {} ({} images): {}'.format(
        image_folder_name,len(images_absolute),humanfriendly.format_timespan(total_time_seconds)))
    
    
    #%% Unused variable suppression
    
    patch_results_after_nms_file = None
    patch_folder_for_folder = None
    
    
    #%% Preview results for patches at a variety of confidence thresholds
    
    patch_results_file = patch_results_after_nms_file
            
    from api.batch_processing.postprocessing.postprocess_batch_results import (
        PostProcessingOptions, process_batch_results)
    
    postprocessing_output_folder = os.path.join(project_dir,'preview')

    base_task_name = os.path.basename(patch_results_file)
        
    for confidence_threshold in [0.4,0.5,0.6,0.7,0.8]:
        
        options = PostProcessingOptions()
        options.image_base_dir = patch_folder_for_folder
        options.include_almost_detections = True
        options.num_images_to_sample = 7500
        options.confidence_threshold = confidence_threshold
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
        
        options.api_output_file = patch_results_file
        options.output_dir = output_base
        ppresults = process_batch_results(options)
        html_output_file = ppresults.output_html_file
        
        path_utils.open_file(html_output_file)
    

    #%% Render boxes on one of the original images
    
    input_folder_base = '/media/user/My Passport/2022-10-09/cam3'
    md_results_image_level_nms_fn = os.path.expanduser(
        '~/tmp/usgs-inference/image_level_results/'+\
        'media_user_My_Passport_2022-10-09_cam3_md_results_image_level_nms.json')
    
    with open(md_results_image_level_nms_fn,'r') as f:
        md_results_image_level = json.load(f)

    i_image = 0
    output_image_file = os.path.join(project_dir,'test.jpg')
    detections = md_results_image_level['images'][i_image]['detections']    
    image_fn_relative = md_results_image_level['images'][i_image]['file']
    image_fn = os.path.join(input_folder_base,image_fn_relative)
    assert os.path.isfile(image_fn)
    
    detector_label_map = {}
    for category_id in yolo_category_id_to_name:
        detector_label_map[str(category_id)] = yolo_category_id_to_name[category_id]
        
    vis_utils.draw_bounding_boxes_on_file(input_file=image_fn,
                          output_file=output_image_file,
                          detections=detections,
                          confidence_threshold=0.4,
                          detector_label_map=detector_label_map, 
                          thickness=1, 
                          expansion=0)
    
    path_utils.open_file(output_image_file)
    
