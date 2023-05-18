#
# usgs-geese-inference.py
#
# Run inference on a folder of images, by breaking each image into overlapping 
# 1280x1280 patches, running the model on each patch, and eliminating redundant
# boxes from the results.
#
# TODO:
#
# * Finish folder inference
# * Run more chunks than devices, to support checkpointing
# * Add timestamps to the folder name
#

#%% Constants and imports

import os
import stat
import json
import glob
import humanfriendly

from multiprocessing.pool import ThreadPool
from multiprocessing.pool import Pool
from functools import partial
from tqdm import tqdm

import torch
from torchvision import ops

import path_utils
from visualization import visualization_utils as visutils
from data_management.yolo_output_to_md_output import yolo_json_output_to_md_output

# We will explicitly verify that images are actually this size
expected_image_width = 8688
expected_image_height = 5792

patch_size = (1280,1280)

project_dir = os.path.expanduser('~/tmp/usgs-inference')    
project_symlink_dir = os.path.join(project_dir,'symlink_images')
patch_folder_base = os.path.join(project_dir,'patches')
chunk_cache_dir = os.path.join(project_dir,'chunk_cache')
md_formatted_results_dir = os.path.join(project_dir,'md_formatted_results')

working_dir = os.path.expanduser('~/git/yolov5-current')
model_base = os.path.expanduser('~/models/usgs-geese')
model_file = os.path.join(model_base,
                          'usgs-geese-yolov5x6-b8-img1280-e125-of-200-20230401-ss-best.pt')
dataset_definition_file = os.path.expanduser('~/data/usgs-geese/dataset.yaml')  
assert os.path.isfile(model_file)

default_batch_size = 8
default_conf_thres = '0.001'
image_size = 1280

n_cores_patch_generation = 16
parallelize_patch_generation_with_threads = True

overwrite_existing_patches = False
overwrite_md_results_files = False

devices = [0,1]

os.chdir(working_dir)


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
        patch_stride = (round(patch_size[0]/2),round(patch_size[1]/2))
        
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
    
    image_name = rp.lower().replace('\\','/').replace('/','_')
    return image_name


def patch_info_to_patch_name(image_name,patch_x_min,patch_y_min):
    
    patch_name = image_name + '_' + \
        str(patch_x_min).zfill(4) + '_' + str(patch_y_min).zfill(4)
    return patch_name


def extract_patches_for_image(image_fn,patch_folder,image_name_base=None,overwrite=True):
    """
    Extract patches from image_fn to separate image files in patch_folder.
    Returns a list of patch information structs.
    """
    
    patch_jpeg_quality = 95    
    patch_stride = None
    
    os.makedirs(patch_folder,exist_ok=True)
    
    if image_name_base is None:
        image_name_base = os.path.dirname(image_fn)
        
    image_relative_path = os.path.relpath(image_fn,image_name_base)    
    image_name = relative_path_to_image_name(image_relative_path)
    
    pil_im = visutils.open_image(image_fn)
    assert pil_im.size[0] == expected_image_width
    assert pil_im.size[1] == expected_image_height
    
    image_width = pil_im.size[0]
    image_height = pil_im.size[1]
    image_size = (image_width,image_height)
    patch_start_positions = get_patch_boundaries(image_size,patch_size,patch_stride)
    
    patches = []
    
    # i_patch = 0; patch_xy = patch_start_positions[i_patch]
    for i_patch,patch_xy in enumerate(patch_start_positions):
        
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

        patch_name = patch_info_to_patch_name(image_name,patch_x_min,patch_y_min)
        patch_image_fn = os.path.join(patch_folder,patch_name + '.jpg')
        
        if os.path.isfile(patch_image_fn) and (not overwrite):
            # print('Skipping image write to {}'.format(patch_image_fn))
            pass
        else:        
            patch_im.save(patch_image_fn,quality=patch_jpeg_quality)
        
        patch_info = {}
        patch_info['xmin'] = patch_x_min
        patch_info['xmax'] = patch_x_max
        patch_info['ymin'] = patch_y_min
        patch_info['ymax'] = patch_y_max
        patch_info['patch_fn'] = patch_image_fn
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
        
        if os.path.exists(link_new):
            assert os.path.islink(link_new)
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
                   batch_size=default_batch_size,
                   conf_thres=default_conf_thres):
    """
    Invoke Python in a shell to run the model on a folder of images.
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
        os.chdir(working_dir)
        
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
    
    # i_image = 5; im = md_results['images'][i_image]
    # i_image = 1; im = md_results['images'][i_image]
    for i_image,im in enumerate(md_results['images']):
        
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

    
#%%

# input_folder_base = '/media/user/My Passport/2017-2019/01_JPGs/2017/Replicate_2017-10-01/Cam1'
def run_model_on_folder(input_folder_base):
    
    #%% Input validation
    
    assert os.path.isdir(input_folder_base), \
        'Could not find input folder {}'.format(input_folder_base)
    
    
    #%% Generate patches
    
    images_absolute = path_utils.find_images(input_folder_base,recursive=True)
    images_relative = [os.path.relpath(fn,input_folder_base) for fn in images_absolute]    
    
    print('Generating patches')
        
    if n_cores_patch_generation == 1:
        all_image_patch_info = []
        for image_fn_relative in tqdm(images_relative):
            image_patch_info = generate_patches_for_image(image_fn_relative,patch_folder_base,
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
                    patch_folder_base=patch_folder_base,
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

    
    #%% Generate symlink folder(s)
    
    def split_list(L, n):
        k, m = divmod(len(L), n)
        return list(L[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
            
    # Split patches into chunks
    n_chunks = len(devices)
    patch_chunks = split_list(all_patch_files,n_chunks)
    assert sum([len(p) for p in patch_chunks]) == len(all_patch_files)
    
    chunk_info = []
    
    for i_chunk,chunk_files in enumerate(patch_chunks):
    
        chunk_symlink_dir = os.path.join(project_symlink_dir,'chunk_{}'.format(str(i_chunk).zfill(2)))
    
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

    
    #%% Generate .yaml file (to tell YOLO where the data is)
    
    for i_chunk,chunk in enumerate(chunk_info):
                
        chunk['dataset_file'] = os.path.join(project_dir,chunk['chunk_id'] + '_dataset.yaml')
        print('Writing dataset file for chunk {} to {}'.format(i_chunk,chunk['dataset_file']))
        create_yolo_dataset_file(chunk['dataset_file'],chunk['symlink_dir'],yolo_category_id_to_name)
    
        
    #%% Prepare commands to run the model on symlink folder(s)
    
    for i_chunk,chunk in enumerate(chunk_info):
        
        device_string = devices[i_chunk]
        
        chunk['run_name'] = 'inference-output-' + chunk['chunk_id']        
        chunk['cmd'] = run_yolo_model(project_dir,chunk['run_name'],chunk['dataset_file'],
                                      model_file,execute=False,
                                      device_string=device_string)
        chunk['script_name'] = os.path.join(project_dir,'run_chunk_{}_device_{}.sh'.format(
            str(i_chunk).zfill(2),device_string))
        with open(chunk['script_name'],'w') as f:
            f.write(chunk['cmd'])
        st = os.stat(chunk['script_name'])
        os.chmod(chunk['script_name'], st.st_mode | stat.S_IEXEC)
        print('Wrote chunk {} script to {}'.format(i_chunk,chunk['script_name']))
        
    # ...for each chunk
    
    
    #%% Save/load chunk state for debugging (because stuff crashes)
    
    os.makedirs(chunk_cache_dir,exist_ok=True)
    chunk_cache_file = os.path.join(chunk_cache_dir,
                                    input_folder_base.replace('\\','/').replace('/','_') + '.json')
    
    
    if False:
        
        # Save state
        with open(chunk_cache_file,'w') as f:
            json.dump(chunk_info,f,indent=1)
    
    if False:

        # Load state
        with open(chunk_cache_file,'r') as f:
            chunk_info = json.load(f)
    

    #%% Run the scripts
    
    # TODO, currently done outside of this scripts, because I was too lazy to parallelize the invocation here    
    
    
    #%% Read and convert patch results for each chunk
    
    # We're reading patch results just to validate that they were written sensibly; we don't use the loaded
    # results directly.  We'll convert them to MD format and use that version.
    
    model_short_name = os.path.basename(model_file).replace('.pt','')    
    
    # i_chunk = 0; chunk = chunk_info[i_chunk]
    for i_chunk,chunk in enumerate(chunk_info):
        
        run_dir = os.path.join(project_dir,chunk['run_name'])
        
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
            for patch_id in chunk['patch_id_to_file'].keys():
                fn = chunk['patch_id_to_file'][patch_id]
                relative_fn = os.path.relpath(fn,patch_folder_base)
                patch_id_to_relative_path[patch_id] = relative_fn
                        
            yolo_json_output_to_md_output(yolo_json_file,
                                          image_folder=patch_folder_base,
                                          output_file=md_formatted_results_file,
                                          yolo_category_id_to_name=yolo_category_id_to_name,
                                          detector_name=model_short_name,
                                          image_id_to_relative_path=patch_id_to_relative_path,
                                          offset_yolo_class_ids=False)
    
        del md_formatted_results_file,run_dir
        
    # ...for each chunk
    
    
    #%% Merge results files
    
    os.makedirs(md_formatted_results_dir,exist_ok=True)
    md_formatted_results_files_for_chunks = [chunk['md_formatted_results_file'] for chunk in chunk_info]
    md_formatted_results_file_for_folder = os.path.join(md_formatted_results_dir,
                                    input_folder_base.replace('\\','/').replace('/','_') + '.json')
    
    from api.batch_processing.postprocessing import combine_api_outputs
    
    combine_api_outputs.combine_api_output_files(md_formatted_results_files_for_chunks,
                                                 md_formatted_results_file_for_folder,
                                                 require_uniqueness=True)
    
    
    #%% Preview results for patches
    
    patch_results_file = md_formatted_results_file_for_folder
            
    from api.batch_processing.postprocessing.postprocess_batch_results import (
        PostProcessingOptions, process_batch_results)
    
    postprocessing_output_folder = os.path.join(project_dir,'preview')

    base_task_name = os.path.basename(patch_results_file)
    
    options = PostProcessingOptions()
    options.image_base_dir = patch_folder_base
    options.include_almost_detections = True
    options.num_images_to_sample = 7500
    options.confidence_threshold = 0.4
    options.almost_detection_confidence_threshold = options.confidence_threshold - 0.025
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
    
        
    #%% Clean up


#%% Interactive driver

if False:
    
    pass

    #%% Generate patches for one image
    
    input_folder_base = '/media/user/My Passport/2017-2019/01_JPGs'
    image_fn_relative = '2018/Replicate_2018-10-20/CAM2/CAM21601.JPG'
    image_name_base = input_folder_base 
    
    image_fn = os.path.join(input_folder_base,image_fn_relative)    
    assert os.path.isfile(image_fn)
    
    patch_folder = os.path.join(patch_folder_base,image_fn_relative)
    
    image_patch_info = extract_patches_for_image(image_fn,patch_folder,image_name_base)    
               

    #%% Perform NMS within each image
                
    with open(md_formatted_results_file,'r') as f:        
        md_results = json.load(f)
    
    in_place_nms(md_results)
    
    md_results_after_nms_file = md_formatted_results_file.replace('.json',
                                                                  '_nms.json')
    assert md_results_after_nms_file != md_formatted_results_file
    
    with open(md_results_after_nms_file,'w') as f:
        json.dump(md_results,f,indent=1)
                
    
    
    #%% Combine all the patch results to an image-level results set
        
    patch_results_file = md_formatted_results_file
    # patch_results_file = md_results_after_nms_file
    
    with open(patch_results_file,'r') as f:
        all_patch_results = json.load(f)
    
    # Map absolute paths to detections
    patch_fn_to_results = {}
    for im in all_patch_results['images']:
        abs_fn = os.path.join(patch_folder_base,im['file'])
        patch_fn_to_results[abs_fn] = im
                              
    image_fn = os.path.join(input_folder_base,image_fn_relative)
    assert os.path.isfile(image_fn)
            
    output_im = {}
    output_im['file'] = image_fn_relative
    output_im['detections'] = []
        
    pil_im = visutils.open_image(image_fn)
    assert pil_im.size[0] == expected_image_width
    assert pil_im.size[1] == expected_image_height
    
    image_w = pil_im.size[0]
    image_h = pil_im.size[1]
    
    assert image_patch_info['image_fn'] == image_fn
    
    patch_fn_to_patch_info = {}
    
    for patch_info in image_patch_info['patches']:
        patch_fn_to_patch_info[patch_info['patch_fn']] = patch_info
    
    assert len(patch_fn_to_patch_info) == len(patch_fn_to_results)
    
    # For each patch
    # patch_fn = list(patch_fn_to_patch_info.keys())[0]
    for patch_fn in patch_fn_to_patch_info.keys():
        
        patch_results = patch_fn_to_results[patch_fn]
        patch_info = patch_fn_to_patch_info[patch_fn]
        
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
            h_pixels = w_patch_relative * patch_h
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
    
    md_results_image_level = {}
    md_results_image_level['info'] = all_patch_results['info']
    md_results_image_level['detection_categories'] = all_patch_results['detection_categories']
    md_results_image_level['images'] = [output_im]
    
    in_place_nms(md_results_image_level)
    
    md_results_image_level_fn = os.path.join(project_dir,'md_results_image_level.json')
    with open(md_results_image_level_fn,'w') as f:
        json.dump(md_results_image_level,f,indent=1)


    #%% Render boxes on the original image

    from visualization import visualization_utils as visutils
    
    output_image_file = os.path.join(project_dir,'test.jpg')
    detections = md_results_image_level['images'][0]['detections']    
    
    detector_label_map = {}
    for category_id in yolo_category_id_to_name:
        detector_label_map[str(category_id)] = yolo_category_id_to_name[category_id]
        
    visutils.draw_bounding_boxes_on_file(input_file=image_fn,
                          output_file=output_image_file,
                          detections=detections,
                          confidence_threshold=0.2,
                          detector_label_map=detector_label_map, 
                          thickness=1, 
                          expansion=0)
    
    path_utils.open_file(output_image_file)
    
    