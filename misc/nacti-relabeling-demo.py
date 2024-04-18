########
#
# nacti-relabeling-demo.py
#
# Demonstrate the labelme box correction process on a folder of images.
#
# If you are running on Windows, you will need to run this notebook with admin
# priveleges to create symlinks.
#
########

#%% Imports and constants

import os
import json
import sys

# I ran MegaDetector before running this script; this is the results file.
md_results_file = r'c:\Users\dmorr/postprocessing\nacti\nacti-2024-04-17-v5a.0.0\combined_api_outputs\nacti-2024-04-17-v5a.0.0_detections.filtered_rde_0.100_0.850_50_0.300.json'

# This is where the images live; it should be the same folder on which you ran MegaDetector
relabeling_folder_base = r'c:\temp\nacti-labeling'

# Subfolders will be created within this folder; each subfolder will have the symlinks for a particular
# labeling chunk
symlink_folder = r'c:\temp\nacti-labeling-symlinks'

# Normalize paths
md_results_file = md_results_file.replace('\\','/')
relabeling_folder_base = relabeling_folder_base.replace('\\','/')
symlink_folder = symlink_folder.replace('\\','/')

# This is a hack I use because sometimes I run this process in WSL instead of native Windows
if sys.platform != 'win32':
    md_results_file = md_results_file.replace('c:/','/mnt/c/')
    relabeling_folder_base = relabeling_folder_base.replace('c:/','/mnt/c/')
    symlink_folder = symlink_folder.replace('c:/','/mnt/c/')

assert os.path.isfile(md_results_file)
assert os.path.isdir(relabeling_folder_base)

os.makedirs(symlink_folder,exist_ok=True)

# This defines the set of backup label files we generate from MD results at lower confidence thresholds
index_to_threshold = {
    1:0.1,
    2:0.05,
    3:0.01
}

# Support function for parsing our canonical file name structure
def get_base_filename(fn):
    
    fn = fn.replace('\\','/')
    basename = os.path.splitext(fn)[0]
    tokens = basename.split('.')
    if tokens[-1].startswith('alt'):
        assert fn.endswith('.json')
        basename = '.'.join(tokens[0:-1])    
    return basename    


#%% Convert MD results to labelme format with a default threshold

from api.batch_processing.postprocessing.md_to_labelme import md_to_labelme

_ = md_to_labelme(results_file=md_results_file,
                  image_base=relabeling_folder_base,
                  confidence_threshold=0.2,
                  overwrite=True,
                  extension_prefix='',
                  n_workers=10,
                  use_threads=False,
                  bypass_image_size_read=False,
                  verbose=True)


#%% Create alternative .json files based on MD results at lower thresholds

for index in index_to_threshold.keys():
    
    print('Generating alternative labels for index {} (threshold {})'.format(
        index,index_to_threshold[index]))
    
    md_to_labelme(results_file=md_results_file,
                  image_base=relabeling_folder_base,
                  confidence_threshold=index_to_threshold[index],
                  overwrite=True,
                  use_threads=False,
                  bypass_image_size_read=False,
                  extension_prefix='.alt-{}'.format(index),
                  n_workers=10)


#%% Enumerate files

from md_utils.path_utils import recursive_file_list

all_files_relative = recursive_file_list(relabeling_folder_base,
                                         return_relative_paths=True,
                                         convert_slashes=True,
                                         recursive=True)

print('Enumerated {} files'.format(len(all_files_relative)))


##%% Match .json files to images

from md_utils.path_utils import  find_image_strings
image_files_relative = find_image_strings(all_files_relative)
json_files = [fn for fn in all_files_relative if fn.endswith('.json')]
json_files = sorted(json_files)

print('Enumerated {} image files and {} .json files'.format(
    len(image_files_relative),len(json_files)))


##%% Group json files by the image they belong to

# We'll use this to create symlinks to every file that goes with each image in
# a chunk.

from collections import defaultdict
from tqdm import tqdm

image_file_base_to_json_files = defaultdict(list)

# json_file = json_files[0]
for json_file in tqdm(json_files):

    basename = get_base_filename(json_file)
    image_file_base_to_json_files[basename].append(json_file)


##%% Make sure every image has the right number of .json files

unlabeled_image_files = []

for image_file in tqdm(image_files_relative):    
    basename = get_base_filename(image_file)
    json_files_this_image = image_file_base_to_json_files[basename]
    assert len(json_files_this_image) == 4
    if len(json_files_this_image) == 0:
        unlabeled_image_files.append(image_file)

    
#%% Divide into chunks, create symlinks

from md_utils.ct_utils import split_list_into_fixed_size_chunks
from md_utils.path_utils import safe_create_link

batch_name = 'nacti-cxl'
max_images_per_chunk = 5000

chunks = split_list_into_fixed_size_chunks(image_files_relative,max_images_per_chunk)

print('Split images into {} chunks of {} images'.format(len(chunks),max_images_per_chunk))

chunk_folder_base = os.path.join(relabeling_folder_base,'symlinks-{}'.format(batch_name))
chunk_folders = []
error_files = []

# i_chunk = 0; chunk = chunks[i_chunk]
for i_chunk,chunk in enumerate(chunks):
    
    print('Creating symlinks for chunk {} of {}'.format(i_chunk,len(chunks)))

    chunk_folder_abs = os.path.join(chunk_folder_base,'chunk_{}'.format(
        str(i_chunk).zfill(3)))
    os.makedirs(chunk_folder_abs,exist_ok=True)
    chunk_folders.append(chunk_folder_abs)
    
    # Find matching files
    relative_files_this_chunk = []
    
    # i_image=0; image_file = chunk[i_image]
    for i_image,image_file in enumerate(chunk):
        
        # image_file_abs = os.path.join(training_images_resized_folder,image_file); open_file(image_file_abs)
        basename = get_base_filename(image_file)
        json_files_this_image = image_file_base_to_json_files[basename]
        
        # These are typically images that failed to load
        if len(json_files_this_image) == 0:
            print('Warning: no .json files for {}'.format(image_file))
            error_files.append(image_file)
            continue
        
        assert len(json_files_this_image) > 0
        relative_files_this_chunk.append(image_file)
        
        for json_file in json_files_this_image:
            relative_files_this_chunk.append(json_file)            
    
    # Create symlinks
    #
    # relative_file = relative_files_this_chunk[0]
    for relative_file in tqdm(relative_files_this_chunk):
        source_file_abs = os.path.join(relabeling_folder_base,relative_file)
        assert os.path.isfile(source_file_abs)
        target_file_abs = os.path.join(chunk_folder_abs,relative_file)
        os.makedirs(os.path.dirname(target_file_abs),exist_ok=True)
        safe_create_link(source_file_abs,target_file_abs)

# ...for each chunk

error_file_list_file = os.path.join(chunk_folder_base,'error_images.json')
print('Saving list of {} error images to {}'.format(len(error_files),error_file_list_file))
with open(error_file_list_file,'w') as f:
    json.dump(error_files,f,indent=1)


#%% Label one chunk

# Specifically, generate the command to start labelme, pointed at this chunk, and copy that
# command to the clipboard.

i_chunk = 0
resume = True

chunk_folder_abs = os.path.join(chunk_folder_base,'chunk_{}'.format(
    str(i_chunk).zfill(3)))
assert os.path.isdir(chunk_folder_abs)

flags = ['ignore','empty']

flag_file = os.path.join(chunk_folder_abs,'flags.txt')
with open(flag_file,'w') as f:
    for flag in flags:        
        f.write(flag + '\n')

last_updated_file = os.path.join(chunk_folder_abs,'labelme_last_updated.txt')
cmd = 'python labelme "{}" --labels animal,person,vehicle --linewidth 4 --last_updated_file "{}" --flags "{}"'.format(
    chunk_folder_abs,last_updated_file,flag_file)
if resume:
    cmd += ' --resume_from_last_update'
import clipboard; print(cmd); clipboard.copy(cmd)
