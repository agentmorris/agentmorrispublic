########
#
# usgs-geese-training-data-prep.py
#
# Given the COCO-formatted data created by usgs-data-review.py, cut the source
# data into patches, write those patches out as YOLO-formatted annotations, and
# copy files into train/test folders.
#
# Assumes a fixed patch size, and just slides this along each image looking for
# overlapping annotations.  In general there is no overlap between patches, but
# the rightmost column and bottommost row are shifted left and up to stay at the
# desired patch size, so geese near the right and bottom of each image will be 
# sampled twice.
#
########

#%% TODO

"""

* When I first ran this, I generated the train/val splits, but didn't write out the
  corresponding list of images, and had to jump through some hoops to recover that
  list from the patch file names.  If I ever run this again, write out the train/val
  image splits!

"""

#%% Constants and imports

import os
import json
import shutil
import random

from collections import defaultdict
from tqdm import tqdm

from md_visualization import visualization_utils as visutils

input_annotations_file = os.path.expanduser('~/data/usgs_geese.json')
input_base_folder = '/media/user/My Passport/2017-2019'
input_image_folder = os.path.join(input_base_folder,'01_JPGs')

debug_max_image = -1

if debug_max_image < 0:
    output_dir = os.path.expanduser('~/data/usgs-geese')
else:
    output_dir = os.path.expanduser('~/data/usgs-geese-mini-{}'.format(debug_max_image))

# This will contain *all* yolo-formatted patches
yolo_all_dir = os.path.join(output_dir,'yolo_all')
yolo_dataset_file = os.path.join(output_dir,'dataset.yaml')

# This file will be compatible with the MegaDetector repo's inference scripts
md_class_mapping_file = os.path.join(output_dir,'usgs-geese-md-class-mapping.json')

# Just the train/val subsets
yolo_train_dir = os.path.join(output_dir,'yolo_train')
yolo_val_dir = os.path.join(output_dir,'yolo_val')

patch_size = [1280,1280]

# Boxes in the input data have width and height, but AFAIK, they are basically arbitrary,
# we will use a fixed size for all boxes that's roughly the size of a bird, and then a bit
# extra.
#
# There's absolutely nothing special about this being a power of two, I was just picking
# a number around this range, and it seemed like good Karm to use 64.
box_size = [50,50]

# Should we clip boxes to image boundaries, even if we know the object extends
# into an adjacent image?
clip_boxes = True

# The "Mystery" class corresponds to objects that didn't have a label string; this
# is no longer a mystery: they were CountThings false positive that were manually
# removed.
category_names_to_exclude = ['Mystery']

# We are going to write multiple copies of the class list file, because
# YOLOv5 expects "classes.txt", but BoundingBoxEditor expects "object.data"
class_file_names = ['object.data','classes.txt']

# This will store a mapping from patches back to the original images
patch_metadata_file = 'patch_metadata.json'

val_image_fraction = 0.2

# If we have N images with annotations, we will choose hard_negative_fraction * N
# hard negatives, and from each of those we'll choose a number of patches equal to the
# average number of patches per image.
#
# YOLOv5 recommends 0%-10% hard negatives.
#
# As of the time I'm writing this comment, I just couldn't get confident in the unlabeled images
# as hard negatives, so leaving them out.  They appear to be slightly less likely to contain
# geese than the labeled images, but not much less likely.
hard_negative_fraction = 0.0 # 0.1

# The YOLO spec leaves it slightly ambiguous wrt whether empty annotation files are required/banned
# for hard negatives
write_empty_annotation_files_for_hard_negatives = True

random.seed(0)

patch_jpeg_quality = 95

# When we clip bounding boxes that are right on the edge of an image, clip them back
# by just a little more than we need to, to make BoundingBoxEditor happy
clip_epsilon_pixels = 0.01

do_tile_writes = True
do_image_copies = True

if not do_tile_writes:
    print('*** Warning: tile output disabled ***')

if not do_image_copies:
    print('*** Warning: image copying disabled ***')

# We will explicitly verify that images are actually this size
image_width = 8688
image_height = 5792


#%% Folder prep

assert os.path.isdir(input_base_folder)
assert os.path.isdir(input_image_folder)
assert os.path.isfile(input_annotations_file)

os.makedirs(output_dir,exist_ok=True)
os.makedirs(yolo_all_dir,exist_ok=True)
os.makedirs(yolo_train_dir,exist_ok=True)
os.makedirs(yolo_val_dir,exist_ok=True)

# For a while I was writing images and annotations to different folders, so 
# the code still allows them to be different.
dest_image_folder = yolo_all_dir
dest_txt_folder = yolo_all_dir

# Just in case...
os.makedirs(dest_image_folder,exist_ok=True)
os.makedirs(dest_txt_folder,exist_ok=True)
            

#%% Read source annotations

with open(input_annotations_file,'r') as f:
    d = json.load(f)
    
image_id_to_annotations = defaultdict(list)
for ann in d['annotations']:
    image_id_to_annotations[ann['image_id']].append(ann)

print('Read {} annotations for {} images'.format(
    len(d['annotations']),len(d['images'])))

# This is a list of relative paths to images with no annotations available; we'll
# sample from those to include some hard negatives
assert 'images_without_annotations' in d
print('Read a list of {} images without annotations'.format(len(d['images_without_annotations'])))

category_id_to_name = {c['id']:c['name'] for c in d['categories']}
category_name_to_id = {c['name']:c['id'] for c in d['categories']}
category_ids_to_exclude = []
for s in category_names_to_exclude:
    if s in category_name_to_id:
        category_ids_to_exclude.append(category_name_to_id[s])
    else:
        print("Warning: I'm supposed to ignore category {}, but this .json file doesn't"
              "have that category".format(s))


#%% Make sure all images are the same size

# im = d['images'][0]
for im in tqdm(d['images']):
    assert im['width'] == image_width and im['height'] == image_height
    

#%% Verify image ID uniqueness

image_ids = set()
for im in d['images']:
    assert im['id'] not in image_ids
    image_ids.add(im['id'])
    
    
#%% Define patch boundaries

# We'll use the same patch boundaries for all images

patch_start_positions = []

n_x_patches = image_width // patch_size[0]
if image_width - (patch_size[0]*n_x_patches) != 0:
    n_x_patches += 1

n_y_patches = image_height // patch_size[1]
if image_height - (patch_size[1]*n_y_patches) != 0:
    n_y_patches += 1

for i_x_patch in range(0,n_x_patches):
    
    x_start = patch_size[0] * i_x_patch
    x_end = x_start + patch_size[0] - 1
    if x_end >= image_width:
        assert i_x_patch == n_x_patches - 1
        overshoot = (x_end - image_width) + 1
        x_start -= overshoot
    for i_y_patch in range(0,n_y_patches):
        y_start = patch_size[1] * i_y_patch
        y_end = y_start + patch_size[1] - 1
        if y_end >= image_height:
            assert i_y_patch == n_y_patches - 1
            overshoot =  (y_end - image_height) + 1
            y_start -= overshoot
        patch_start_positions.append([x_start,y_start])

assert patch_start_positions[-1][0]+patch_size[0] == image_width
assert patch_start_positions[-1][1]+patch_size[1] == image_height
        

#%% Create YOLO-formatted patches

# TODO: this is trivially parallelizable

# This will be a dict mapping patch names (YOLO files without the extension)
# to metadata about their sources
patch_metadata_mapping = {}

image_ids_with_annotations = []
n_patches = 0
n_boxes = 0
n_clipped_boxes = 0
n_excluded_boxes = 0

def relative_path_to_image_name(rp):
    
    image_name = rp.lower().replace('/','_')
    assert image_name.endswith('.jpg')
    image_name = image_name.replace('.jpg','')
    return image_name
    
def patch_info_to_patch_name(image_name,patch_x_min,patch_y_min):
    patch_name = image_name + '_' + str(patch_x_min).zfill(4) + '_' + str(patch_y_min).zfill(4)
    return patch_name
    
# i_image = 0; im = d['images'][0]
for i_image,im in tqdm(enumerate(d['images']),total=len(d['images'])):

    if debug_max_image >= 0 and i_image > debug_max_image:
        break

    annotations = image_id_to_annotations[im['id']]
    
    # Skip images that have no annotations at all
    if len(annotations) == 0:
        continue
    
    image_fn = os.path.join(input_image_folder,im['file_name'])
    pil_im = visutils.open_image(image_fn)
    assert pil_im.size[0] == image_width
    assert pil_im.size[1] == image_height
    
    image_name = relative_path_to_image_name(im['file_name'])
    
    # The loop I'm about to do would be catastrophically inefficient if the numbers of annotations
    # per image were very large, but in practice, this whole script is going to be limited by
    # image I/O, and it's going to be negligible compared to training time anyway, so err'ing on
    # the side of readability, rather than using a more efficient box lookup system.
    
    n_patches_this_image = 0
    
    # Print patch positions, useful for debugging issues with individual patches
    if False:
        for i_p in range(0,len(patch_start_positions)):
            print('{}\t {}\t {}'.format(i_p,
                                        patch_start_positions[i_p][0],
                                        patch_start_positions[i_p][1]))
            
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

        # List of x/y/category center coordinates in absolute coordinates in the original image
        patch_box_centers = []
        
        # Find all the boxes that have at least N pixel overlap with this patch
        # i_ann = 0; ann = annotations[0]
        for i_ann,ann in enumerate(annotations):
            
            # Does this annotation overlap with this patch?
            
            # In the input annotations, boxes are x/y/w/h
            box_x_center = ann['bbox'][0] + (ann['bbox'][2]/2.0)
            box_y_center = ann['bbox'][1] + (ann['bbox'][3]/2.0)
            
            box_x_min = box_x_center - (box_size[0]/2.0)
            box_x_max = box_x_center + (box_size[0]/2.0)
            box_y_min = box_y_center - (box_size[1]/2.0)
            box_y_max = box_y_center + (box_size[1]/2.0)
            
            patch_contains_box = (patch_x_min < box_x_max and box_x_min < patch_x_max \
                                 and patch_y_min < box_y_max and box_y_min < patch_y_max)
            
            if patch_contains_box:
                
                # Is this a category we're ignoring?
                if ann['category_id'] in category_ids_to_exclude:
                    n_excluded_boxes += 1
                    continue                
                
                n_boxes += 1
                patch_box_centers.append([box_x_center,box_y_center,ann['category_id']])
        
        # ...for each annotation
        
        # Don't write out patches with no matching annotations
        if len(patch_box_centers) == 0:
            continue
        
        n_patches_this_image += 1
        
        patch_name = patch_info_to_patch_name(image_name,patch_x_min,patch_y_min)
        patch_image_fn = os.path.join(dest_image_folder,patch_name + '.jpg')
        patch_ann_fn = os.path.join(dest_txt_folder,patch_name + '.txt')
        
        # Write out patch image
        if (do_tile_writes):
            patch_im.save(patch_image_fn,quality=patch_jpeg_quality)
        
        # Create YOLO annotations
        
        yolo_boxes_this_patch = []
        
        # xy = patch_boxes[0]
        for xy in patch_box_centers:
            
            x_center_absolute_original = xy[0]
            y_center_absolute_original = xy[1]
            box_w_absolute = box_size[0]
            box_h_absolute = box_size[1]
            
            x_center_absolute_patch = x_center_absolute_original - patch_x_min
            y_center_absolute_patch = y_center_absolute_original - patch_y_min
            
            assert (1 + patch_x_max - patch_x_min) == patch_size[0]
            assert (1 + patch_y_max - patch_y_min) == patch_size[1]
            
            x_center_relative = x_center_absolute_patch / patch_size[0]
            y_center_relative = y_center_absolute_patch / patch_size[1]
            
            box_w_relative = box_w_absolute / patch_size[0]
            box_h_relative = box_h_absolute / patch_size[1]
            
            if clip_boxes:
                
                clipped_box = False
                clip_epsilon_relative = clip_epsilon_pixels / patch_size[0]
                
                box_right = x_center_relative + (box_w_relative / 2.0)                    
                if box_right > 1.0:
                    clipped_box = True
                    overhang = box_right - 1.0
                    box_w_relative -= overhang
                    x_center_relative -= ((overhang / 2.0) + clip_epsilon_relative)

                box_bottom = y_center_relative + (box_h_relative / 2.0)                                        
                if box_bottom > 1.0:
                    clipped_box = True
                    overhang = box_bottom - 1.0
                    box_h_relative -= overhang
                    y_center_relative -= ((overhang / 2.0) + clip_epsilon_relative)
                
                box_left = x_center_relative - (box_w_relative / 2.0)
                if box_left < 0.0:
                    clipped_box = True
                    overhang = abs(box_left)
                    box_w_relative -= overhang
                    x_center_relative += ((overhang / 2.0) + clip_epsilon_relative)
                    
                box_top = y_center_relative - (box_h_relative / 2.0)
                if box_top < 0.0:
                    clipped_box = True
                    overhang = abs(box_top)
                    box_h_relative -= overhang
                    y_center_relative += ((overhang / 2.0) + clip_epsilon_relative)
                    
                if clipped_box:
                    n_clipped_boxes += 1
            
            # ...if we're clipping boxes
            
            # YOLO annotations are category x_center y_center w h
            yolo_box = [ann['category_id'],
                        x_center_relative, y_center_relative, 
                        box_w_relative, box_h_relative]
            yolo_boxes_this_patch.append(yolo_box)
            
        # ...for each box
        
        patch_metadata = {
            'patch_name':patch_name,
            'original_image_id':im['id'],
            'patch_x_min':patch_x_min,
            'patch_y_min':patch_y_min,
            'patch_x_max':patch_x_max,
            'patch_y_max':patch_y_max,
            'hard_negative':False
            }
                    
        patch_metadata_mapping[patch_name] = patch_metadata
        
        with open(patch_ann_fn,'w') as f:
            for yolo_box in yolo_boxes_this_patch:
                f.write('{} {} {} {} {}\n'.format(
                    yolo_box[0],yolo_box[1],yolo_box[2],yolo_box[3],yolo_box[4]))            
        
    # ...for each patch
    
    n_patches += n_patches_this_image
    
    if n_patches_this_image > 0:
        image_ids_with_annotations.append(im['id'])
    
# ...for each image

# We should only have processed each image once
assert len(image_ids_with_annotations) == len(set(image_ids_with_annotations))

n_images_with_annotations = len(image_ids_with_annotations)

print('Processed {} boxes ({} clipped) ({} excluded) from {} patches on {} images'.format(
    n_boxes,n_clipped_boxes,n_excluded_boxes,n_patches,n_images_with_annotations))

annotated_image_ids = set([p['original_image_id'] for p in patch_metadata_mapping.values()])
assert len(annotated_image_ids) == n_images_with_annotations


#%% Write out class list and metadata mapping

print('Generating class lists')

# Write the YOLO-formatted class file
#
# ...possibly more than once, to satisfy a couple of conventions on 
# what it's supposed to be called.
for class_file_name in class_file_names:
    class_file_full_path = os.path.join(yolo_all_dir,class_file_name)
    with open(class_file_full_path, 'w') as f:
        print('Writing class list to {}'.format(class_file_full_path))
        for i_class in range(0,len(category_id_to_name)):
            # Category IDs should range from 0..N-1
            assert i_class in category_id_to_name
            f.write(category_id_to_name[i_class] + '\n')

patch_metadata_file_full_path = os.path.join(yolo_all_dir,patch_metadata_file)
with open(patch_metadata_file_full_path,'w') as f:
    json.dump(patch_metadata_mapping,f,indent=2)
    
md_category_mapping = {}
for category_id in category_id_to_name:
    md_category_mapping[str(category_id)] = category_id_to_name[category_id]
with open(md_class_mapping_file,'w') as f:
    json.dump(md_category_mapping,f,indent=1)


#%% Sample hard negatives

n_hard_negative_source_images = round(hard_negative_fraction*n_images_with_annotations)
average_patches_per_image = n_patches / n_images_with_annotations

candidate_hard_negatives = []
n_bypassed_candidates = 0
non_empty_images = set(d['images_that_might_not_be_empty'])
for s in tqdm(d['images_without_annotations']):
    if s in non_empty_images:
        n_bypassed_candidates += 1
    else:
        candidate_hard_negatives.append(s)
assert n_bypassed_candidates > 0        
print('Bypassed {} hard negative candidates because they might not be empty'.format(
    n_bypassed_candidates))
        
hard_negative_source_images = random.sample(candidate_hard_negatives,
                                             k=n_hard_negative_source_images)
assert len(hard_negative_source_images) == len(set(hard_negative_source_images))

# For each hard negative source image
#
# image_fn_relative = hard_negative_source_images[0]
for image_fn_relative in tqdm(hard_negative_source_images):
    
    image_id = image_fn_relative.replace('/','_')
    assert image_id not in annotated_image_ids
    
    # Sample random patches from this image
    sampled_patch_start_positions = random.sample(patch_start_positions,
                                                   k=round(average_patches_per_image))
    
    # For each sampled patch
    # i_patch = 0; patch_xy = sampled_patch_start_positions[i_patch]
    for i_patch,patch_xy in enumerate(sampled_patch_start_positions):
        
        patch_x_min = patch_xy[0]
        patch_y_min = patch_xy[1]
        patch_x_max = patch_x_min + patch_size[0] - 1
        patch_y_max = patch_y_min + patch_size[1] - 1
    
        patch_im = pil_im.crop((patch_x_min,patch_y_min,patch_x_max+1,patch_y_max+1))
        assert patch_im.size[0] == patch_size[0]
        assert patch_im.size[1] == patch_size[1]
        
        image_name = relative_path_to_image_name(image_fn_relative)
        
        patch_name = image_name + '_' + str(patch_x_min).zfill(4) + '_' + str(patch_y_min).zfill(4)
        patch_image_fn = os.path.join(dest_image_folder,patch_name + '.jpg')
        patch_ann_fn = os.path.join(dest_txt_folder,patch_name + '.txt')
        
        # Write out patch image
        patch_im.save(patch_image_fn,quality=patch_jpeg_quality)
        
        # Write empty annotation file
        if write_empty_annotation_files_for_hard_negatives:
            with open(patch_ann_fn,'w') as f:
                pass
            assert os.path.isfile(patch_ann_fn)
        
        # Add to patch metadata list
        patch_name = patch_info_to_patch_name(image_name,patch_x_min,patch_y_min)
        assert patch_name not in patch_metadata_mapping
        patch_metadata = {
            'patch_name':patch_name,
            'original_image_id':image_id,
            'patch_x_min':patch_x_min,
            'patch_y_min':patch_y_min,
            'patch_x_max':patch_x_max,
            'patch_y_max':patch_y_max,
            'hard_negative':True
            }
        patch_metadata_mapping[patch_name] = patch_metadata
    
    # ...for each patch

# ...for each hard negative image    


#%% Measure folder size

import humanfriendly
from pathlib import Path

annotated_image_ids = set([p['original_image_id'] for p in patch_metadata_mapping.values()])

root_directory = Path(yolo_all_dir)
output_size_bytes = sum(f.stat().st_size for f in root_directory.glob('**/*') if f.is_file())

print('Output size for {} images is {}'.format(
    len(annotated_image_ids),humanfriendly.format_size(output_size_bytes)))

projected_size_bytes = (len(d['images']) / len(annotated_image_ids)) * output_size_bytes
print('Projected size for all {} images is {}'.format(
    len(d['images']),humanfriendly.format_size(projected_size_bytes)))


#%% Split image IDs into train/val

patch_metadata_file_full_path = os.path.join(yolo_all_dir,patch_metadata_file)
with open(patch_metadata_file_full_path,'r') as f:
    patch_metadata = json.load(f)

all_image_ids = set()
for patch_name in patch_metadata.keys():
    patch_info = patch_metadata[patch_name]
    all_image_ids.add(patch_info['original_image_id'])
print('Found {} unique image IDs for {} patches'.format(
    len(all_image_ids),len(patch_metadata)))

all_image_ids = list(all_image_ids)

n_val_image_ids = int(val_image_fraction*len(all_image_ids))

# This call to random.seed() was added *after* the original train/val split.
random.seed(0)
val_image_ids = random.sample(all_image_ids,k=n_val_image_ids)


#%% Copy images to train/val folders

train_patch_names = []
val_patch_names = []

# For each patch
for patch_name in tqdm(patch_metadata.keys(),total=len(patch_metadata)):
    
    patch_info = patch_metadata[patch_name]
    
    # Make sure we have annotation/image files for this patch
    source_image_path = os.path.join(yolo_all_dir,patch_name + '.jpg')
    source_ann_path = os.path.join(yolo_all_dir,patch_name + '.txt')
    
    assert os.path.isfile(source_image_path)
    assert os.path.isfile(source_ann_path)
    
    # Copy to the place it belongs
    if patch_info['original_image_id'] in val_image_ids:
        val_patch_names.append(patch_name)
        target_folder = yolo_val_dir
    else:
        train_patch_names.append(patch_name)
        target_folder = yolo_train_dir
    
    target_image_path = os.path.join(target_folder,os.path.basename(source_image_path))
    target_ann_path = os.path.join(target_folder,os.path.basename(source_ann_path))
    
    if do_image_copies:
        shutil.copyfile(source_image_path,target_image_path)
        shutil.copyfile(source_ann_path,target_ann_path)
    
# ...for each patch        

print('\nCopied {} train patches, {} val patches'.format(
    len(train_patch_names),len(val_patch_names)))


#%% Generate the YOLO training dataset file

# Read class names
class_file_path = os.path.join(yolo_all_dir,'classes.txt')
with open(class_file_path,'r') as f:
    class_lines = f.readlines()
class_lines = [s.strip() for s in class_lines]    
class_lines = [s for s in class_lines if len(s) > 0]

# Write dataset.yaml
with open(yolo_dataset_file,'w') as f:
    
    yolo_train_folder_relative = os.path.relpath(yolo_train_dir,output_dir)
    yolo_val_folder_relative = os.path.relpath(yolo_val_dir,output_dir)
    
    f.write('# Train/val sets\n')
    f.write('path: {}\n'.format(output_dir))
    f.write('train: {}\n'.format(yolo_train_folder_relative))
    f.write('val: {}\n'.format(yolo_val_folder_relative))
    
    f.write('\n')
    
    f.write('# Classes\n')
    f.write('names:\n')
    for i_class,class_name in enumerate(class_lines):
        f.write('  {}: {}\n'.format(i_class,class_name))


#%% Prepare simlinks for BoundingBoxEditor

# ...so it can appear that images and labels are in separate folders

def safe_create_link(link_exists,link_new):
    
    if not os.path.exists(link_new):
        os.symlink(link_exists,link_new)


def create_virtual_yolo_dirs(yolo_base_dir):
    
    images_dir = os.path.join(yolo_base_dir,'images')
    labels_dir = os.path.join(yolo_base_dir,'labels')
    
    os.makedirs(images_dir,exist_ok=True)
    os.makedirs(labels_dir,exist_ok=True)
    
    files = os.listdir(os.path.join(yolo_all_dir))
    source_images = [fn for fn in files if fn.lower().endswith('.jpg')]
    source_labels = [fn for fn in files if fn.lower().endswith('.txt') and fn != 'classes.txt']
    
    # fn = source_images[0]
    for fn in source_images:
        link_exists = os.path.join(yolo_all_dir,fn)
        link_new = os.path.join(images_dir,fn)
        safe_create_link(link_exists,link_new)
    for fn in source_labels:
        link_exists = os.path.join(yolo_all_dir,fn)
        link_new = os.path.join(labels_dir,fn)
        safe_create_link(link_exists,link_new)
    
    link_exists = os.path.join(yolo_all_dir,'object.data')        
    link_new = os.path.join(labels_dir,'object.data')
    safe_create_link(link_exists,link_new)

create_virtual_yolo_dirs(yolo_all_dir)
create_virtual_yolo_dirs(yolo_train_dir)
create_virtual_yolo_dirs(yolo_val_dir)
