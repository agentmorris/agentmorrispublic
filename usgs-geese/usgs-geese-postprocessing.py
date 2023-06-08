########
#
# usgs-geese-postprocessing.py
#
# Stuff we do with model results after inference.
#
# * Generate patch-level preview pages given a set of image-level results.
#
# * Generate estimated image-level counts from image-level results.
#
########

#%% Constants and imports

import os
import json
import random

from tqdm import tqdm

import importlib
usgs_geese_inference = importlib.import_module('usgs-geese-inference')

from md_visualization import visualization_utils as vis_utils
from md_utils import path_utils

default_preview_confidence_thresholds = [0.4,0.5,0.6,0.7,0.8]


#%% Support functions

def intersects(box1,box2):
    """
    Determine whether two rectangles in x,y,w,h format intersect
    """
    
    # Convert to x1,y1,x2,y2 format
    r1 = [box1[0],box1[1],box1[0]+box1[2],box1[1]+box1[3]]
    r2 = [box2[0],box2[1],box2[0]+box2[2],box2[1]+box2[3]]
    
    if (r1[0]>=r2[2]) or (r1[2]<=r2[0]) or (r1[3]<=r2[1]) or (r1[1]>=r2[3]):
        return False
    else:
        return True


#%% Postprocessing functions

def patch_level_preview(image_level_results_file,image_folder_base,preview_folder,
                        n_patches=10000,preview_confidence_thresholds=None):
    """
    Given an image-level results file:
    
    * Map each image name to a list of detections
    * Randomly sample patches from that results file
    * For each sampled patch
      * Find all results within that patch, convert to patch-relative coordinates
      * Extract the patch and write to disk
    * Generate the preview page, including counts for each patch
    """
    
    #%%
    
    if preview_confidence_thresholds is None:
        preview_confidence_thresholds = default_preview_confidence_thresholds
        
    assert os.path.isfile(image_level_results_file)
    assert os.path.isdir(image_folder_base)
    os.makedirs(preview_folder,exist_ok=True)
    
    with open(image_level_results_file,'r') as f:
        image_level_results = json.load(f)
        
    print('Loaded results for {} images from {}'.format(
        len(image_level_results['images']),image_level_results_file))
            
    relative_image_fn_to_results = {}
    
    # Map image filenames to results for that image
    # im = image_level_results['images'][0]
    for im in image_level_results['images']:
        relative_image_fn_to_results[im['file']] = im
      
    # Convert relative coordinates to absolute coordinates
    # im = image_level_results['images'][0]
    for im in image_level_results['images']:
        
        # Coordinates are x/y/w/h, in normalized coordinates
        image_size = [usgs_geese_inference.expected_image_width,usgs_geese_inference.expected_image_height]
        
        detections_absolute = []
        # det = im['detections'][0]
        for det in im['detections']:
            assert 'bbox' in det and len(det['bbox']) == 4
            bbox_absolute = [
                det['bbox'][0] * image_size[0],
                det['bbox'][1] * image_size[1],
                det['bbox'][2] * image_size[0],
                det['bbox'][3] * image_size[1]
                ]
            det_absolute = {}
            det_absolute['bbox'] = bbox_absolute
            det_absolute['conf'] = det['conf']
            det_absolute['category'] = det['category']
            detections_absolute.append(det_absolute)
        
        im['detections_absolute'] = detections_absolute
        assert len(im['detections_absolute']) == len(im['detections'])
        
    # ...for each image        
    
    image_size = (usgs_geese_inference.expected_image_width,usgs_geese_inference.expected_image_height)
    patch_size = usgs_geese_inference.patch_size
    
    patch_boundaries = usgs_geese_inference.get_patch_boundaries(image_size,patch_size,patch_stride=None)
    print('Sampling from {} images, with {} candidate patches per image'.format(
        len(relative_image_fn_to_results),len(patch_boundaries)))

    # Generate a list of image/patch tuples to sample
    all_image_patch_tuples = []
    for i_image,fn in enumerate(relative_image_fn_to_results.keys()):
        for i_patch,patch_xy in enumerate(patch_boundaries):
            all_image_patch_tuples.append((fn,patch_xy))
            
    if len(all_image_patch_tuples) <= n_patches:
        sampled_patch_tuples = all_image_patch_tuples
    else:
        random.seed(0)
        sampled_patch_tuples = random.sample(all_image_patch_tuples,n_patches)
    
    patch_folder = os.path.join(preview_folder,'patches')
    os.makedirs(patch_folder,exist_ok=True)
    
    patch_level_results = {}
    patch_level_results['info'] = image_level_results['info']
    patch_level_results['detection_categories'] = image_level_results['detection_categories']
    patch_level_results['images'] = []
    
    # i_patch = 0; patch = sampled_patch_tuples[i_patch]
    #
    # TODO: parallelize this loop
    for i_patch,patch in tqdm(enumerate(sampled_patch_tuples),total=len(sampled_patch_tuples)):
    
        image_fn_relative = patch[0]
        image_fn_absolute = os.path.join(image_folder_base,image_fn_relative)
        assert os.path.isfile(image_fn_absolute)
        patch_xy = patch[1]
        
        # Generate a usable filename for this patch
        image_name = usgs_geese_inference.relative_path_to_image_name(image_fn_relative)
        patch_fn_relative = usgs_geese_inference.patch_info_to_patch_name(
            os.path.splitext(image_name)[0],patch_xy[0],patch_xy[1]) + '.jpg'
        patch_fn_absolute = os.path.join(patch_folder,patch_fn_relative)
        
        pil_im = vis_utils.open_image(image_fn_absolute)
        assert pil_im.size[0] == usgs_geese_inference.expected_image_width
        assert pil_im.size[1] == usgs_geese_inference.expected_image_height

        # Extract the patch
        _ = usgs_geese_inference.extract_patch_from_image(pil_im,patch_xy,
                                     patch_image_fn=patch_fn_absolute,
                                     patch_folder=None,image_name=None,overwrite=True)
        
        # Find detections that overlap with this patch, and convert to patch-relative,
        # normalized coordinates
        im_results = relative_image_fn_to_results[image_fn_relative]
        detections_this_image = im_results['detections_absolute']
        
        detections_this_patch = []
        
        # x/y/w/h
        r1 = [patch_xy[0],patch_xy[1],patch_size[0],patch_size[1]]
        
        # image_level_detection = detections_this_image[0]
        for image_level_detection in detections_this_image:
            
            r2 = image_level_detection['bbox']
            
            if intersects(r1,r2):
                
                patch_level_detection = {}
                patch_level_detection['conf'] = image_level_detection['conf']
                patch_level_detection['category'] = image_level_detection['category']
                
                # Convert to patch-relative absolute coordinates
                patch_bbox = [
                    image_level_detection['bbox'][0] - patch_xy[0],
                    image_level_detection['bbox'][1] - patch_xy[1],
                    image_level_detection['bbox'][2],
                    image_level_detection['bbox'][3]
                    ]
                    
                # Now normalize to patch-relative normalized coordinates
                patch_bbox[0] = patch_bbox[0]/patch_size[0]
                patch_bbox[1] = patch_bbox[1]/patch_size[1]
                patch_bbox[2] = patch_bbox[2]/patch_size[0]
                patch_bbox[3] = patch_bbox[3]/patch_size[1]
                
                patch_level_detection['bbox'] = patch_bbox
                                      
                detections_this_patch.append(patch_level_detection)
            
            # ...if this detection is within this patch
            
        # ...for every detection on this image
        
        # Add patch-specific results to patch_level_results['images']
        patch_im = {}
        patch_im['detections'] = detections_this_patch
        patch_im['file'] = patch_fn_relative
        patch_level_results['images'].append(patch_im)
        
    # ...for every sampled patch
    
    #%%
    
    # Write out a .json file with the patch-level results
    patch_level_results_file = os.path.join(preview_folder,'patch_level_results.json')
    with open(patch_level_results_file,'w') as f:
        json.dump(patch_level_results,f,indent=1)
    
    # Generate the preview page
    preview_page_folder = os.path.join(preview_folder,'preview_page')
    os.makedirs(preview_page_folder,exist_ok=True)
    
    from api.batch_processing.postprocessing.postprocess_batch_results import (
        PostProcessingOptions, process_batch_results)
    
    base_task_name = os.path.basename(image_level_results_file)
        
    for confidence_threshold in preview_confidence_thresholds:
        
        options = PostProcessingOptions()
        options.image_base_dir = patch_folder
        options.include_almost_detections = True
        options.num_images_to_sample = None # n_patches
        options.confidence_threshold = confidence_threshold
        options.almost_detection_confidence_threshold = options.confidence_threshold - 0.05
        options.ground_truth_json_file = None
        options.separate_detections_by_category = True
        # options.sample_seed = 0
        
        options.parallelize_rendering = False
        options.parallelize_rendering_n_cores = 16
        options.parallelize_rendering_with_threads = False
        
        output_base = os.path.join(preview_page_folder,
            base_task_name + '_{:.3f}'.format(options.confidence_threshold))
        
        os.makedirs(output_base, exist_ok=True)
        print('Processing to {}'.format(output_base))
        
        options.api_output_file = patch_level_results_file
        options.output_dir = output_base
        ppresults = process_batch_results(options)
        html_output_file = ppresults.output_html_file
        
        path_utils.open_file(html_output_file)
    
    # ...for each confidence threshold

    #%%
    
# ...def patch_level_preview()
    
def image_level_counting(image_level_results_file):
    """
    Given an image-level results file:
        
    * Count the number of occurrences in each class above a few different thresholds
    * Write the resulting counts to .csv
    """
    pass


#%% Interactive driver

if False:
    
    #%%
    
    image_level_results_file = os.path.expanduser('~/tmp/usgs-inference/image_level_results/home_user_data_usgs-geese_eval_images_md_results_image_level_nms.json')
    
    image_folder_base = os.path.expanduser('~/data/usgs-geese/eval_images')
    
    preview_folder_base = os.path.expanduser('~/tmp/usgs-inference/preview')
    preview_folder = os.path.join(preview_folder_base,
                                  os.path.splitext(os.path.basename(image_level_results_file))[0])
    
    

    