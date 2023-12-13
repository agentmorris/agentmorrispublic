########
#
# uidaho-lau-comparison-analysis.py
#
# Compare thresholds and false positive rates with and without TTA/RDE.
#
########

#%% Constants and imports

import os
import clipboard
import json

from tqdm import tqdm

base_folder_data = os.path.expanduser('~/data/idaho-lau')
base_folder_processing = os.path.expanduser('~/postprocessing/idaho-lau')

known_grouse_folder = os.path.join(base_folder_data,'KnownGrouse_2023_megadetector')
assert os.path.isdir(known_grouse_folder)

# md_results_all_tta_including_corrupted_images = os.path.join(base_folder_processing,
#  'idaho-lau-2023-10-13-v5a.0.0/combined_api_outputs/idaho-lau-2023-10-13-v5a.0.0_detections_combined.json')
md_results_all_tta = os.path.join(base_folder_processing,
  'idaho-lau-2023-10-13-v5a.0.0/combined_api_outputs/idaho-lau-2023-10-13-v5a.0.0_detections_combined_exist.json')
md_results_all_tta_filtered = os.path.join(base_folder_processing,
  'idaho-lau-2023-10-13-v5a.0.0/combined_api_outputs/idaho-lau-2023-10-13-v5a.0.0_detections_combined_exist.filtered_rde_0.050_0.850_15_0.200.json')

md_results_all_no_tta = os.path.join(base_folder_processing, 
  'idaho-lau-2023-11-09-no-aug-v5a.0.0/combined_api_outputs/idaho-lau-2023-11-09-no-aug-v5a.0.0_detections.json')
md_results_all_no_tta_filtered = os.path.join(base_folder_processing,
  'idaho-lau-2023-11-09-no-aug-v5a.0.0/combined_api_outputs/idaho-lau-2023-11-09-no-aug-v5a.0.0_detections.filtered_rde_0.050_0.850_8_0.500.json')

known_grouse_results_file_no_tta = os.path.join(base_folder_data,
                                                'KnownGrouse_2023_megadetector_no_tta.json')
known_grouse_results_file_with_tta = os.path.join(base_folder_data,
                                                  'KnownGrouse_2023_megadetector_tta.json')

target_recall = 0.95

md_animal_category = '1'

assert all([os.path.isfile(fn) for fn in (
    md_results_all_tta, md_results_all_tta_filtered,
    md_results_all_no_tta, md_results_all_no_tta_filtered)])


#%% Run with and without TTA on the known-grouse folder

cmd_no_tta = 'CUDA_VISIBLE_DEVICES=1 python run_detector_batch.py MDV5A "{}" "{}"'.format(known_grouse_folder,
                                                                     known_grouse_results_file_no_tta)
cmd_no_tta += ' --recursive --output_relative_filenames --quiet'

print(cmd_no_tta)
# clipboard.copy(cmd_no_tta)

print('')

yolo_working_folder = os.path.expanduser('~/git/yolov5')
cmd_with_tta = \
    'python run_inference_with_yolov5_val.py MDV5A "{}" "{}"'.format(
        known_grouse_folder,known_grouse_results_file_with_tta)
cmd_with_tta += ' --yolo_working_folder "{}" --device_string 1'.format(yolo_working_folder)

print(cmd_with_tta)
clipboard.copy(cmd_with_tta)


#%% Compare results with/without TTA

cmd = 'python api/batch_processing/postprocessing/compare_batch_results.py "{}" "{}" "{}" "{}"'.format(
    os.path.expanduser('~/idaho-lau/known-grouse-tta-comparison'),
    known_grouse_folder,
    known_grouse_results_file_no_tta,
    known_grouse_results_file_with_tta)

cmd += ' --detection_thresholds 0.08 0.08 '
cmd += ' --rendering_thresholds 0.08 0.08 '
cmd += ' --open_results'

print(cmd)
clipboard.copy(cmd)


#%% Choose confidence thresholds with/without TTA to hit the target recall

def find_threshold_for_target_recall(fn):
    """
    Given a .json results file [fn] in which all images contain animals, find the confidence
    threshold that considers [target_recall]*100 percent of them to be positive, by iterating
    over confidence thresholds.
    """
    
    with open(fn,'r') as f:
        d = json.load(f)

    max_confidences = []
    
    # Find the maximum confidence value for the "animal" category for each image
    # im = d['images'][0]
    for im in d['images']:
        
        # Only look at animal detections
        im['detections'] = [det for det in im['detections'] if det['category'] == md_animal_category]
        if len(im['detections']) == 0:
            max_confidences.append(0)
        else:
            confidence_values = [det['conf'] for det in im['detections']]
            max_confidences.append(max(confidence_values))
            
    # ...for each image
    
    # Sweep over all possible confidence thresholds, starting with 1.0, to find the 
    # highest threshold that still meets our target recall.  Stop iterating when we get
    # to our traget recall.
    threshold_increment = 0.001
    min_threshold = 0
    max_threshold = 1
    
    highest_threshold_meeting_target_recall = None
    
    threshold = max_threshold
    
    while(threshold > min_threshold):
        
        detections = [val for val in max_confidences if val > threshold]
        recall = len(detections) / len(max_confidences)
        if recall >= target_recall:
            highest_threshold_meeting_target_recall = threshold
            break
        
        threshold -= threshold_increment
    
    # ...for each possible threshold
    
    print('Threshold for {}: {:.3f}'.format(fn,highest_threshold_meeting_target_recall))
          
    return highest_threshold_meeting_target_recall

# ...def find_threshold_for_target_recall(...)

threshold_with_tta = find_threshold_for_target_recall(known_grouse_results_file_with_tta)
threshold_no_tta = find_threshold_for_target_recall(known_grouse_results_file_no_tta)

print('Threshold for TTA: {:.3f}'.format(threshold_with_tta))
print('Threshold no TTA: {:.3f}'.format(threshold_no_tta))


#%% Detection analysis

def count_detections(fn,threshold):
    """
    Count the number of images containing predicted animals in the MD results file
    [fn], using the confidence threshold [threshold].
    """
    
    with open(fn,'r') as f:
        md_results = json.load(f)
    
    n_detections = 0
    
    for im in tqdm(md_results['images']):
        
        # Skip failed images
        if 'detections' not in im or im['detections'] is None:
            continue
    
        # Only look at animal detections
        im['detections'] = [det for det in im['detections'] if det['category'] == md_animal_category]
        if len(im['detections']) == 0:
            continue
        confidence_values = [det['conf'] for det in im['detections']]
        if max(confidence_values) >= threshold:
            n_detections += 1

    # ...for each image
    
    print('For file {} at threshold {:.2f}, {} of {} images are detections'.format(
        os.path.basename(fn),threshold,n_detections,len(md_results['images'])))

count_detections(md_results_all_tta,threshold_with_tta)
count_detections(md_results_all_tta_filtered,threshold_with_tta)

count_detections(md_results_all_no_tta,threshold_no_tta)
count_detections(md_results_all_no_tta_filtered,threshold_no_tta)
