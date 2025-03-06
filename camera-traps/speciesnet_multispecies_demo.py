"""

speciesnet_multispecies_demo.py

Run the SpeciesNet classifier separately on each crop in a folder of images, then
visualize the results.

Uses the megadetector-utils package for a bunch of image and results files manipulation, 
and launches the run_model script from the speciesnet package:

pip install speciesnet
pip install megadetector-utils

"""

#%% Imports and constants

import os

from megadetector.utils.process_utils import execute_and_print
from megadetector.postprocessing.create_crop_folder import create_crop_folder
from megadetector.utils.wi_utils import generate_predictions_json_from_md_results
from megadetector.utils.wi_utils import generate_md_results_from_predictions_json
from megadetector.postprocessing.create_crop_folder import crop_results_to_image_results

image_folder = '/mnt/g/temp/multispecies-test-images'
working_folder = '/mnt/g/temp/multispecies-test-cache'
crop_folder = os.path.join(working_folder,'crops')

country_code = 'ZAF'

detector_results_file = \
    os.path.join(working_folder,'detector-results.json')
detector_results_file_md_format = \
    os.path.join(working_folder,'detector-results-md-format.json')
detector_results_with_detection_ids = \
    os.path.join(working_folder,'detector-results-md-format-with-ids.json')
detector_results_file_for_cropped_images = \
    os.path.join(working_folder,'detector-results-for-cropped-images.json')
detector_results_file_for_cropped_images_speciesnet_format = \
    os.path.join(working_folder,'detector-results-for-cropped-images-speciesnet-format.json')

classifier_results_file = \
    os.path.join(working_folder,'classifier-results.json')
classifier_results_file_md_format = \
    os.path.join(working_folder,'classifier-results-md-format.json')

ensemble_output_file = \
    os.path.join(working_folder,'ensemble-results.json')
ensemble_output_file_md_format = \
    os.path.join(working_folder,'ensemble-results-md-format.json')

multispecies_results_md_format = \
    os.path.join(working_folder,'multispecies-results-md-format.json')

os.makedirs(working_folder,exist_ok=True)
os.makedirs(crop_folder,exist_ok=True)


#%% Run all the scripts
    
# Run detector
cmd = 'python -m speciesnet.scripts.run_model --detector_only '
cmd += f' --folders {image_folder} --predictions_json {detector_results_file}'
execute_and_print(cmd)

# Convert output file to MegaDetector/Timelapse format
cmd = 'python -m speciesnet.scripts.speciesnet_to_md'
cmd += f' {detector_results_file} {detector_results_file_md_format} --base_folder {image_folder}/'
execute_and_print(cmd)

# Create a new folder of images, where each image is a crop from the detection results that
# exceeds a confidence threshold.
# 
# This maintains the folder structure from the original images.  The "crops_output_file"
# will be a file of detections for the crops, where each image just has a single detection
# that's the entire image.  The purpose of that file is to attach detection category
# and confidence information to each cropped image.
create_crop_folder(input_file=detector_results_file_md_format,
                   input_folder=image_folder,
                   output_folder=crop_folder,
                   output_file=detector_results_with_detection_ids,
                   crops_output_file=detector_results_file_for_cropped_images)

# Run the classifier on the folder of crops (don't supply any detections, it will 
# just process the images, including resizing)
cmd = 'python -m speciesnet.scripts.run_model --classifier_only --bypass_prompts'
cmd += f' --folders {crop_folder} --predictions_json {classifier_results_file}'
execute_and_print(cmd)

# Convert the detection results for the crops (still with one box per crop) to SpeciesNet format
generate_predictions_json_from_md_results(md_results_file=detector_results_file_for_cropped_images,
    predictions_json_file=detector_results_file_for_cropped_images_speciesnet_format,
    base_folder=crop_folder + '/')

# Run the ensemble on the folder of crops
cmd = 'python -m speciesnet.scripts.run_model --ensemble_only'
cmd += f' --folders {crop_folder}'
cmd += f' --predictions_json {ensemble_output_file}'
cmd += f' --classifications_json {classifier_results_file}'
cmd += f' --detections_json {detector_results_file_for_cropped_images_speciesnet_format}'

if country_code is not None:
    cmd += f' --country {country_code}'
execute_and_print(cmd)

# Convert the ensmeble results file back to Timelapse/MD format
generate_md_results_from_predictions_json(predictions_json_file=ensemble_output_file,
                                          md_results_file=ensemble_output_file_md_format,
                                          base_folder=crop_folder + '/')

# Map the results back to the original images
crop_results_to_image_results(
    image_results_file_with_crop_ids=detector_results_with_detection_ids,
    crop_results_file=ensemble_output_file_md_format,
    output_file=multispecies_results_md_format)


#%% Visualize results

from megadetector.visualization.visualize_detector_output import visualize_detector_output
preview_folder = os.path.join(working_folder,'preview')
visualize_detector_output(detector_output_path=multispecies_results_md_format,
                          out_dir=preview_folder,
                          images_dir=image_folder,
                          confidence_threshold=0.15,
                          sample=1000,
                          output_image_width=800,
                          html_output_file=os.path.join(preview_folder,'index.html'),
                          parallelize_rendering=True)
